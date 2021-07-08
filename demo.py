import argparse
from PIL import Image

from matplotlib import pyplot as plt
from mano_train.demo.attention import AttentionHook
from handobjectdatasets.queries import BaseQueries, TransQueries

import cv2

from mano_train.exputils import argutils


from detection.detection import detection_init, detection
from multiprocessing import Process
from crop import crop
from mano_train.demo.preprocess import prepare_input, preprocess_frame
import numpy as np
import ray
from mano_train.netscripts.reload import reload_ray_model
import os, pickle, torch
import time
from handobjectdatasets.viz2d import visualize_joints_2d_cv2
from copy import deepcopy
from mano_train.visualize import displaymano
from mano_train.modelutils import modelio


def forward_pass_3d(input_image, pred_obj=True, left=True):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = ["left" if left else "right"]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample["root"] = "wrist"
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones(
            (1, 600, 3)
        ).float()
    #print(sample)

    return sample

#@ray.remote(num_cpus=4, max_calls=1)
def plot(hand, output, fig):

    hand_idx, hand_crop, left = hand

    # Pose Estimation (L-only)
    if left:
        inpimage = deepcopy(hand_crop)
    else:
        inpimage = deepcopy(np.flip(hand_crop, axis=1))

    if "joints2d" in output:
        joints2d = output["joints2d"]
        pose = visualize_joints_2d_cv2(
            inpimage, joints2d.cpu().detach().numpy()[0]
        )

    if left: 
        pose = cv2.flip(inpimage, 1)
        cv2.imshow(f"Hand #{hand_idx} Pose", pose)
    
    # Mesh Reconstruction
    verts = output["verts"].cpu().detach().numpy()[0]
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    displaymano.add_mesh(ax, verts, faces, flip_x=left)
    if "objpoints3d" in output:
        objverts = output["objpoints3d"].cpu().detach().numpy()[0]
        displaymano.add_mesh(
            ax, objverts, output["objfaces"], flip_x=left, c="r"
        )

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    
    cv2.imshow(f"Hand #{hand_idx} Mesh", buf)

if __name__ == "__main__":
    ray.init()
    print(ray.get_gpu_ids())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint",
        default="release_models/obman/checkpoint.pth.tar",
        required=True
    )
    parser.add_argument("--video_path", help="Path to video")
    parser.add_argument('--checksession', dest='checksession',
                      help='Checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='Checkepoch to load network',
                      default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='Checkpoint to load network',
                      default=90193, type=int, required=True)
    parser.add_argument('--workers', dest='workers',
                      help='Number of workers to initialize',
                      default=3, type=int,)                  
    args = parser.parse_args()
    argutils.print_args(args)


    # Load model options
    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)
    
    # Load faces of hand
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    # Initialize network
    fasterRCNN = detection_init(args.checksession, args.checkepoch, args.checkpoint)

    # Initialize stream from camera
    if args.video_path is None:
        # Read from webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)

    if cap is None:
        raise RuntimeError("OpenCV could not use webcam")

    print(" ------------------- Load 3D Mesh Model Weights ------------------- \n")
    weights = modelio.load_state_dict(args.resume)
    weights_id = ray.put(weights)

    print(" ------------------- Start Ray Multiprocessing Workers ------------------- \n")

    HandNets = [reload_ray_model(args.resume, opts, weights_id) for i in range(args.workers)]
    HandNets_id = ray.put(HandNets)
    
    #attention_hands = [AttentionHook(ray.get(model.get_base_net.remote())) for model in HandNets]
    figs = [plt.figure(figsize=(4, 4)) for i in range(args.workers)]

    while True:
        for fig in figs:
            fig.clf()

        ret, frame = cap.read()
        cv2.imshow("orig", frame)
        if not ret:
            raise RuntimeError("OpenCV could not load frame")
        total_tic = time.time()
        
        hand_dets = detection(frame, fasterRCNN)
        if hand_dets is not None:
            # Preprocess and crop hands
            hand_dets = [(hand_idx + 1, hand_dets[i, :]) for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))) ]
            hands = [(hand_idx, crop(frame, det, 1.2), det[-1]) for hand_idx, det in hand_dets]
            # [
            #     cv2.imshow(f"Hand #{hand_idx}", frame)
            #     for hand_idx, frame, side in hands
            # ]
            hands = [(hand_idx, cv2.resize(preprocess_frame(frame), (256, 256)), not bool(side)) for hand_idx, frame, side in hands]
            hands_input = [(hand_idx, prepare_input(frame, flip_left_right=not side,), side) for hand_idx, frame, side in hands]


            samples = [
                forward_pass_3d(hand, left=side)
                for hand_idx, hand, side in hands_input
            ]

            results= ray.get([HandNets[i%args.workers].forward.remote(samples[i], no_loss=True) for i in range(len(samples))])
            
            for i in range(len(results)): plot(hands[i], results[i][1], figs[i])

        cv2.waitKey(1)


    cap.release()
    cv2.destroyAllWindows()
