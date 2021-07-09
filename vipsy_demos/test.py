import argparse

from matplotlib import pyplot as plt
from mano_train.demo.attention import AttentionHook
from handobjectdatasets.queries import BaseQueries, TransQueries

import cv2

from mano_train.exputils import argutils

from detection.detection import detection_init, detection
from multiprocessing import Process
from crop import crop
from mano_train.demo.preprocess import preprocess_frame
import numpy as np
import ray
from mano_train.netscripts.reload import reload_ray_model
import os, pickle, torch
import time
from handobjectdatasets.viz2d import visualize_joints_2d_cv2
from copy import deepcopy
from mano_train.visualize import vispy_displaymano

from vispy import plot as vp
from vispy import scene
from vispy import app, gloo, visuals, io, geometry


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

@ray.remote(num_cpus=5, max_calls=1)
def plot(hands, output, i):
    # fig = vp.Fig(size=(4, 4), show=False)
    canvas = scene.SceneCanvas(keys='interactive', always_on_top=True)
    left = hands[i][2]
    hand_crop = hands[i][1]
    hand_idx = hands[i][0]

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
    
    # Mesh Reconstruction
    verts = output["verts"].cpu().detach().numpy()[0]
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'

    vispy_displaymano.add_mesh(view, verts, faces, flip_x=left)
    if "objpoints3d" in output:
        objverts = output["objpoints3d"].cpu().detach().numpy()[0]
        vispy_displaymano.add_mesh(
            view, objverts, output["objfaces"], flip_x=left, c="r"
        )

    canvas.show()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    cv2.imshow(f"Hand #{hand_idx} Pose", pose)

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

    print(" ------------------- Start Ray Multiprocessing Workers ------------------- \n")

    HandNets = [reload_ray_model(args.resume, opts) for i in range(args.workers)]
    HandNets_id = ray.put(HandNets)
    
    attention_hands = [AttentionHook(model.get_base_net.remote()) for model in HandNets]

    while True:
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
            hands = [(hand_idx, preprocess_frame(frame), not bool(side)) for hand_idx, frame, side in hands]

            hands_id = ray.put(hands)

            results_id = ray.wait([
                forward_pass_3d.remote(HandNets_id, hands_id, i)
                for i in range(len(hands))
            ])

            # clear graph
            # view.add(surface)
            # surface.parent = None
            # view.add(surface)
            
            plt.clf()
            [
                plot(hands_id, results_id[i], i)
                for i in range(len(results_id))
            ]


    cap.release()
    cv2.destroyAllWindows()
