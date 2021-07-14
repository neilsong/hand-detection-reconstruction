import argparse

from matplotlib import pyplot as plt
from handobjectdatasets.queries import BaseQueries, TransQueries

import cv2

from mano_train.exputils import argutils


from detection.detection import detection_init, detection, get_state_dict
from multiprocessing import Process
from crop import crop
from mano_train.demo.preprocess import prepare_input, preprocess_frame
import numpy as np
import ray
from mano_train.netscripts.reload import reload_ray_model
import os, pickle
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

def addtxt(frame, meta, top):
    hand_idx, side = meta
    white = np.zeros([20, int(frame_h/2),3],dtype=np.uint8)
    white.fill(255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{'Left' if side else 'Right'} #{hand_idx}"

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 0.47, 1)[0]

    # get coords based on boundary
    textX = (white.shape[1] - textsize[0]) / 2
    textY = (white.shape[0] + textsize[1]) / 2

    # add text centered on image
    cv2.putText(white, text, (int(textX), int(textY) ), font, 0.47, (0, 0, 0), 1, cv2.LINE_AA)

    text_white = cv2.cvtColor(white, cv2.COLOR_RGB2RGBA)

    return cv2.vconcat([text_white, frame] if top else [frame, text_white])


def createframe(**kwargs):
    if kwargs['mode']==0:
        white = np.zeros([h, w-frame_w,3],dtype=np.uint8)
        white.fill(255)
        # white = cv2.cvtColor(white, cv2.COLOR_RGB2RGBA)
        return cv2.hconcat([kwargs['frame'], white])
    elif kwargs['mode']==1:
        vert_meshes = [cv2.cvtColor(kwargs['frame'], cv2.COLOR_RGB2RGBA)]
        for i in range(0,len(kwargs['meshes']),2):
            top = kwargs['meshes'][i]
            bot = kwargs['meshes'][i+1]
            top = addtxt(top, kwargs['meta'][i], True)
            bot = addtxt(bot, kwargs['meta'][i+1], False)
            
            vert_meshes.append(cv2.vconcat([top, bot]))
        if w-frame_w - int(len(kwargs['meshes'])/2):
            rest = np.zeros([h, w-frame_w - int(frame_h/2) * int(len(kwargs['meshes'])/2),3],dtype=np.uint8)
            rest.fill(255)
            rest = cv2.cvtColor(rest, cv2.COLOR_RGB2RGBA)
            vert_meshes.append(rest)
        return cv2.hconcat(vert_meshes)
    elif kwargs['mode']==2:
        vert_meshes = [cv2.cvtColor(kwargs['frame'], cv2.COLOR_RGB2RGBA)]
        last = len(kwargs['meshes']) - 1
        for i in range(0,last,2):
            top = kwargs['meshes'][i]
            bot = kwargs['meshes'][i+1]
            top = addtxt(top, kwargs['meta'][i], True)
            bot = addtxt(bot, kwargs['meta'][i+1], False)
            
            vert_meshes.append(cv2.vconcat([top, bot]))
        white =  np.zeros([int(frame_h/2), int(frame_h/2),3], dtype=np.uint8)
        white.fill(255)
        white = cv2.cvtColor(white, cv2.COLOR_RGB2RGBA)
        vert_meshes.append(cv2.vconcat([addtxt(kwargs['meshes'][last], kwargs['meta'][last], True), white]))
        if w-frame_w - int(len(kwargs['meshes'])/2 + 1):
            rest = np.zeros([h, w-frame_w - int(frame_h/2) * int(len(kwargs['meshes'])/2 + 1),3],dtype=np.uint8)
            rest.fill(255)
            rest = cv2.cvtColor(rest, cv2.COLOR_RGB2RGBA)
            vert_meshes.append(rest)
        return cv2.hconcat(vert_meshes)

#@ray.remote(num_cpus=4, max_calls=1)
def plot(hand, output, fig):

    hand_idx, hand_crop, left = hand

    # Pose Estimation (L-only)
    # if left:
    #     inpimage = deepcopy(hand_crop)
    # else:
    #     inpimage = deepcopy(np.flip(hand_crop, axis=1))

    # if "joints2d" in output:
    #     joints2d = output["joints2d"]
    #     pose = visualize_joints_2d_cv2(
    #         inpimage, joints2d.cpu().detach().numpy()[0]
    #     )

    # if left: 
    #     pose = cv2.flip(inpimage, 1)
    #     cv2.imshow(f"Hand #{hand_idx} Pose", pose)
    
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
    
    return buf

if __name__ == "__main__":
    ray.init()
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    global frame_w, frame_h, w, h

    # init frames
    frames, det_frames, dets_arr, mesh_frames = [], [], [], []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint",
        default="release_models/obman/checkpoint.pth.tar",
        required=True
    )
    parser.add_argument("--video_path", help="Path to video", required=True)
    parser.add_argument('--checksession', dest='checksession',
                      help='Checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='Checkepoch to load network',
                      default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='Checkpoint to load network',
                      default=90193, type=int, required=True)             
    args = parser.parse_args()
    argutils.print_args(args)

    # Init CV2 Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, 'output/')
    
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    iternum = 1
    while os.path.exists(output_directory + str(iternum) + '.mp4'):
        iternum+=1
    # writer = cv2.VideoWriter(output_directory + str(iternum) + '.mp4', fourcc, 30, (w, h))

    # Load model options
    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)
    
    # Load faces of hand
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    # Initialize stream from video + get frames
    cap = cv2.VideoCapture(args.video_path)

    if cap is None:
        raise RuntimeError("OpenCV could not read video")

    print(" ------------------- Reading Video ------------------- \n")
    ret, frame = cap.read()
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framenum = 1
    while True:
        if framenum == len_frames:
            break
        frames.append(frame)
        ret, frame = cap.read()
        framenum += 1

    print(" ------------------- Load Detection Model Weights ------------------- \n")
    model_id = get_state_dict(args.checksession, args.checkepoch, args.checkpoint)

    print(" ------------------- Start Ray DetNets ------------------- \n")
    DetNets = [detection_init(model_id) for i in range(gpus)]
    
    print(" ------------------- Start Detection ------------------- \n")

    results= ray.get([DetNets[i%gpus].forward.remote(frames[i]) for i in range(len(frames))])
    mhands = 0
    for result in results:
        hand_dets, det_frame  = detection(result)
        if hand_dets is not None:
            det_frames.append(det_frame)
            dets_arr.append(hand_dets)
            mhands = max(mhands, len(hand_dets))
        else:
            dets_arr.append(0)
            det_frames.append(0)

    print(" ------------------- End Detection ------------------- \n")
    for actor in DetNets:
        ray.kill(actor)

    figs = [plt.figure(figsize=(4, 4)) for i in range(mhands)]
    import math
    w = frame_w + (int(math.floor(mhands/2)*math.floor(frame_w/2)) if mhands%2==0 else int(math.floor(mhands/2+1)*math.floor(frame_w/2)))
    h = frame_h

    print(" ------------------- Load 3D Mesh Model Weights ------------------- \n")
    weights = modelio.load_state_dict(args.resume)
    weights_id = ray.put(weights)

    print(" ------------------- Start Ray HandNets ------------------- \n")
    HandNets = [reload_ray_model(args.resume, opts, weights_id, mhands) for i in range(mhands)]

    print(" ------------------- Start Mesh Reconstruction ------------------- \n")
    for i in range(len(frames)):
        hand_dets = dets_arr[i]
        frame = frames[i]
        det_frame = det_frames[i]
        if type(hand_dets) is int: 
            mesh_frames.append(createframe(mode=0, frame=frame))
            continue
        hand_dets = [(hand_idx + 1, hand_dets[i, :]) for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))) ]
        hands = [(hand_idx, crop(frame, det, 1.2), det[-1]) for hand_idx, det in hand_dets]
        # [
        #     cv2.imshow(f"Hand #{hand_idx}", frame)
        #     for hand_idx, frame, side in hands
        # ]
        hands = [(hand_idx, cv2.resize(preprocess_frame(frame), (256, 256)), not bool(side)) for hand_idx, frame, side in hands]
        hands_input = [(hand_idx, prepare_input(frame, flip_left_right=not side,), side) for hand_idx, frame, side in hands]


        samples = [
            (forward_pass_3d(hand, left=side), hand_idx, side)
            for hand_idx, hand, side in hands_input
        ]

        meta = [i[1:3] for i in samples]
        results= ray.get([HandNets[i%mhands].forward.remote(samples[i][0], no_loss=True) for i in range(len(samples))])
        meshes = [ cv2.resize(plot(hands[i], results[i][1], figs[i]), (int(frame_h/2), int(frame_h/2)-20)) for i in range(len(results))]
        
        mesh_frames.append(createframe(meshes=meshes, mode=1 if len(meshes)%2==0 else 2, frame=det_frame, meta=meta))

for frame in mesh_frames:
    cv2.imshow("final frame", frame)
    cv2.waitKey(1)