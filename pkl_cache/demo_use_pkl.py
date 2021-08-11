import argparse, sys, os
from PIL import Image, ImageDraw, ImageFont
from numpy.lib import histogram_bin_edges

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from matplotlib import pyplot as plt
from handobjectdatasets.queries import BaseQueries, TransQueries

import cv2

from mano_train.exputils import argutils


from detection.detection import detection_init, detection, get_state_dict
from multiprocessing import Process
from utils.crop import crop
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
from utils.pyrender_util import render_mesh


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

def gentext(meta):
    hand_idx, side = meta
    
    font = ImageFont.truetype('lib/model/utils/times_b.ttf', size=12)
    text = f"{'Left' if side else 'Right'} #{hand_idx}"
    
    curr = deepcopy(white_text)
    draw = ImageDraw.Draw(curr)
    w1, h1 = draw.textsize(text, font=font)
    draw.text(((int(frame_h/2)-w1)/2,(20-h1)/2), text, fill="black", font=font)
    curr = np.array(curr)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2RGBA)

    return curr


def createframe(mode=0,frame=[], meshes=[], meta=[]):
    if mode==0:
        return cv2.hconcat([frame, white0])
    elif mode==1:
        vert_meshes = [cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)]
        for i in range(0,len(meshes),2):
            top = meshes[i]
            bot = meshes[i+1]
            vert_meshes.append(cv2.vconcat([ white1[int(meta[i][1] == True)][meta[i][0]-1], top, bot, white1[int(meta[i+1][1]==True)][meta[i+1][0]-1]]))
        if len(meshes) < mhands: vert_meshes.append(rest[int(len(meshes)/2)-1])
        return cv2.hconcat(vert_meshes)
    elif mode==2:
        vert_meshes = [cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)]
        last = len(meshes) - 1
        for i in range(0,last,2):
            top = meshes[i]
            bot = meshes[i+1]
            vert_meshes.append(cv2.vconcat([ white1[int(meta[i][1]==True)][meta[i][0]-1], top, bot, white1[int(meta[i+1][1]==True)][meta[i+1][0]-1]]))
        vert_meshes.append(cv2.vconcat([white1[int(meta[last][1]==True)][meta[last][0]-1], meshes[last], odd_white]))
        if len(meshes) < mhands: vert_meshes.append(rest[int((len(meshes)+1)/2) - 1])
        return cv2.hconcat(vert_meshes)
# @ray.remote
def pyren_plot(i, hand, verts, fig):
    im = render_mesh( hand, verts, faces)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    return cv2.resize(im, (int(frame_h/2), int(frame_h/2)-20))

    # Implement Display Mesh

if __name__ == "__main__":
    ray.init()
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    global frame_w, frame_h, w, h, white0, white1, rest, white_text, odd_white, mhands, display_mesh
    white1 = [[],[]]
    rest = []


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
    parser.add_argument('--display_mesh', action='store_true')              
    args = parser.parse_args()
    argutils.print_args(args)

    display_mesh = args.display_mesh

    # Init CV2 Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, 'output/')
    
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    iternum = 1
    while os.path.exists(output_directory + str(iternum) + '.mp4'):
        iternum+=1

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

    print(" ------------------- Loading Cached Detections & Meshes ------------------- \n")
    results_pklfile = open('pkl_cache/results_pkl', 'rb')

    hands_pklfile = open('pkl_cache/hands_pkl', 'rb')

    det_frames_pklfile = open('pkl_cache/det_frames_pkl', 'rb')

    results_pkl = pickle.load(results_pklfile)
    hands_pkl = pickle.load(hands_pklfile)
    det_frames = pickle.load(det_frames_pklfile)

    print(" ------------------- Reading Video ------------------- \n")
    ret, frame = cap.read()
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framenum = 1
    while True:
        if framenum == len_frames:
            break
        # frame = cv2.resize(frame, (640, 480))
        frames.append(frame)
        ret, frame = cap.read()
        framenum += 1
    
    cap.release()

    mhands = 0
    for hands in hands_pkl:
        if type(hands) is not int:
            mhands = max(mhands, len(hands))

    figs = [plt.figure(figsize=(20, 20)) for i in range(mhands)]
    import math
    w = frame_w + (int(math.floor(mhands/2)*math.floor(frame_h/2)) if mhands%2==0 else int(math.floor(mhands/2+1)*math.floor(frame_h/2)))
    h = frame_h

    white0 = np.zeros([h, w-frame_w,3],dtype=np.uint8)
    white0.fill(255)
    white_text = np.zeros([20, int(frame_h/2),3],dtype=np.uint8)
    white_text.fill(255)
    white_text = white_text[:,:,::-1]
    white_text = Image.fromarray(white_text).convert("RGB")
    for i in range(1, mhands+1):
        white1[0].append(gentext((i, False)))
        white1[1].append(gentext((i, True)))

        if i%2==0:
            rest.append( cv2.cvtColor( white0[:,0: (w-frame_w - int(frame_h/2) * int(i/2))], cv2.COLOR_RGB2RGBA))

    odd_white =  np.zeros([int(frame_h/2), int(frame_h/2),3], dtype=np.uint8)
    odd_white.fill(255)
    odd_white = cv2.cvtColor(odd_white, cv2.COLOR_RGB2RGBA)

    writer = cv2.VideoWriter(output_directory + str(iternum) + '.mp4', fourcc, 20, (w, h))

    for i in range(len(frames)):
        hands = hands_pkl[i]
        frame = frames[i]
        if type(det_frames[i]) is int: 
            frame = createframe(frame=frame, mode=0)
            cv2.imwrite(f"/home/cgalab/handobj/hand-detection-reconstruction/pkl_cache/final/{i}_frame.jpg", frame)
            print(i)
            continue
        hands_input = [(hand_idx, prepare_input(frame, flip_left_right=not side,), side) for hand_idx, frame, side in hands]
        samples = [
            (forward_pass_3d(hand, left=side), hand_idx, side)
            for hand_idx, hand, side in hands_input
        ]

        meta = [i[1:3] for i in samples]
        results = results_pkl[i]
        meshes = [
            pyren_plot(f"pkl_cache/store/{i}_{j}.jpg", hands[j], results[j][1]["verts"].cpu().detach().numpy()[0], figs[j]) for j in range(len(results))
        ]

        frame = (meshes, 1 if len(meshes)%2==0 else 2, det_frames[i], meta)

        if type(frame) is not tuple:
            frame = createframe(frame=frame, mode=0)
        else:
            frame = createframe(meshes=frame[0], mode=frame[1], frame=frame[2], meta=frame[3])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        cv2.imwrite(f"/home/cgalab/handobj/hand-detection-reconstruction/pkl_cache/final/{i}_frame.jpg", frame)
        print(i)
        
        del frame
        del meshes
    
    writer.release()
    cv2.destroyAllWindows()
