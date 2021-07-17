import argparse
from PIL import Image, ImageDraw, ImageFont

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
        if last: vert_meshes.append(cv2.vconcat([top, odd_white]))
        if len(meshes) < mhands: vert_meshes.append(rest[int((len(meshes)+1)/2) - 1])
        return cv2.hconcat(vert_meshes)

@ray.remote
def plot(hand, verts, fig):
    fig.clf()

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
    
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    displaymano.add_mesh(ax, verts, faces, flip_x=left)

    if display_mesh:
        fig1 = plt.figure(figsize=(9, 9))
        ax1 = fig1.add_subplot(1, 1, 1, projection="3d")
        displaymano.add_mesh(ax1, verts, faces, flip_x=left)
        plt.axis('off')
        fig1.canvas.draw()
        w1, h1 = fig1.canvas.get_width_height()
        buf1 = np.fromstring(fig1.canvas.tostring_argb(), dtype=np.uint8)
        buf1.shape = (w1, h1, 4)

        current_directory = os.getcwd()
        output_directory = os.path.join(current_directory, 'output_im/')
        
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        iternum = 1
        while os.path.exists(output_directory + "im" + str(iternum) + '.png'):
            iternum+=1
        
        cv2.imwrite(output_directory + "im" + str(iternum) + '.png', buf1)
        plt.axis('on')

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    
    return cv2.resize(buf, (int(frame_h/2), int(frame_h/2)-20))

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

    print(" ------------------- Reading Video ------------------- \n")
    ret, frame = cap.read()
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = 480
    frame_w = 640
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framenum = 1
    while True:
        if framenum == len_frames:
            break
        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)
        ret, frame = cap.read()
        framenum += 1
    
    cap.release()

    print(" ------------------- Load Detection Model Weights ------------------- \n")
    model_id = get_state_dict(args.checksession, args.checkepoch, args.checkpoint)

    print(" ------------------- Start Ray DetNets ------------------- \n")
    DetNets = [detection_init(model_id) for i in range(gpus)]
    
    print(" ------------------- Start Detection ------------------- \n")

    results= np.copy(ray.get([DetNets[i%gpus].forward.remote(frames[i]) for i in range(len(frames))]))
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

    odd_white =  np.zeros([int(frame_h/2)+20, int(frame_h/2),3], dtype=np.uint8)
    odd_white.fill(255)
    odd_white = cv2.cvtColor(odd_white, cv2.COLOR_RGB2RGBA)

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
        start = time.time()
        results= np.copy(ray.get([HandNets[i%mhands].forward.remote(samples[i][0], no_loss=True) for i in range(len(samples))]))
        mesh_end = time.time()
        meshes = np.copy(ray.get([plot.remote(hands[i], results[i][1]["verts"].cpu().detach().numpy()[0], figs[i]) for i in range(len(results))]))
        frame_end = time.time()
        mesh_frames.append((meshes, 1 if len(meshes)%2==0 else 2, det_frame, meta))
        
        print(f"\n\nFrame #{i+1} complete\nMesh Time: {(mesh_end - start)}\nPlot Time: {(frame_end - mesh_end)}")

    writer = cv2.VideoWriter(output_directory + str(iternum) + '.mp4', fourcc, 20, (w, h))

    for frame in mesh_frames:
        if type(frame) is not tuple:
            frame = createframe(frame=frame, mode=0)
        else:
            frame = createframe(meshes=frame[0], mode=frame[1], frame=frame[2], meta=frame[3])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        writer.write(frame)
    
    writer.release()
    cv2.destroyAllWindows()
