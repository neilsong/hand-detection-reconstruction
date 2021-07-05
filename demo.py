import argparse
import os
import pickle

import cv2
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from PIL import Image

from handobjectdatasets.queries import TransQueries, BaseQueries
from handobjectdatasets.viz2d import visualize_joints_2d_cv2

from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.attention import AttentionHook
from mano_train.demo.preprocess import prepare_input, preprocess_frame

from detection.detection import detection_init, detection


def forward_pass_3d(model, input_image, pred_obj=True, left=True):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = ["left" if left else "right"]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample["root"] = "wrist"
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones(
            (1, 600, 3)
        ).float()
    _, results, _ = model.forward(sample, no_loss=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint",
        default="release_models/obman/checkpoint.pth.tar",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hand_side", default="left")
    parser.add_argument("--video_path", help="Path to video")
    parser.add_argument(
        "--no_beta", action="store_true", help="Force shape to average"
    )
    parser.add_argument(
        "--left", action="store_true", help="Force shape to average"
    )
    parser.add_argument(
        "--right", action="store_true", help="Force shape to average"
    )
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=90193, type=int, required=True)
    args = parser.parse_args()
    argutils.print_args(args)

    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)

    # Initialize network
    model = reload_model(args.resume, opts, no_beta=args.no_beta)
    fasterRCNN = detection_init(args.checksession, args.checkepoch, args.checkpoint)

    model.eval()

    # Initialize stream from camera
    if args.video_path is None:
        # Read from webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)

    if cap is None:
        raise RuntimeError("OpenCV could not use webcam")

    print("Please use {} hand !".format(args.hand_side))

    # load faces of hand
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    # Add attention map
    attention_hand = AttentionHook(model.module.base_net)
    if hasattr(model.module, "atlas_base_net"):
        attention_atlas = AttentionHook(model.module.atlas_base_net)
        has_atlas_encoder = True
    else:
        has_atlas_encoder = False

    fig = plt.figure(figsize=(4, 4))
    while True:
        fig.clf()
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("OpenCV could not load frame")
        hand_dets = detection(frame, fasterRCNN)
        print(hand_dets)
        frame = preprocess_frame(frame)
        blend_img_hand = attention_hand.blend_map(frame)
        if has_atlas_encoder:
            blend_img_atlas = attention_atlas.blend_map(frame)
            cv2.imshow("attention atlas", blend_img_atlas)
        img = Image.fromarray(frame.copy())
        hand_crop = cv2.resize(np.array(img), (256, 256))
        hand_image = prepare_input(
            hand_crop, flip_left_right=args.right
        )
        output = forward_pass_3d(model, hand_image, left=args.left)
        if args.left:
            inpimage = deepcopy(hand_crop)
        else:
            inpimage = deepcopy(np.flip(hand_crop, axis=1))
        # noflip_inpimage = cv2.flip(noflip_inpimage, 0)
        # noflip_inpimage = cv2.rotate(noflip_inpimage, cv2.ROTATE_180)
        # noflip_inpimage = cv2.flip(noflip_inpimage, 1)
        if "joints2d" in output:
            joints2d = output["joints2d"]
            inpimage = visualize_joints_2d_cv2(
                inpimage, joints2d.cpu().detach().numpy()[0]
            )

        cv2.imshow("attention hand", blend_img_hand)

        verts = output["verts"].cpu().detach().numpy()[0]
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        displaymano.add_mesh(ax, verts, faces, flip_x=args.left)
        if "objpoints3d" in output:
            objverts = output["objpoints3d"].cpu().detach().numpy()[0]
            displaymano.add_mesh(
                ax, objverts, output["objfaces"], flip_x=args.left, c="r"
            )
        
        if args.left: cv2.imshow("pose", cv2.flip(inpimage, 1))
        
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        # Captured right hand of user is seen as right (mirror effect)
        
        # cv2.imshow("pose estimation", cv2.flip(frame, 1))
        cv2.imshow("mesh", buf)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
