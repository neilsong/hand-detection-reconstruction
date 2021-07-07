from mano_train.demo.preprocess import prepare_input, preprocess_frame
from mano_train.visualize import displaymano
from handobjectdatasets.viz2d import visualize_joints_2d_cv2
import numpy as np
from copy import deepcopy
from PIL import Image
import pickle
from mano_train.demo.attention import AttentionHook
from mano_train.netscripts.reload import reload_model
import os, cv2
from matplotlib import pyplot as plt
import torch
from handobjectdatasets.queries import TransQueries, BaseQueries

def forward_pass_3d(model, input_image, device, pred_obj=True, left=True):
    sample = {}
    sample[TransQueries.images] = input_image.to(device)
    sample[BaseQueries.sides] = ["left" if left else "right"]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float().to(device)
    sample["root"] = "wrist"
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones(
            (1, 600, 3)
        ).float().to(device)
    # print(sample)
    _, results, _ = model.forward(sample, no_loss=True)

    return results

def worker(input):
    fig = plt.figure(figsize=(4, 4))
    queue, gpu, number, resume = input

    # Worker startup declation
    print(f"Worker #{number} initializing...")


    # Load model options
    checkpoint = os.path.dirname(resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)
    
    # Load faces of hand
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    # Reload model from checkpoint
    model = reload_model(resume, opts)

    # Send model to worker's GPU
    device = torch.device("cuda:" + str(gpu))
    model.to(device)

    # Add attention map
    attention_hand = AttentionHook(model.module.base_net)
    if hasattr(model.module, "atlas_base_net"):
        attention_atlas = AttentionHook(model.module.atlas_base_net)
        has_atlas_encoder = True
    else:
        has_atlas_encoder = False

    # Finish Worker Init
    print(f"Worker #{number} initialized")

    while True:
        fig.clf()
        hand_idx, frame, left = queue.get()

        # (Atlas) Attention
        blend_img_hand = attention_hand.blend_map(preprocess_frame(frame))
        if has_atlas_encoder:
            blend_img_atlas = attention_atlas.blend_map(frame)
            cv2.imshow("Hand #{hand_idx} Atlas Attention", blend_img_atlas)
        cv2.imshow(f"Hand #{hand_idx} Attention", blend_img_hand)


        img = Image.fromarray(frame.copy())

        hand_crop = cv2.resize(np.array(img), (256, 256))
        hand_image = prepare_input(
            hand_crop, flip_left_right=not left
        )

        output = forward_pass_3d(model, hand_image, device, left=left)

        # Pose Estimation (L-only)
        if left:
            inpimage = deepcopy(hand_crop)
        else:
            inpimage = deepcopy(np.flip(hand_crop, axis=1))

        if "joints2d" in output:
            joints2d = output["joints2d"]
            inpimage = visualize_joints_2d_cv2(
                inpimage, joints2d.cpu().detach().numpy()[0]
            )

        if left: cv2.imshow(f"Hand #{hand_idx} Pose", cv2.flip(inpimage, 1))
        
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

        cv2.waitKey(1)