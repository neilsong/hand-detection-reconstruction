import pickle
import pyrender
import trimesh
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_scene(vert, faces, scale, pose=None):

    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                        ambient_light=np.array([1.0, 1.0, 1.0]))

    camera = pyrender.PerspectiveCamera(yfov=1.0471975511965976, znear= 0.05, zfar= 2325.572423894192)

    mesh = trimesh.Trimesh(vertices=vert, faces=faces)
    mesh1 = pyrender.Mesh.from_trimesh(mesh)
    mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
    mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
    mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
    node1 = scene.add(mesh1)
    node2 = scene.add(mesh2)

    # scale = (1.0/scale)

    # z_axis = pose[:3,2].flatten()
    # eye = pose[:3,3].flatten()
    # radius = np.linalg.norm(eye - scene.centroid)
    # translation = (scale * radius - radius) * z_axis
    # t_tf = np.eye(4)
    # t_tf[:3,3] = translation
    # pose = t_tf.dot(pose)

    # scene.add(camera, pose=pose)

    return scene

def render_mesh(hand, vert, faces, scale, trans):

    pose = np.array(
        [
            [-8.09554183e-01,  5.51611133e-01, -2.00866080e-01, -6.11017235e+01],
            [ 5.85274996e-01,  7.84947535e-01, -2.03249961e-01, -6.30069441e+01],
            [ 4.55543931e-02, -2.82103750e-01, -9.58301765e-01, -2.11703737e+02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ]
    )
    scene_r = create_scene(vert, faces, scale, pose)

    height, width, channels = hand[1].shape

    r = pyrender.OffscreenRenderer(viewport_width=width,
                                    viewport_height=height)

    im_render, _ = r.render(scene_r)

    im_real = hand[1]
    im_real = im_real[:, :, ::-1]

    im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
    im = im.astype(np.uint8)

    del r
    del scene_r
    return im

    # ax.imshow(im)
    # ax.set_axis_off()
    # return ax

if __name__ == '__main__':
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    results_pklfile = open('/home/cgalab/handobj/hand-detection-reconstruction/pkl_cache/results_pkl', 'rb')

    hands_pklfile = open('/home/cgalab/handobj/hand-detection-reconstruction/pkl_cache/hands_pkl', 'rb')

    results_pkl = pickle.load(results_pklfile)
    hands_pkl = pickle.load(hands_pklfile)

    

    pose = np.array(
        [
            [-5.15717424e-01,  8.29574890e-01, -2.14105212e-01, -6.20680875e+01],
            [ 8.56633925e-01,  5.03547132e-01, -1.12332554e-01, -3.16658734e+01],
            [ 1.46237992e-02, -2.41341644e-01, -9.70330025e-01, -2.15828996e+02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ]
    )

    def save_cam(Viewer):
        global pose
        i = list(Viewer.scene.camera_nodes)[0]
        pose = Viewer.scene.get_pose(i)
        print(pose)

    for i in range(40,len( results_pkl)):
        print(i)
        if type(results_pkl[i]) is int:
            continue

        results = results_pkl[i][0]
        hands = hands_pkl[i][0]

       
        # print('Close the window to continue.')

        # cv2.imwrite("Hand.jpg", hands[1])
        scene = create_scene( results[1]["verts"].cpu().detach().numpy()[0],faces, results[3], pose)

        pyrender.Viewer(scene,
            registered_keys= {
                'y': (save_cam)
            }
        )

        # scene_r = create_scene(results[1]["verts"].cpu().detach().numpy()[0],faces, results[3],pose)

        height, width, channels = hands[1].shape

        r = pyrender.OffscreenRenderer(viewport_width=width,
                                        viewport_height=height)

        # im_render, _ = r.render(scene_r)

        im_real = hands[1]
        # im_real = im_real[:, :, ::-1]
        cv2.imshow(f"pkl_cache/store/{i}", im_real)
        cv2.waitKey(0)

        # im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
        # im = im.astype(np.uint8)
        
        # cv2.imwrite(f"pkl_cache/store/{i}.jpg", im)