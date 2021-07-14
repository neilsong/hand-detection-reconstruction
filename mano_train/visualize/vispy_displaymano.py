from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from handobjectdatasets.queries import TransQueries, BaseQueries
from handobjectdatasets.viz3d import visualize_joints_3d
from handobjectdatasets.viz2d import visualize_joints_2d

from vispy import plot as vp
from vispy.geometry import polygon
from vispy import scene
from vispy.scene.visuals import Mesh
from vispy.visuals.mesh import MeshVisual

def add_mesh(view, verts, faces, flip_x=False, c="b", alpha=0.1):
    # view.view_init(elev=90, azim=-90) #add this?
    # mesh = polygon.PolygonData(verts[faces], alpha=alpha)
    # mesh = Poly3DCollection(verts[faces], alpha=alpha)
    
    mesh = Mesh(vertices=verts[faces], faces=faces,)
    view.add(mesh)
    view.camera = scene.TurntableCamera()

    # cam_equal_aspect_3d(view, verts, flip_x=flip_x) #update this function
    # plt.tight_layout() #convert

#not used
def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
        ax.set_ylim(centers[1] + r, centers[1] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
        ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] + r, centers[2] - r)