import numpy as np
from vispy import app, scene
from vispy.gloo.util import _screenshot

canvas = scene.SceneCanvas(keys='interactive')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

xx, yy = np.arange(-1,1,.02),np.arange(-1,1,.02)
surface = scene.visuals.SurfacePlot(x= xx-0.1, y=yy+0.2, z=0,shading='smooth', color=(0.5, 0.5, 1, 1))
view.add(surface)
canvas.show