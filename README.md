# Monocular RGB Hand-Object Detection and 3D Hand Reconstruction - Demo

A fully-integrated, real-time, multi-focal hand and object detection and 3D hand reconstruction system. Hand detection and state/side classification are used to address limitations in 3D reconstruction model.
# Papers
## Learning Joint Reconstruction of Hands and Manipulated Objects

Yana Hasson, Gül Varol, Dimitris Tzionas, Igor Kalevatykh, Michael J. Black,  Ivan Laptev, Cordelia Schmid, CVPR 2019

- [Project page](https://hassony2.github.io/obman)
- [Dataset repository](https://github.com/hassony2/obman)
- [Robotic grasps generation using MANO](https://github.com/ikalevatykh/mano_grasp)
- [Dataset generation](https://github.com/hassony2/obman_render)

## Understanding Human Hands in Contact at Internet Scale

Dandan Shan, Jiaqi Geng*, Michelle Shu*, David F. Fouhey, CVPR 2020, Oral

 - [Project and dataset webpage](http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/)

# Get the code

`git clone https://github.com/neilsong/hand-detection-reconstruction && cd hand-detection-reconstruction`

# Download and prepare models
## Download model files

- Download model files from [here](http://www.di.ens.fr/willow/research/obman/release_models.zip) `wget http://www.di.ens.fr/willow/research/obman/release_models.zip` (Note: directly downloading the files is probably faster)
- unzip `unzip release_models.zip`

## Install python dependencies

- create conda environment with dependencies: `conda env create -f environment.yml`
- activate environment: `conda activate hand-det-recon`
- install detection dependencies: `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
- build detection libraries: `cd lib && python setup.py build develop`

## Install the MANO PyTorch layer

- Follow the instructions from [here](https://github.com/hassony2/manopth)

### Download the MANO model files

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the content of the *models* folder into the misc/mano folder

- Your structure should look like this:

```
hand-detection-reconstruction/
  misc/
    mano/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
  release_models/
    fhb/
    obman/
    hands_only/

```

# Install VisPy
`conda create -c conda-forge -n vispy_env python=3.6 vispy pyqt=5* numpy nose pytest`  
`activate vispy_env`

# Demo

You can test it on a recorded video or live using a webcam by launching :

`python webcam_demo.py --resume release_models/obman/checkpoint.pth.tar  --hand_side left`

Hand side detection is not handled in this pipeline, therefore, you should explicitly indicate whether you want to use the right or left hand with `--hand_side`.

Note that the video demo has some lag time, which comes from the visualization bottleneck (matplotlib image rendering is quite slow).

### Limitations

- This demo doesn't operate hand detection, so the model expects a roughly centered hand
- As we are deforming a sphere, the topology of the object is 0, resulting in hand poses curving in on objects (incorrect pose)
- the model is trained only on hands holding objects, and therefore doesn't perform well on hands in the absence of objects for poses that do not resemble common grasp poses.
- the model is trained on grasping hands only, and therefore struggles with hand poses that are associated with object-handling
  - In addition to the models, we also provide a hand-only model trained on various hand datasets, including our ObMan dataset, that captures a wider variety of hand poses
  - to try it, launch `python webcam_demo.py --resume release_models/hands_only/checkpoint.pth.tar`
  - Note that this model also regresses a translation and scale parameter that allows to overlay the predicted 2D joints on the images according to an orthographic projection model

# Citations

If you find this code useful for your research, consider citing:

```
@INPROCEEDINGS{hasson19_obman,
  title     = {Learning joint reconstruction of hands and manipulated objects},
  author    = {Hasson, Yana and Varol, G{\"u}l and Tzionas, Dimitris and Kalevatykh, Igor and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2019}
}
```

```
@INPROCEEDINGS{Shan20, 
    author = {Shan, Dandan and Geng, Jiaqi and Shu, Michelle  and Fouhey, David},
    title = {Understanding Human Hands in Contact at Internet Scale},
    booktitle = CVPR, 
    year = {2020} 
}
```

# Acknowledgements

## AtlasNet code

Code related to [AtlasNet](http://imagine.enpc.fr/~groueixt/atlasnet/) is in large part adapted from the official [AtlasNet repository](https://github.com/ThibaultGROUEIX/AtlasNet).
Thanks [Thibault](https://github.com/ThibaultGROUEIX/) for the provided code !

## Hand evaluation code

Code for computing hand evaluation metrics was reused from [hand3d](https://github.com/lmb-freiburg/hand3d), courtesy of [Christian Zimmermann](https://lmb.informatik.uni-freiburg.de/people/zimmermc/) with an easy-to-use interface!


## Laplacian regularization loss

[Code](https://github.com/akanazawa/cmr) for the laplacian regularization and precious advice was provided by [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) !


## First Hand Action Benchmark dataset

Helpful advice to work with the dataset was provided by [Guillermo Garcia-Hernando](https://guiggh.github.io/) !