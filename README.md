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
 - [Project repository](https://github.com/ddshan/hand_object_detector.git)

# Get the code

`git clone https://github.com/neilsong/hand-detection-reconstruction && cd hand-detection-reconstruction`

# Download and prepare models
## Download model files

- Download model files from [here](http://www.di.ens.fr/willow/research/obman/release_models.zip) `wget http://www.di.ens.fr/willow/research/obman/release_models.zip` (Note: directly downloading the files is probably faster)
- Unzip `unzip release_models.zip`

## Install python dependencies

- Create conda environment with dependencies: `conda env create -f environment.yml`
- Activate environment: `conda activate hand-det-recon`
- Install detection dependencies: `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
- Build detection library: `cd lib && python setup.py build develop`

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

# Demo

You can test it on a recorded video or live using a webcam by launching :

`python webcam_demo.py --resume release_models/obman/checkpoint.pth.tar  --hands 2 --checksession 1 --checkepoch 10 --checkpoint 90193`

Concurrency with 3D mesh rendering is achieved by assigning a single to model to a single hand, so the max number of hands must be specified to achieve 1 to 1 concurrency. Models will be distributed as evenly as possible across all available GPUs.

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
Thanks [Thibault](https://github.com/ThibaultGROUEIX/) for the provided code!

## Hand evaluation code

Code for computing hand evaluation metrics was reused from [hand3d](https://github.com/lmb-freiburg/hand3d), courtesy of [Christian Zimmermann](https://lmb.informatik.uni-freiburg.de/people/zimmermc/) with an easy-to-use interface!


## Laplacian regularization loss

[Code](https://github.com/akanazawa/cmr) for the laplacian regularization and precious advice was provided by [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)!


## First Hand Action Benchmark dataset

Helpful advice to work with the dataset was provided by [Guillermo Garcia-Hernando](https://guiggh.github.io/)!