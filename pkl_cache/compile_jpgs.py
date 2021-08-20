import os
from os import listdir
from os.path import join, basename, split, splitext
import cv2
PATH = "/home/cgalab/handobj/hand-detection-reconstruction/pkl_cache/final"

def path_leaf(path):
    head, tail = split(path)
    return tail or basename(head)
files = []
files += [join(PATH, fname) for fname in listdir(PATH) if join(PATH, fname).endswith('.jpg')]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
current_directory = os.getcwd()
output_directory = os.path.join(current_directory, 'output/')
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
iternum = 1
while os.path.exists(output_directory + str(iternum) + '.mp4'):
    iternum+=1

img = cv2.imread(files[0])
height, width, layers = img.shape

writer = cv2.VideoWriter(output_directory + str(iternum) + '.mp4', fourcc, 20, (width, height))

for file in files:
    img = cv2.imread(file)
    writer.write(img)

writer.release()


