import pickle
import cv2

hands_pklfile = open('pkl_cache/hands_pkl', 'rb')
hands_pkl = pickle.load(hands_pklfile)
for idx, i in enumerate(hands_pkl):
    if type(i) is int: continue
    for idx2, j in enumerate(i):
        stri = str(j[2])
        cv2.imwrite(f'/home/cgalab/handobj/hand-detection-reconstruction/pkl_cache/store/raw/{idx}_{idx2}_{stri}.jpg', j[1])