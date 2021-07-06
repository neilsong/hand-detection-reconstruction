import argparse

import cv2

from mano_train.exputils import argutils


from detection.detection import detection_init, detection
from multiprocessing import Process
from crop import crop
from mano_train.demo.preprocess import preprocess_frame
import numpy as np
import multiprocessing
from workers import worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint",
        default="release_models/obman/checkpoint.pth.tar",
        required=True
    )
    parser.add_argument("--video_path", help="Path to video")
    parser.add_argument('--checksession', dest='checksession',
                      help='Checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='Checkepoch to load network',
                      default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='Checkpoint to load network',
                      default=90193, type=int, required=True)
    parser.add_argument('--workers', dest='workers',
                      help='Number of workers to initialize',
                      default=multiprocessing.cpu_count()-1 or 1, type=int,)                  
    args = parser.parse_args()
    argutils.print_args(args)

    # Initialize network
    fasterRCNN = detection_init(args.checksession, args.checkepoch, args.checkpoint)

    # Initialize stream from camera
    if args.video_path is None:
        # Read from webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)

    if cap is None:
        raise RuntimeError("OpenCV could not use webcam")

    print(" ------------------- Start Multiprocessing Workers ------------------- \n")
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool()
    queue = multiprocessing.Manager().Queue()
    pool.map(worker, [(queue, i%3, i+1, args.resume) for i in range(args.workers)] )

    while True:
        ret, frame = cap.read()
        cv2.imshow("orig", frame)
        if not ret:
            raise RuntimeError("OpenCV could not load frame")
        hand_dets = detection(frame, fasterRCNN)
        if hand_dets is not None:
            hand_dets = [(hand_idx + 1, hand_dets[i, :]) for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))) ]
            hands = [(hand_idx, crop(frame, det, 1.2), det[-1]) for hand_idx, det in hand_dets]
            # cv2.imshow("crop", hands[0])
            hands = [(hand_idx, preprocess_frame(frame), not side) for hand_idx, frame, side in hands]
            for hand in hands: queue.put(hand)
        cv2.waitKey(1)


    cap.release()
    cv2.destroyAllWindows()
