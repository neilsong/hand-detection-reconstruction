import cv2 as cv

def crop(orig_img, hand_dets, scale):
    SMALLEST_SIZE = 256
    
    #assign the edges of the bounding box based on hand_dets
    xmin = hand_dets[0]
    ymin = hand_dets[1]
    xmax = hand_dets[2]
    ymax = hand_dets[3]

    #define the side length based on width and height
    width = abs(xmin-xmax)
    height = abs(ymin-ymax)
    side = max(width,height)*scale
    side = max(side, SMALLEST_SIZE)

    #define the center of the bounding box
    xcenter = (xmax+xmin)/2
    ycenter = (ymax+ymin)/2
    
    #define the edges of the cropped image
    left_edge = max(0, int(xcenter - side/2))
    right_edge = min(len(orig_img)-1, int(xcenter + side/2 + 1))
    upper_edge = max(0, int(ycenter - side/2))
    bottom_edge = min(len(orig_img[0])-1, int(ycenter + side/2 + 1))

    img = orig_img[left_edge:right_edge,upper_edge:bottom_edge].copy()

    return cv.resize(img,(SMALLEST_SIZE,SMALLEST_SIZE),interpolation = cv.INTER_AREA)
