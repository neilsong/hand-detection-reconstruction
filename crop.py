import cv2 as cv
def crop(orig_img, xmin, ymin, xmax, ymax, scale):
    final SMALLEST_SIZE = 256
    
    width = abs(xmin-xmax)
    height = abs(ymin-ymax)
    side = max(width,height)*scale
    side = max(side, SMALLEST_SIZE)

    xcenter = (xmax-xmin+1)/2
    ycenter = (ymax-ymin+1)/2

    left_edge = xcenter - side/2
    right_edge = xcenter + (side+1)/2
    upper_edge = ycenter - side/2
    bottom_edge = ycenter + (side+1)/2

    img = orig_img[left_edge:right_edge,upper_edge:bottom_edge].copy()

    return cv.resize(img,(SMALLEST_SIZE,SMALLEST_SIZE),interpolation = cv2.INTER_AREA)

orig = cv.imread()