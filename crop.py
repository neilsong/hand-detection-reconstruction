#import cv2 as cv

def crop(orig_img, hand_dets, scale):
    SMALLEST_SIZE = 256
    
    xmin = hand_dets[0]
    ymin = hand_dets[1]
    xmax = hand_dets[2]
    ymax = hand_dets[3]
    print(xmin+" "+xmax)
    print(ymin+" "+ymax)
    width = abs(xmin-xmax)
    height = abs(ymin-ymax)
    side = max(width,height)*scale
    side = max(side, SMALLEST_SIZE)

    xcenter = int((xmax-xmin+1)/2)
    ycenter = int((ymax-ymin+1)/2)
    
    left_edge = max(0, xcenter - int(side/2))
    right_edge = min(len(orig_img), xcenter + int((side+1)/2))
    upper_edge = max(0, ycenter - int(side/2))
    bottom_edge = min(len(orig_img[0]), ycenter + int((side+1)/2))

    img = orig_img[left_edge:right_edge,upper_edge:bottom_edge].copy()
    print(left_edge+" "+right_edge)
    print(upper_edge+" "+bottom_edge)
    return cv.resize(img,(SMALLEST_SIZE,SMALLEST_SIZE),interpolation = cv.INTER_AREA)

xmin = 220
xmax = 230
ymin = 220
ymax = 230
scale = 2.0

SMALLEST_SIZE = 256
width = abs(xmin-xmax)
height = abs(ymin-ymax)
side = max(width,height)*scale
side = max(side, SMALLEST_SIZE)

xcenter = int((xmax+xmin+1)/2)
ycenter = int((ymax+ymin+1)/2)

left_edge = max(0, xcenter - int(side/2))
right_edge = min(500, xcenter + int((side+1)/2))
upper_edge = max(0, ycenter - int(side/2))
bottom_edge = min(500, ycenter + int((side+1)/2))

print(str(left_edge)+" "+str(right_edge))
print(str(upper_edge)+" "+str(bottom_edge))