import cv2 as cv

def crop(orig_img, handbbx, scale, diagnostics = False):
    SMALLEST_SIZE = 256
    
    #assign the edges of the bounding box based on hand_dets
    xmin = handbbx[0]
    ymin = handbbx[1]
    xmax = handbbx[2]
    ymax = handbbx[3]

    #define the side length based on width and height
    width = abs(xmin-xmax)
    height = abs(ymin-ymax)
    side = int(max(width,height)*scale)
    side = max(side, SMALLEST_SIZE)

    #define the center of the bounding box
    xcenter = (xmax+xmin)/2
    ycenter = (ymax+ymin)/2
    
    #define the edges of the cropped image
    left_edge = max(0, int(xcenter - side/2))
    right_edge = min(len(orig_img[0])-1, int(xcenter + side/2 + 1))
    upper_edge = max(0, int(ycenter - side/2))
    bottom_edge = min(len(orig_img)-1, int(ycenter + side/2 + 1))

    img = orig_img[upper_edge:bottom_edge,left_edge:right_edge].copy()
    img = cv.resize(img,(SMALLEST_SIZE,SMALLEST_SIZE),interpolation = cv.INTER_AREA)
    
    if diagnostics:
        print("original image size :" + str(len(orig_img))+"x"+str(len(orig_img[0])))
        print("final image size :" + str(len(img))+"x"+str(len(img[0])))
        print("xcenter:"+str(xcenter)+" ycenter:"+str(ycenter)+" side:"+str(side))
        print("img["+str(upper_edge)+":"+str(bottom_edge)+", "+str(left_edge)+":"+str(right_edge)+"]")    

    return img

def crop_all(img, hand_dets,scale,diagnostics=False): #takes hand_dets array and returns an array of cropped images
    images = []
    for hand_index in range(0,len(hand_dets)):
        if diagnostics:
            print("handbbx #"+str(hand_index))
        images.append(crop(img,hand_dets[hand_index],scale,diagnostics))
    return images