import cv2

def keypointdetection(method, img):
    if method == "SIFT":
        return getSIFT(img)
    elif method == "SURF":
        return getSURF(img)
    else:
        return getSIFT(img)

def getSIFT(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des

def getSURF(img):
    surf = cv2.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    return kp, des