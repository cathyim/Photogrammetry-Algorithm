'''
Adapted from
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
'''
import cv2
import numpy as np
from PIL import Image
from imageio.core import asarray
from matplotlib import pyplot as plt


def keypoints(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    imagematch(kp1, kp2, des1, des2)

def imagematch(img1, img2, kp1, kp2, des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    goodmatches = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            goodmatches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    epilines(img1, img2, pts1, pts2)

def drawlines(img1,img2,lines,pts1,pts2):
    print(img1.shape)
    r1,c1,g1 = img1.shape

    img1new = img1[:, :, [2, 1, 0]]
    img1 = Image.fromarray(img1new)

    img2new = img2[:, :, [2, 1, 0]]
    img2 = Image.fromarray(img2new)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c1, -(r[2]+r[0]*c1)/r[1] ])
        img1bs = asarray(img1)
        img2bs = asarray(img2)

        img1 = cv2.line(img1bs, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1bs,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2bs,tuple(pt2),5,color,-1)
    return img1,img2

def epilines(img1, img2, pts1, pts2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


def main():
    img1 = cv2.imread(filename1) # replace filename with path of first photo
    img2 = cv2.imread(filename2) # replace filename with path of second photo
    keypoints(img1, img2)

main()
