'''
Gained reference from https://github.com/dionysus/2D_3D/blob/main/match.py
'''
import cv2
import glob
import numpy as np

from imagedistorion import undistort_img
from cameracalibration import load
from keypoints import keypointdetection
from imagematching import getMatches
from triangulation import projected_point, normalization
from output import save


def principalcoordinates(focal, camera, angle):
    c1 = camera[0] + (-focal * np.sin(angle))
    c2 = camera[1]
    c3 = camera[2] + (-focal * np.cos(angle))
    point = np.array([c1, c2, c3])
    return point


def pointloc(principal, plane, imagepoint, mmp, angle):
    xpix = imagepoint[0] - plane[0]
    ypix = imagepoint[1] - plane[1]
    ymm = ypix * mmp
    xmm = xpix * mmp
    image_angle = angle + np.radians(90)
    c1 = principal[0] + (xmm * np.sin(image_angle))
    c2 = ymm
    c3 = principal[2] + (xmm * np.cos(image_angle))
    point = np.array([c1, c2, c3])
    return point


def convert(img1pts, img2pts, img1index, img2index, img1image):
  pairpts = []
  pairrgb = []
  nummatches = img1pts.shape[0]
  K = load("K")
  focal = K[0][0]
  img1p = np.array([int(K[0][2]//1), int(K[1][2]//1)])
  img2p = np.array([int(K[0][2]//1), int(K[1][2]//1)])
  cameramatrix = load("cameraproperties")

  for i in range(nummatches):
    img1pt = img1pts[i]
    w, h = img1image.shape[1], img1image.shape[0]

    # camera properties
    sensordim = 9, 6
    focal_mm = focal * sensordim[1] / w
    mmp = focal_mm/focal

    img1_distance, img1_location = get_img_dist_loc(cameramatrix, img1index, focal_mm, img1p, img1pts, i, mmp)
    img2_distance, img2_location = get_img_dist_loc(cameramatrix, img2index, focal_mm, img2p, img2pts, i, mmp)

    # calculate projected points
    projected = projected_point(img1_distance, img1_location, img2_distance, img2_location)
    pairpts.append(projected)

    # retrieve image rgb values
    rgb = img1image[int(img1pt[1])][int(img1pt[0])]
    pairrgb.append(rgb)

  return pairpts, pairrgb

def get_img_dist_loc(camera_matrix, img_index, focal_mm, img_p, img_pts, i, mmp):
  img_camera_coord = np.array([
    camera_matrix[img_index][0],
    camera_matrix[img_index][1],
    camera_matrix[img_index][2]])

  img_angle = np.radians(camera_matrix[img_index][3])
  img_principal = principalcoordinates(focal_mm, img_camera_coord, img_angle)
  img_location = pointloc(img_principal, img_p, img_pts[i], mmp, img_angle)
  distance = img_location - img_camera_coord
  value = normalization(distance)
  img_distance = distance / value

  return img_distance, img_location

def twoimages(img1, img2, img1index, img2index):
  img1new = undistort_img(img1)
  img1Gnew = cv2.cvtColor(img1new, cv2.COLOR_BGR2GRAY)
  kp1, des1 = keypointdetection("SIFT", img1Gnew)

  img2new = undistort_img(img2)
  img2Gnew = cv2.cvtColor(img2new, cv2.COLOR_BGR2GRAY)
  kp2, des2 = keypointdetection("SIFT", img2Gnew)


  allmatches, goodmatches, matchesmask = getMatches(des1, des2, False)

  img1pts = np.float32([ kp1[m.queryIdx].pt for m in goodmatches ])
  img2pts = np.float32([ kp2[m.trainIdx].pt for m in goodmatches ])

  pairpts, pairrgb = convert(img1pts, img2pts, img1index, img2index, img1new)

  return pairpts, pairrgb

def makemodel(folderpath, photos):
  images = sorted(glob.glob(folderpath + "/*.JPG"))

  iters = len(images) if photos else len(images) - 1
  cloud_pts = []
  cloud_rgb = []

  for i in range(iters):


    img1 = cv2.imread(images[i], 1)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    if i < len(images) - 1:
      img2_index = i+1
    elif photos:
      img2_index = 0
    else:
      break # error here, shouldn't reach
    img2 = cv2.imread(images[img2_index], 1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


    pair_pts, pair_rgb = twoimages(img1, img2, i, img2_index)

    cloud_pts.extend(pair_pts)
    cloud_rgb.extend(pair_rgb)

  cloud_pts = np.array(cloud_pts)
  cloud_rgb = np.array(cloud_rgb)
  save(cloud_pts, cloud_rgb)


def BF(des1, des2):
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    return all_matches

def FLANN(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    all_matches = flann.knnMatch(des1,des2,k=2)
    return all_matches



def getMatches(des1, des2, method, list=True):
    if method == "BF":
        matchpoints = BF(des1, des2)
    elif method == "FLANN":
        matchpoints = FLANN(des1, des2)
    else:
        matchpoints = BF(des1, des2)

    good = []
    goodflat = []
    matches_mask = [[0, 0] for i in range(len(matchpoints))]

    for i, (m, n) in enumerate(matchpoints):
        if m.distance < 0.75 * n.distance:
            good.append([m])
            goodflat.append(m)
            matches_mask[i] = [1, 0]

    if list:
        return matchpoints, good, matches_mask
    else:
        return matchpoints, goodflat, matches_mask

