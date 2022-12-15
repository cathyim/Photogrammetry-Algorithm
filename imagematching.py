'''
Gained reference from https://github.com/dionysus/2D_3D/blob/main/match.py
Focal calculation methods are taken directly from https://github.com/dionysus/2D_3D/blob/main/match.py
'''
import cv2
import glob
import numpy as np

from imagedistorion import undistort_img
from cameracalibration import load
from keypoints import keypointdetection
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


def get_focal_mm(focal, sensor_width, image_width):
    return focal*sensor_width/image_width


def mm_pixel(focal_mm, focal_pixel):
    return focal_mm/focal_pixel

def convert(img1_pts, img2_pts, img1_index, img2_index, img1_image):
    pair_pts = []
    pair_rgb = []
    num_matches = img1_pts.shape[0]
    K = load("K")
    focal = K[0][0]
    img1_p = np.array([int(K[0][2]//1), int(K[1][2]//1)])
    img2_p = np.array([int(K[0][2]//1), int(K[1][2]//1)])
    camera_matrix = load("cameraproperties")
    for i in range(num_matches):
        img1_pt = img1_pts[i]
        img2_pt = img2_pts[i]
        w, h = img1_image.shape[1], img1_image.shape[0]
        sensor_dim = 9, 6
        sensor_width = sensor_dim[1]
        focal_mm = get_focal_mm(focal, sensor_width, w)
        mmp = mm_pixel(focal_mm, focal)
        img1_distance, img1_location = get_img_dist_loc(camera_matrix, img1_index, focal_mm, img1_p, img1_pts, i, mmp)
        img2_distance, img2_location = get_img_dist_loc(camera_matrix, img2_index, focal_mm, img2_p, img2_pts, i, mmp)

        # calculate projected points
        projected = projected_point(img1_distance, img1_location, img2_distance, img2_location)
        pair_pts.append(projected)

        # retrieve image rgb values
        rgb = img1_image[int(img1_pt[1])][int(img1_pt[0])]
        pair_rgb.append(rgb)
    return pair_pts, pair_rgb

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

def processphoto(img):
    img_undistorted = undistort_img(img)
    img_undistorted_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
    kp, des = keypointdetection("SIFT", img_undistorted_gray)
    return kp, des, img_undistorted

def twoimages(img1, img2, img1index, img2index):
    kp1, des1, img1_undistorted = processphoto(img1)
    kp2, des2, img2_undistorted = processphoto(img2)

    # get matches
    all_matches, good_matches, matches_mask = getmatches(kp1, des1, kp2, des2, "BRUTE", False)

    print(good_matches)
    img1pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    img2pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    pairpts, pairrgb = convert(img1pts, img2pts, img1index, img2index, img1_undistorted)
    return pairpts, pairrgb

def makemodel(folderpath, photos):
    images = sorted(glob.glob(folderpath + "/*.JPG"))
    iters = len(images) if photos else len(images) - 1
    cloudpts = []
    cloudrgb = []
    for i in range(iters):
        print(i)
        img1 = cv2.imread(images[i], 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        if i < len(images) - 1:
            img2_index = i+1
        elif photos:
            img2_index = 0
        else:
            break

        img2 = cv2.imread(images[img2_index], 1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


        pairpts, pairrgb = twoimages(img1, img2, i, img2_index)

        cloudpts.extend(pairpts)
        cloudrgb.extend(pairrgb)
    print(cloudpts)
    cloudpts = np.array(cloudpts)
    cloudrgb = np.array(cloudrgb)
    save(cloudpts, cloudrgb)


def Brute(des1, des2):
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


def getmatches(kp1, des1, kp2, des2, method, list=True):
    if method == "BRUTE":
        all_matches = Brute(des1, des2)
    elif method == "FLANN":
        all_matches = FLANN(des1, des2)
    else:
        all_matches = Brute(des1, des2)

    # Apply ratio test
    good_matches = []
    good_matches_flat = []
    matches_mask = [[0, 0] for i in range(len(all_matches))]

    for i, (m, n) in enumerate(all_matches):
        if m.distance < 0.75 * n.distance:  # only accept matchs that are considerably better than the 2nd best match
            good_matches.append([m])
            good_matches_flat.append(m)  # this is to simplify finding a homography later
            matches_mask[i] = [1, 0]

    if list:
        return all_matches, good_matches, matches_mask
    else:
        return all_matches, good_matches_flat, matches_mask
