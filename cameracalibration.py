'''
Adapted from https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
and https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
'''

import numpy as np
import cv2
import glob


def calibratecamera(foldername):
    board = (9,6)
    width = 6
    height = 9

    # termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1, 2)

    objectpoints = []
    imagepoints = []
    # edit file path to folder holding calibration images
    calibration_images = sorted(glob.glob(foldername+"/*"+ ".JPG"))


    for image in calibration_images:
        img = cv2.imread(image, 0)

        # find corners
        ret, corners = cv2.findChessboardCorners(img, board, None)

        # error message
        if not ret:
            print("checkerboard not found")
        else:
            sub_corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            imagepoints.append(sub_corners)
            objectpoints.append(objp)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, img.shape[::-1], None, None)

    # save
    np.save(".ret"+'.npy', ret)
    #print(ret)
    np.save(".K"+'.npy', K)
    #print(K)
    np.save(".dist"+'.npy', dist)
    #print(dist)


def saveprops(camera, degree, distancex, distancey):
    camera_props = []

    curr_coord = [0, distancey, distancex]
    curr_angle = 0

    curr_camera_prop = curr_coord + [curr_angle]
    camera_props.append(curr_camera_prop)

    for i in range(1, camera):
        curr_angle += degree
        curr_coord = update_degrees(curr_coord, degree)
        curr_camera_prop = curr_coord + [curr_angle]
        camera_props.append(curr_camera_prop)
    print("camera_props")
    print(camera_props)
    camera_props = np.array(camera_props)

    np.save(".camera_props"+'.npy', camera_props)


def update_degrees(coord, degree):
    x, y, z = coord[0], coord[1], coord[2]
    rot = np.radians(degree)

    x_new = x * np.cos(rot) + z * np.sin(rot)
    z_new = -x * np.sin(rot) + z * np.cos(rot)

    return [x_new, y, z_new]

def load(prop):
    if prop == "cameraproperties":
        camera_props = np.load(".camera_props.npy")
        return camera_props
    elif prop == "K":
        K = np.load(".K.npy")
        return K
    elif prop == "ret":
        ret = np.load(".ret.npy")
        return ret
    elif prop == "dist":
        dist = np.load(".dist.npy")
        return dist


def main():
    foldername = "/path/to/folder" # replace with path of folder containing calibration photos
    focal_length = 3.3  # edit focal length
    cameras = 18 # change to number of photos
    distancex = 340 # change to distance from each other along the floor
    distancey = 178 # change to height of camera
    degree = 360/cameras
    calibratecamera(foldername)
    saveprops(cameras,degree,distancex,distancey)
    cameraproperties = load("cameraproperties")
    np.save(".focal" + '.npy', focal_length)
main()