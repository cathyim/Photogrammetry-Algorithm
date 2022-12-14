import cv2

from cameracalibration import load

def undistort_img(img):
  K = load("K")
  dist = load("dist")
  h,w = img.shape[:2]

  new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
  img_undistorted = cv2.undistort(img, K, dist, None, new_camera_matrix)
  return img_undistorted