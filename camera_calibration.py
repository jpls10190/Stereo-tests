import cv2
import numpy as np
from tqdm import tqdm

# RESIZE IMAGE 
def resize_img(img, width, height):
    scale_width = width / img.shape[1]
    scale_height = height / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    img = cv2.resize(img, (window_width, window_height))
    return img

# Set the path to the images captured by the left and right cameras
pathL = "./img/stereoL/"
pathR = "./img/stereoR/"
 
# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
 
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
 
img_ptsL = []
img_ptsR = []
obj_pts = []
 
for i in tqdm(range(1,12)):
 imgL = cv2.imread(pathL+"img%d.jpg"%i)
 imgR = cv2.imread(pathR+"img%d.jpg"%i)
 imgL_gray = cv2.imread(pathL+"img%d.jpg"%i,0)
 imgR_gray = cv2.imread(pathR+"img%d.jpg"%i,0)
 
 outputL = imgL.copy()
 outputR = imgR.copy()
 
 retR, cornersR =  cv2.findChessboardCorners(outputR,(9,6),None)
 retL, cornersL = cv2.findChessboardCorners(outputL,(9,6),None)
 
 if retR and retL:
  obj_pts.append(objp)
  cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
  cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
  cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
  cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)
  cv2.imshow('cornersR',resize_img(outputR,900,900))
  cv2.imshow('cornersL',resize_img(outputL,900,900))
  cv2.waitKey(0)
 
  img_ptsL.append(cornersL)
  img_ptsR.append(cornersR)
 
 
# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))
 
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))