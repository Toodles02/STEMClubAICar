import cv2 as cv
import numpy as np
import imutils


color = (255,255,255)
# I have defined lower and upper boundaries for each color
# Customize for camera used
colors = {'blue': [np.array([95, 255, 85]), np.array([120, 255, 255])],
          'red': [np.array([161, 165, 127]), np.array([178, 255, 255])],
          'yellow': [np.array([16, 0, 99]), np.array([39, 255, 255])],
          'green': [np.array([33, 19, 105]), np.array([77, 255, 255])]}


def find_color(frame, points):
    mask = cv.inRange(frame, points[0], points[1])#create mask with boundaries 
    cnts = cv.findContours(mask, cv.RETR_TREE, 
                           cv.CHAIN_APPROX_SIMPLE) # find contours from mask
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        area = cv.contourArea(c) # find how big countour is
        if area > 5000:       #if countour is big enough, then
            M = cv.moments(c)
            cx = int(M['m10'] / M['m00']) # calculate X position
            cy = int(M['m01'] / M['m00']) # calculate Y position
            return c, cx, cy
