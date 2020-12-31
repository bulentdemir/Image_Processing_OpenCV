# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:53:02 2020

@author: Bulent Demir
"""

import cv2
import numpy as np
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

# reading image which is "rakamlar.jpg"
img_BGR = cv2.imread('rakamlar.jpg', 1)
#cv2.imshow('Color Image', img_BGR)

# Copying of colored image
BGR_copy = img_BGR.copy()

# Converting color of image to grayScale
img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray Image', img_gray)

# Thresholding image by filter of THRESH_BINARY_INV
_, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('Binary Image', img_thresh)

# bware function that is affected from the code part written by "Mahmut Sinecen"
def bware(image, connectivity, min_size):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity)

    sizes = stats[1:, -1]
    nb_components -= 1

    # Result image consists of black pixels
    img2 = np.zeros(output.shape)

    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2


img_bw = bware(img_thresh, 8, 10).astype(np.uint8)
cv2.imshow('Bwared Image', img_bw)


img_contours, _ = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_drawContours = cv2.drawContours(BGR_copy, img_contours, -1, (0, 255, 0), 1)
#cv2.imshow('Draw Countor on Image', img_drawContours)

# Dilation image with array of strel
strel = np.ones([2, 2])
I_dilation = cv2.dilate(img_bw, strel, iterations=1)
#cv2.imshow('Dilationed Image', I_dilation)

# Finding Countors
cnts = cv2.findContours(I_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

orig = img_BGR.copy()


def midpoint(ptA, ptB):
    return [(ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5]


# bounidingBox function that draws the box to image
def boundingBox(orig, box):
    
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 255), 2)


# calculateAxisLengths function returns the axises (minor and major) of the box
def calculateAxisLengths(midpoints):
    axisList = []

    axisList.append(np.sqrt((midpoints[0][0] - midpoints[1][0]) ** 2 + (midpoints[0][1] - midpoints[1][1]) ** 2))
    axisList.append(np.sqrt((midpoints[2][0] - midpoints[3][0]) ** 2 + (midpoints[2][1] - midpoints[3][1]) ** 2))

    return axisList


# minorAxisLength function returns the min value from calculateAxisLengths function
def minorAxisLength(midpoints):
    minAxis = min(calculateAxisLengths(midpoints))
    return minAxis


# majorAxisLength function returns the max value from calculateAxisLengths function
def majorAxisLength(midpoints):
    maxAxis = max(calculateAxisLengths(midpoints))
    return maxAxis


# boxArea function returns the area of the box
def boxArea(midpoints):
    area = minorAxisLength(midpoints) * majorAxisLength(midpoints)
    return area


# eccentricity is calculated from sqrt(1 - (ma / MA)^2)
def eccentricity(midpoints):
    ecc = np.sqrt(1 - (minorAxisLength(midpoints) / majorAxisLength(midpoints)) ** 2)
    return ecc


for c in cnts:
    if cv2.contourArea(c) < 30:
        continue

    # calculating min Area rect by cv2.minAreaRect
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype='int')

    (tl, tr, br, bl) = box

    # Calculate midpoints of the bounding box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    midpoints = [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]

    ## 5 functions that satisfy "regionprops" in matlab
    # Drawing the box to original image
    boundingBox(orig, box)
    # Calculating Minor Axis Length
    minorAxis = minorAxisLength(midpoints)
    # Calculating Major Axis Length
    majorAxis = majorAxisLength(midpoints)
    # Calculating eccentricity
    ecc = eccentricity(midpoints)
    # Calculating area of bounding box
    area = boxArea(midpoints)

    print("Minor Axis Length: " + str(minorAxis))
    print("Major Axis Length: " + str(majorAxis))
    print("Box Area: " + str(area))
    print("Eccentricity: " + str(ecc))
    print("---------------------------")
    ## end of the part

    for (x, y) in box:
        # Draw circle to corners of the bounding box
        cv2.circle(orig, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Draw circle to midpoint of the corners of the bounding box
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # Draw line to axis's of the bounding box
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 1)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 1)

    # For the showing one by one box in the image
    # Because of detecting which minor axis, major axis, box area and eccentricity are owned
    cv2.imshow('Result image', orig)
    cv2.waitKey(0)


if cv2.waitKey(0) & 0xff == 27 | ord('q'):
    cv2.destroyAllWindows()
