from distutils.log import WARN
import os
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt



def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def perspective_transform(imagepath):

	# Step 1: EDGE DETECTION
	image = cv2.imread(imagepath)
	orig = image.copy()
	kernel = np.ones((5,5),np.uint8)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= 3)
	
	blur = cv2.GaussianBlur(image, (7, 7), 0)
	edged = cv2.Canny(blur, 50, 100)
	edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

	# Step 2: FINDING CONTOURS
	# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points, then we can assume that we have found our screen
		warped = orig
		if len(approx) == 4:
			screenCnt = approx
			# Step 4: PERSPECTIVE TRANSFORMATION
			warped = four_point_transform(orig, screenCnt.reshape(4, 2))
			break
	return warped

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Path to raw image and saved image.')
	parser.add_argument('--imagepath', help='path to raw image')
	parser.add_argument('--savepath', help='path to saved image')
	args = parser.parse_args()
	imagepath = args.imagepath
	savepath = args.savepath

	warped = perspective_transform(imagepath)
	plt.imsave(savepath, warped)
	print('[INFO] Perspective-transformed image was saved at ' ,os.path.join(os.getcwd(), savepath))
