import cv2
from cv2 import aruco
import numpy as np
cap=cv2.VideoCapture(0)
markerIds = []
markerIds = np.array(markerIds)
markerCorners = []
rejectedCandidates = []
def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			# show the output image
	return image


detectorParams =aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
det=aruco.ArucoDetector(dictionary, detectorParams)
while True:
    _,frame=cap.read()
    corners, ids, rejected=det.detectMarkers(frame)
    aruco.drawDetectedMarkers(frame, corners, ids)
    #detected_markers = aruco_display(corners, ids, rejected, frame)
    cv2.imshow("Image", frame)
    if cv2.waitKey(2) & 0xff==27:
        break
        
cap.release()
cv2.destroyAllWindows()