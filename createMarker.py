import cv2
from cv2 import aruco
import numpy as np

markerImage=np.zeros((600, 600), dtype=np.uint8)
dictionary =aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco.generateImageMarker(dictionary, 23, 600, markerImage, 1)
cv2.imshow("marker23", markerImage)
cv2.waitKey(0)