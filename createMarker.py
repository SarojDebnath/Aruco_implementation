import cv2
from cv2 import aruco
import numpy as np

markerImage=np.zeros((600, 600), dtype=np.uint8)
dictionary =aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
aruco.generateImageMarker(dictionary, 0, 600, markerImage, 1)
cv2.imshow("marker23", markerImage)
cv2.waitKey(0)
cv2.imwrite('Marker5x5_100.PNG',markerImage)