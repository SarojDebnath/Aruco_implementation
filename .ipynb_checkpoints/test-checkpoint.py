import cv2
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

def euler_from_quaternion(x, y, z, w):
  
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
      
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
      
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
      
    return roll_x, pitch_y, yaw_z

def angles(marker_ids,tvecs,rvecs):
    for i, marker_id in enumerate(marker_ids):
        transform_translation_x = tvecs[i][0][0]
        transform_translation_y = tvecs[i][0][1]
        transform_translation_z = tvecs[i][0][2]
    
        # Store the rotation information
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        quat = r.as_quat()   
         
        # Quaternion format     
        transform_rotation_x = quat[0] 
        transform_rotation_y = quat[1] 
        transform_rotation_z = quat[2] 
        transform_rotation_w = quat[3] 
         
        # Euler angle format in radians
        roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, transform_rotation_y, transform_rotation_z, transform_rotation_w)
         
        roll_x = math.degrees(roll_x)
        pitch_y = math.degrees(pitch_y)
        yaw_z = math.degrees(yaw_z)
        print("transform_translation_x: {}".format(transform_translation_x))
        print("transform_translation_y: {}".format(transform_translation_y))
        print("transform_translation_z: {}".format(transform_translation_z))
        print("roll_x: {}".format(roll_x))
        print("pitch_y: {}".format(pitch_y))
        print("yaw_z: {}".format(yaw_z))

#X: red, Y: green, Z: blue
cap=cv2.VideoCapture(0)

camera_matrix=np.matrix([[966.80083369,0.0,649.99730882],[0.0,971.15680184,362.74810822],[0.0,0.0,1.0]])
dist_coeffs=np.matrix([[0.12043679,-0.12656466,-0.00104852,0.00258223,-0.67734902]])

markerIds = []
markerIds = np.array(markerIds)
markerCorners = []
rejectedCandidates = []

detectorParams =aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)#DICT_6X6_250
det=aruco.ArucoDetector(dictionary, detectorParams)
while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected)=cv2.aruco.detectMarkers(gray, dictionary, markerCorners, markerIds, detectorParams, rejectedCandidates)
    
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, obj_points = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        #print(rvecs, tvecs)
        for i in range(ids.size):
            frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], length=0.05 )
            #aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
    #calculateAnglesHERE:
        if cv2.waitKey(2) & 0xff==ord('q'):
            angles(ids,tvecs,rvecs)
    cv2.imshow("Image", frame)
    if cv2.waitKey(2) & 0xff==27:
        break
cap.release()
cv2.destroyAllWindows()
