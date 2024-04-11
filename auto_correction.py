import cv2
from cv2 import aruco
import numpy as np

# Your camera calibration parameters
camera_matrix = np.array([[966.80083369, 0.0, 649.99730882],
                          [0.0, 971.15680184, 362.74810822],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([[0.12043679, -0.12656466, -0.00104852, 0.00258223, -0.67734902]])

# Standard position and orientation of the marker
standard_marker_position = None
standard_marker_rotation = None

# Initialize ArUCo parameters and detector
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
detector_params = aruco.DetectorParameters()
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary, parameters=detector_params)

    if marker_ids is not None:
        # Draw detected markers
        aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        # Estimate pose of the markers
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners, 0.05, camera_matrix, dist_coeffs)

        if standard_marker_position is None:
            # Set the standard position and orientation if not already set
            standard_marker_position = tvecs
            standard_marker_rotation = rvecs

        else:
            # Calculate translation and rotation between current and standard position and orientation
            translation = tvecs - standard_marker_position
            rotation_vector_diff = rvecs - standard_marker_rotation
            rotation_matrix_diff, _ = cv2.Rodrigues(rotation_vector_diff)

            # Apply inverse transformation to the current frame
            inverse_translation = -translation.squeeze()
            inverse_rotation_matrix = np.transpose(rotation_matrix_diff)
            inverse_transform = np.hstack((inverse_rotation_matrix, inverse_translation.reshape(3, 1)))
            transformed_frame = cv2.warpAffine(frame, inverse_transform, (frame.shape[1], frame.shape[0]))

            # Display transformed frame
            cv2.imshow("Transformed Image", transformed_frame)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()