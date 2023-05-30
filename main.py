import cv2
from skimage.feature import local_binary_pattern, hog
from joblib import load
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained models
svm = load('svm_lbp_hog.joblib')
dt = load('dt_lbp_hog.joblib')
scaler = load('scaler_lbp_hog.joblib')

# Prepare for feature extraction
num_points = 24
radius = 8
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# To capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # For each detected face
    for (x, y, w, h) in faces:
        for (x, y, w, h) in faces:
            # Draw the rectangle around each face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get the face ROI and resize it to match training data size
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (64, 128))

        # Compute LBP features
        lbp = local_binary_pattern(face_roi, num_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        # Compute HOG features
        hog_features = hog(face_roi, orientations=orientations, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block, block_norm='L2-Hys', transform_sqrt=True)

        # Concatenate LBP and HOG features
        features = np.concatenate((hist, hog_features))

        # Scale the features using the same scaler that was used on the training data
        features_scaled = scaler.transform([features])

        # Make predictions with both models
        dt_pred = dt.predict(features_scaled)

        # Print the results
        print(f"Face classified by DT as {dt_pred[0]}.")

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
