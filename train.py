import os
import cv2 as cv
import numpy as np

from skimage import feature
from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from joblib import dump, load

from sklearn.metrics import accuracy_score

# ---- LBP ----
def lbp_algorithm(image_path, num_points, radius):
    # Load the image
    image = cv.imread(image_path)

    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # lbp_image = np.zeros_like(gray)
    # for y in range(0, gray.shape[0]):
    #     for x in range(0, gray.shape[1]):
    #         lbp_value = 0 # LBP vrednost za trenutni piksel
    #         for n in range(num_points): # Število sosedov
    #             # Izračunamo koordinate sosedov
    #             theta = 2. * np.pi * (n / num_points)
    #             x_p = x + radius * np.sin(theta)
    #             y_p = y + radius * np.cos(theta)
    #             # Get pixel value
    #             value = gray[int(y_p), int(x_p)] if 0 <= y_p < gray.shape[0] and 0 <= x_p < gray.shape[1] else 0
    #             # Update LBP value
    #             lbp_value += (value >= gray[y, x]) << n # Če je intenzivnost sosedov večja ali enaka centru piksla, nastavimo na 1 (True), drugače na 0 (False)
    #         # Save LBP value to image
    #         lbp_image[y, x] = lbp_value

    # Compute LBP features
    lbp = feature.local_binary_pattern(gray, num_points, radius, method="uniform")  # FROM LIBRARY

    # Compute histogram of LBP
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist


# ---- HOG ----
def hog_algorithm(image_path, orientations, pixels_per_cell, cells_per_block):
    # Load the image
    image = cv.imread(image_path)

    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Resize the image
    gray = cv.resize(gray, (64, 128))  # Make sure all images are the same size

    # Compute HOG features
    hog_features = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm='L2-Hys', transform_sqrt=True)

    return hog_features


# Extracts lbp and hog from the training data
def extract_lbp_and_hog(x_train):
    # Using LBP
    num_points = 24
    radius = 8

    # print("Extracting LBP features from training data...")
    # x_train_lbp = [lbp_algorithm(img_path, num_points, radius) for img_path in x_train]

    # Using HOG
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # print("Extracting HOG features from training data...")
    # Extract HOG features from the training data
    # x_train_hog = [hog_algorithm(img_path, orientations, pixels_per_cell, cells_per_block) for img_path in x_train]
    print("Extracting LBP and HOG features from training data...")
    x_train_hog_and_lbp = [np.concatenate((lbp_algorithm(img_path, num_points, radius),
                                           hog_algorithm(img_path, orientations, pixels_per_cell, cells_per_block)))
                           for img_path in x_train]

    return x_train_hog_and_lbp


# Training and saving the model based on lbp and hog extracted data
def train_and_save(x_train_lbp_and_hog, y_train):
    # TRAINING USING SVM
    print("Starting to train the SVM model on LBP and HOG features...")

    # Scale the features to zero mean and unit variance
    scaler = StandardScaler().fit(x_train_lbp_and_hog)
    x_train_scaled = scaler.transform(x_train_lbp_and_hog)
    print("Scaling successful...")
    dump(scaler, 'scaler_lbp_hog.joblib')

    # Train an SVM on the LBP and HOG features
    svm = SVC(kernel='linear', probability=True)
    svm.fit(x_train_scaled, y_train)
    print("Training on LBP and HOG features succesful")
    dump(svm, 'svm_lbp_hog.joblib')

    # Train a Decision Tree (DT) on the LBP and HOG features
    print("Starting to train the DT model on LBP and HOG features...")
    dt = DecisionTreeClassifier()
    dt.fit(x_train_scaled, y_train)
    print("Training on LBP and HOG features succesful")
    dump(dt, 'dt_lbp_hog.joblib')


# Loading and testing the model
def load_and_test(test_folder, categories):
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

    # For each image in the test folder
    for filename in os.listdir(test_folder):
        img_path = os.path.join(test_folder, filename)
        # Skip if not an image file
        if not os.path.isfile(img_path) or not any(ext in filename.lower() for ext in ['.jpg', '.jpeg', '.png']):
            continue
        # Extract LBP and HOG features
        features = np.concatenate((lbp_algorithm(img_path, num_points, radius),
                                   hog_algorithm(img_path, orientations, pixels_per_cell, cells_per_block)))
        # Scale the features using the same scaler that was used on the training data
        features_scaled = scaler.transform([features])

        # Make predictions with both models
        svm_pred = svm.predict(features_scaled)
        dt_pred = dt.predict(features_scaled)

        # Print the results
        print(f"Image {filename} classified by SVM as {categories[svm_pred[0]]} and by DT as {categories[dt_pred[0]]}.")


# Splits the data into test data and train data based on test_size parameter
def razdeli_podatke(data, test_size):
    test_size = int(len(data) * test_size)

    np.random.shuffle(data)

    test_data = data[:test_size]
    train_data = data[test_size:]

    return train_data, test_data


# ---- MAIN ----


main_folders = ["C:/Users/Burge.LAPTOP-SAFB8NNK/Desktop/College/FERI/2. Letnik/Projektno Delo/rv-face-detection/data/images/Marko", "C:/Users/Burge.LAPTOP-SAFB8NNK/Desktop/College/FERI/2. Letnik/Projektno Delo/rv-face-detection//data/images/Zan", "C:/Users/Burge.LAPTOP-SAFB8NNK/Desktop/College/FERI/2. Letnik/Projektno Delo/rv-face-detection//data/images/Other"]
categories = ["Marko", "Zan", "Other"]


# Data separation.
data = []

for folder in main_folders:  # loop over the main folders
    category = os.path.basename(folder)  # get the category from the folder name
    if category in categories:
        class_num = categories.index(category)  # 0 for Marko, 1 for Zan
        if os.path.exists(folder):
            for img in os.listdir(folder):
                try:
                    img_path = os.path.join(folder, img)
                    data.append([img_path, class_num])
                except Exception as e:
                    pass
        else:
            print(f"Directory {folder} does not exist.")


train_data, test_data = razdeli_podatke(data, test_size=0.2)

# x are image paths and y are labels
x_train = [item[0] for item in train_data]  # image paths
y_train = [item[1] for item in train_data]  # labels

x_test = [item[0] for item in test_data]  # image_paths
y_test = [item[1] for item in test_data]  # labels

x_train_hog_and_lbp = extract_lbp_and_hog(x_train)

train_and_save(x_train_hog_and_lbp, y_train)

load_and_test("C:/Users/Burge.LAPTOP-SAFB8NNK/Desktop/College/FERI/2. Letnik/Projektno Delo/rv-face-detection/data/images/TestImages", categories)



