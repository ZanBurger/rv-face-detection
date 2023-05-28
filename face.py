from flask import Flask, jsonify
import os  # File management
import time  # Time management
import uuid  # Unique file names
import cv2 as cv  # OpenCV
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

# # Flask server uses port 5000. To run on website use localhost:5000/api/data
# app = Flask(__name__)
#
#
# @app.route('/api/data', methods=['GET'])
# def get_data():
#     data = {"message": "Hello from Python backend!"}
#     return jsonify(data)


def collect_images():
    IMAGES_DIR = os.path.join('data', 'images')
    number_of_images = 30

    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    for img in range(number_of_images):
        print('Collecting image {}'.format(img))
        return_value, image = camera.read()

        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv.imshow('Image Capture', image)

        img_name = os.path.join(IMAGES_DIR, '{}.jpg'.format(uuid.uuid1()))
        cv.imwrite(img_name, image)
        time.sleep(0.5)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and destroy the window
    camera.release()
    cv.destroyAllWindows()



def main():
    collect_images()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
        # app.run(host='0.0.0.0', port=5000, debug=True);
    main()

