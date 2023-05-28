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
            print("Can't receive frame. Exiting ...")
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


def limit_gpu_memory_usage():
    # Limit GPU memory usage.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.list_physical_devices('GPU')

def load_image(image_path):
    image = tf.io.read_file(image_path) # Byte encoded image.
    img = tf.image.decode_jpeg(image, channels=3)
    return img

def load_into_tensorflow_dataset():
    # Load images into tensorflow dataset.
    images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)  # Shuffle false to keep images in order.
    # Apply the load_image function to every image in the dataset.
    images = images.map(load_image)
    image_generator = images.batch(4).as_numpy_iterator()  # Batch in groups of 4 images (Grouping images together).
    plot_images = image_generator.next()  # Get the next batch of images.

    # Visualize the images.
    fig, axes = plt.subplots(1, 4, figsize=(10, 10))
    for i, ax in enumerate(axes):
        ax.imshow(plot_images[i])  # Plot each image in the batch

    plt.show()


def main():
    # If you want to collect images, uncomment the line below.
    #  collect_images()
    # Limit GPU memory usage.
    limit_gpu_memory_usage()
    # Load images into tensorflow dataset and visualize them.
    load_into_tensorflow_dataset()



if __name__ == '__main__':
        # app.run(host='0.0.0.0', port=5000, debug=True);
    main()

