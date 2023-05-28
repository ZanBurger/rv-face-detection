import cv2
from flask import Flask, jsonify
import os  # File management
import time  # Time management
import uuid  # Unique file names
import cv2 as cv  # OpenCV
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16




import albumentations as alb


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


def augment_image(image, bboxes, class_labels):
    # Set up transformation pipeline
    augmentor = alb.Compose([
        alb.RandomCrop(width=450, height=450),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
        alb.RandomGamma(p=0.2),
        alb.RGBShift(p=0.2),
        alb.VerticalFlip(p=0.5)
    ], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

    # Apply transformations
    transformed = augmentor(image=image, bboxes=bboxes, class_labels=class_labels)

    return transformed


def run_augmentation_pipeline():
    for partition in ['train', 'test', 'val']:  # Loop through each folder.
        for image in os.listdir(os.path.join('data', partition, 'images')):  # Loop through each image in the folder.
            img = cv2.imread(os.path.join('data', partition, 'images', image))

            coords = [0, 0, 0.00001, 0.00001]
            label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):  # If the label exists, get the coordinates.
                with open(label_path, 'r') as f:
                    label = json.load(f)

                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                coords = list(np.divide(coords, [640, 480, 640, 480]))

            try:
                for x in range(60):  # Augment each image 60 times.
                    augmented = augment_image(image=img, bboxes=[coords], class_labels=['face'])
                    cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'),
                                augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0

                    with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'),
                              'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)


def load_image(image_path):
    image = tf.io.read_file(image_path)  # Byte encoded image.
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


def move_to_labelled():
    for folder in ['train', 'test', 'val']:
        for file in os.listdir(os.path.join('data', folder, 'images')):

            filename = file.split('.')[0] + '.json'
            existing_filepath = os.path.join('data', 'labels', filename)
            if os.path.exists(existing_filepath):
                new_filepath = os.path.join('data', folder, 'labels', filename)
                os.replace(existing_filepath, new_filepath)


def load_augmented_into_tensorflow_dataset():
    train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))  # More efficient to resize here.
    train_images = train_images.map(
        lambda x: x / 255)  # Rather than between 0 and 255 its between 0 and 1. (we can apply sigmoid)

    test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
    test_images = test_images.map(lambda x: x / 255)

    val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
    val_images = val_images.map(load_image)
    val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
    val_images = val_images.map(lambda x: x / 255)

    return train_images, test_images, val_images


def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)

    return [label['class']], [label['bbox']]

def load_labels_into_tensorflow_dataset():
    train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    return train_labels, test_labels, val_labels

def load_into_tensorflow_dataset():
    train_labels, test_labels, val_labels = load_labels_into_tensorflow_dataset()
    train_images, test_images, val_images = load_augmented_into_tensorflow_dataset()

    return train_images, train_labels, test_images, test_labels, val_images, val_labels

def combine_images_labels(train_images, train_labels, test_images, test_labels, val_images, val_labels):
    train = tf.data.Dataset.zip((train_images, train_labels))
    train = train.shuffle(4000)
    train = train.batch(8)
    train = train.prefetch(4)  # Eliminates bottleneck of loading images.

    test = tf.data.Dataset.zip((test_images, test_labels))
    test = test.shuffle(1300)
    test = test.batch(8)
    test = test.prefetch(4)

    val = tf.data.Dataset.zip((val_images, val_labels))
    val = val.shuffle(1000)
    val = val.batch(8)
    val = val.prefetch(4)

    return train, test, val

def view_images_labels(train, test, val):
    data_samples = train.as_numpy_iterator()
    res = data_samples.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx in range(4):
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx].squeeze()

        sample_image = sample_image / np.max(sample_image)

        if sample_coords.ndim != 1 or sample_coords.shape[0] < 4:
            print("Unexpected shape for sample_coords:", sample_coords.shape)
            continue

        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

        ax[idx].imshow(sample_image)
        plt.show()


def main():
    # If you want to collect images, uncomment the line below.
    #  collect_images()
    # Limit GPU memory usage.
    limit_gpu_memory_usage()
    # Load images into tensorflow dataset and visualize them.
    # load_into_tensorflow_dataset()
    # Move the labels to the correct folder.

    # run_augmentation_pipeline()

    train_images, train_labels, test_images, test_labels, val_images, val_labels = load_into_tensorflow_dataset()
    # print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))
    train, test, val = combine_images_labels(train_images, train_labels, test_images, test_labels, val_images, val_labels)
    # print(train.as_numpy_iterator().next()[1])
    view_images_labels(train, test, val)



if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True);
    main()
