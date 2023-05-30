import cv2 as cv
import os
import time
import uuid


def collect_images():
    IMAGES_DIR = os.path.join('data', 'images')
    duration = 100

    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    start_time = time.time()
    end_time = start_time + duration

    img_count = 0
    while time.time() < end_time and img_count < 500:
        print('Collecting image {}'.format(img_count))
        return_value, image = camera.read()

        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img_name = os.path.join(IMAGES_DIR, '{}.jpg'.format(uuid.uuid1()))
        cv.imwrite(img_name, image)
        time.sleep(0.1)

        img_count += 1

    camera.release()
    cv.destroyAllWindows()


collect_images()
