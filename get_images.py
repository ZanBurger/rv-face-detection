import cv2 as cv
import os
import time
import uuid

def collect_images():
    IMAGES_DIR = os.path.join('data', 'M')
    duration = 60  # Duration in seconds

    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    start_time = time.time()
    end_time = start_time + duration

    img_count = 0
    while time.time() < end_time:
        print('Collecting image {}'.format(img_count))
        return_value, image = camera.read()

        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break


        img_name = os.path.join(IMAGES_DIR, '{}.jpg'.format(uuid.uuid1()))
        cv.imwrite(img_name, image)
        time.sleep(0.5)

        img_count += 1

    # When everything done, release the capture and destroy the window
    camera.release()
    cv.destroyAllWindows()

collect_images()