import math
import cv2
import argparse
import numpy as np
import random
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def show_pic(imge):
    cv2.imshow('pic', imge)
    cv2.waitKey()


def sp_noise(image, prob):

    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
    return result

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--source', type=str, default='mr_bean.PNG')

    args = my_parser.parse_args()

    img = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)

    noisy_img = sp_noise(img, 0.05)

    low_noise_img = cv2.medianBlur(noisy_img, 3)

    # low_noise_img = cv2.flip(low_noise_img, 1)

    show_pic(low_noise_img)

    # ----------------------------------------------------------
    detector = MTCNN()
    faces = detector.detect_faces(cv2.cvtColor(low_noise_img, cv2.COLOR_GRAY2RGB))

    # plot faces
    ax = plt.gca()
    for face in faces:

        x, y, width, height = face['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)

        for key, value in face['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)

    plt.show()

    mr_bin = faces[0]
    l_eye = mr_bin["keypoints"]["left_eye"]
    r_eye = mr_bin["keypoints"]["right_eye"]

    l_x, l_y = l_eye
    r_x, r_y = r_eye
    slope = math.atan((l_y - r_y)/(l_x - r_x))

    print(slope)
    rotated_image = rotate_image(low_noise_img, (slope*180)/math.pi)

    cv2.imwrite('mr_bean_rotated.jpg', rotated_image)
    cv2.imshow('out', rotated_image)
    cv2.waitKey()

