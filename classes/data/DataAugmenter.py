from __future__ import print_function

import math
import random

import cv2
import numpy as np

from auxiliary.config import *


class DataAugmenter:

    def __init__(self):
        # Rotation angle
        self.__angle = 60

        # Patch scale
        self.__scale = [0.1, 1.0]

        # Color rescaling
        self.__color = 0.8

    @staticmethod
    def __rotate_image(image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle (in degrees).
        The returned image will be large enough to hold the entire new image, with a black background
        """

        # Get the image size (no that's not an error - NumPy stores image matrices backwards)
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        image_w2, image_h2 = image_size[0] * 0.5, image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos, x_neg = [x for x in x_coords if x > 0], [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos, y_neg = [y for y in y_coords if y > 0], [y for y in y_coords if y < 0]

        right_bound, left_bound, top_bound, bot_bound = max(x_pos), min(x_neg), max(y_pos), min(y_neg)
        new_w, new_h = int(abs(right_bound - left_bound)), int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)], [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

        # Compute the transform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        return cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def __largest_rotated_rect(w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in radians), computes the width and height of
        the largest possible axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow. Converted to Python by Aaron Snoswell
        """
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return bb_w - 2 * x, bb_h - 2 * y

    @staticmethod
    def __crop_around_center(image, width, height):
        """ Given a NumPy / OpenCV 2 image, crops it to the given width and height, around it's centre point """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if width > image_size[0]:
            width = image_size[0]

        if height > image_size[1]:
            height = image_size[1]

        x1, x2 = int(image_center[0] - width * 0.5), int(image_center[0] + width * 0.5)
        y1, y2 = int(image_center[1] - height * 0.5), int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def __rotate_and_crop(self, image, angle: float):
        image_width, image_height = image.shape[:2]
        image_rotated = self.__rotate_image(image, angle)
        target_width, target_height = self.__largest_rotated_rect(image_width, image_height, math.radians(angle))
        image_rotated_cropped = self.__crop_around_center(image_rotated, target_width, target_height)
        return image_rotated_cropped

    def augment(self, img, illumination):

        if img is None:
            return None, None

        scale = math.exp(random.random() * math.log(self.__scale[1] / self.__scale[0])) * self.__scale[0]
        s = min(max(int(round(min(img.shape[:2]) * scale)), 10), min(img.shape[:2]))

        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * self.__color - 0.5 * self.__color

        start_x = random.randrange(0, img.shape[0] - s + 1)
        start_y = random.randrange(0, img.shape[1] - s + 1)

        img = img[start_x:start_x + s, start_y:start_y + s]
        img = self.__rotate_and_crop(img, angle=(random.random() - 0.5) * self.__angle)
        img = cv2.resize(img, (FCN_INPUT_SIZE, FCN_INPUT_SIZE))

        # Perform random left/right flip with probability 0.5
        if random.randint(0, 1):
            img = img[:, ::-1]
        img = img.astype(np.float32)
        img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]], dtype=np.float32)
        new_image = np.clip(img, 0, 65535)

        # TODO: RGB to BGR method
        # RGB to BGR
        new_illum = np.zeros_like(illumination)
        illumination = illumination[::-1]
        for i in range(3):
            for j in range(3):
                new_illum[i] += illumination[j] * color_aug[i, j]
        new_illum = np.clip(new_illum, 0.01, 100)[::-1]

        return new_image, new_illum

    @staticmethod
    def crop(img, scale: float = 0.5):
        return cv2.resize(img, (0, 0), fx=scale, fy=scale)
