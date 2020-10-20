import os

import cv2
import numpy as np

"""
All images in the Color Checker dataset are linear images in the RAW format of the acquisition device, each with a
Macbeth ColorChecker (MCC) chart, which provides an estimation of illuminant colors. To prevent the CNN from detecting
and utilizing MCCs as a visual cue, all images are masked with provided locations of MCC during training and testing
"""


def main():
    if not os.path.exists('./data/ndata/'):
        os.mkdir('./data/ndata')

    if not os.path.exists('./data/nlabel'):
        os.mkdir('./data/nlabel')

    # Generate numpy data
    for l in open('./data/color_checker_data_meta.txt', 'r').readlines():
        file_name = l.strip().split(' ')[1]
        print(file_name)

        illuminants = [float(l.strip().split(' ')[2]), float(l.strip().split(' ')[3]), float(l.strip().split(' ')[4])]
        np.vstack(illuminants)
        np.save('./data/nlabel/' + file_name + '.npy', illuminants)

        # BGR image
        img_without_mcc = load_image_without_mcc(file_name, get_mcc_coord(file_name))
        np.save('./data/ndata/' + file_name + '.npy', img_without_mcc)


def load_image_without_mcc(file_name: str, mcc_coord: np.array) -> np.array:
    raw = load_image(file_name)

    # Clip the values between 0 and 1
    img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.float32)

    # Get the vertices of polygon the
    polygon = mcc_coord * np.array([img.shape[1], img.shape[0]])
    polygon = polygon.astype(np.int32)

    # Fill the polygon to img
    cv2.fillPoly(img, [polygon], (1e-5,) * 3)

    return img


def load_image(file_name: str) -> np.array:
    raw = np.array(cv2.imread('./data/images/' + file_name, -1), dtype='float32')

    # Handle pictures taken with Canon 5d Mark III
    black_point = 129 if file_name.startswith('IMG') else 1

    # Keep only the pixels such that raw - black_point > 0
    return np.maximum(raw - black_point, [0, 0, 0])


def get_mcc_coord(file_name: str) -> np.array:
    """ Computes the relative MCC coordinates for the given image """

    lines = open("./data/coordinates/" + file_name.split('.')[0] + "_macbeth.txt", 'r').readlines()
    width, height = map(float, lines[0].split())
    scale_x, scale_y = 1 / width, 1 / height

    polygon = []
    for line in [lines[1], lines[2], lines[4], lines[3]]:
        line = line.strip().split()
        x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
        polygon.append((x, y))

    return np.array(polygon, dtype='float32')


if __name__ == '__main__':
    main()
