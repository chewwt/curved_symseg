import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Dict
import yaml


IM_NAME = '374067'  # in data folder

CONFIG_PATH = './config/symseg_params.yaml'


def main(config_path, im_name):
    # load config
    conf = get_config(config_path)

    # load data
    img = get_img(im_name)
    gt = get_gt(im_name)
    # debug_plot(gt)

    # sort gt by length
    gt_list = sort_gt(gt)

    if len(gt_list) == 0:
        raise ValueError('GT empty')

    # detect edges
    im_edges = detect_edges(img)
    debug_plot(im_edges)

    # take longest curved symmetry line (for example)
    sym_axis = gt_list[0]
    im_sym_axis = np.zeros(gt.shape).astype(np.uint8)
    im_sym_axis[sym_axis[:, 0], sym_axis[:, 1]] = 255
    debug_plot(im_sym_axis)

    # get a contour based representation of the axis
    im_sym_axis = imfill(im_sym_axis)
    im_sym_axis = cv.ximgproc.thinning(im_sym_axis)
    debug_plot(im_sym_axis)

    # get ordered edges
    # edge_curve

# load

def get_config(path: Path) -> Dict:
    with path.open('r') as f:
        conf = yaml.safe_load(f)

    return conf


def get_img(im_name: str) -> np.ndarray:
    path = Path(f'./data/{im_name}.jpg')
    img = cv.imread(str(path))

    return img


def get_gt(im_name: str) -> np.ndarray:
    path = Path(f'./data/gt_{im_name}.png')
    gt = cv.imread(str(path), 0).astype(np.uint8) * 255

    return gt


# pre-process

def sort_gt(gt: np.ndarray) -> List:
    # get connected components from mask
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(gt, 4, cv.CV_8U)

    gt_list = []

    for i in range(1, num_labels):  # label 0 is background
        # ravel coords
        coords = np.where(labels == i)
        points = np.vstack((coords[0], coords[1])).T
        # inds = np.ravel_multi_index(points, gt.shape)
        # gt_list.append(inds)
        gt_list.append(points)

    gt_list.sort(key=lambda i: -len(i))  # sort in decreasing length

    return gt_list


def detect_edges(img: np.ndarray) -> np.ndarray:
    edges = cv.Canny(img,  150, 200)
    return edges


# segmentation


# matlab equivalents

def imfill(bw):
    im_floodfill = bw.copy()

    # size of mask needs to be 2 pixels more than image for cv.floodFill function
    h, w = bw.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill and inverse to get holes
    # holes are areas not reachable by filling in the background from the edge of the image
    cv.floodFill(im_floodfill, mask, (0, 0), 255)
    holes = cv.bitwise_not(im_floodfill)

    im_filled = cv.bitwise_or(bw, holes)
    return im_filled


# debug

def debug_plot(img):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img)
    plt.show()


# args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, default=CONFIG_PATH)
    parser.add_argument('--im_name', '-i', type=str, default=IM_NAME)

    args1 = parser.parse_args()
    args1.config_path = Path(args1.config_path)

    return args1


if __name__ == '__main__':
    args = get_args()
    main(args.config_path, args.im_name)
