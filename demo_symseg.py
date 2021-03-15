import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
import yaml


IMG_NAME = '374067'  # in data folder

CONFIG_PATH = './config/symseg_params.yaml'


def main(config_path, img_name):
    # load config
    conf = get_config(config_path)

    # load data
    img = get_img(img_name)
    gt = get_gt(img_name)

    # sort gt by length
    gt_ind_list = sort_gt(gt)

    # detect edges
    img_edges = detect_edges(img)


# load

def get_config(path: Path):
    with path.open('r') as f:
        conf = yaml.safe_load(f)

    return conf


def get_img(img_name: str):
    path = Path(f'./data/{img_name}.jpg')
    img = cv.imread(str(path))

    return img


def get_gt(img_name: str):
    path = Path(f'./data/gt_{img_name}.png')
    gt = cv.imread(str(path), 0).astype(np.uint8) * 255

    return gt


# pre-process

def sort_gt(gt: np.ndarray):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(gt, 4, cv.CV_8U)

    gt_ind_list = []

    for i in range(1, num_labels):  # label 0 is background
        coords = np.where(labels == i)
        inds = np.ravel_multi_index(np.vstack((coords[0], coords[1])), gt.shape)
        gt_ind_list.append(inds)

    gt_ind_list.sort(key=lambda i: -len(i))

    return gt_ind_list


def detect_edges(img: np.ndarray):
    # use cv cannyedge
    pass


# args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, default=CONFIG_PATH)
    parser.add_argument('--img_name', '-i', type=str, default=IMG_NAME)

    args1 = parser.parse_args()
    args1.config_path = Path(args1.config_path)

    return args1


if __name__ == '__main__':
    args = get_args()
    main(args.config_path, args.img_name)
