import os
import sys

import numpy as np

def main(path):
    features = [f for f in os.listdir(path) if f.endswith('json')]
    img_path = 'images'
    feature_paths = []
    rgb_paths = []
    depth_paths = []
    seg_paths = []
    data_paths = []

    for f in features:
        base_name = '.'.join(f.split('.')[:-1])
        feature_path = os.path.join(path, f)
        # print(base_name, feature_path)
        rgb_path = os.path.join(img_path, base_name) + '_rgb.png'
        depth_path = os.path.join(img_path, base_name) + '_depth.png'
        seg_path = os.path.join(img_path, base_name) + '_seg.png'

        print(os.path.exists(rgb_path), os.path.exists(depth_path), os.path.exists(seg_path))
        print(rgb_path)
        print(depth_path)
        print(seg_path, '\n')
        

        # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
        ## generate json (bbox-detection + axis-prediction):
        # 1.) images
        # 2.) categories
        # 3.) annotations

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No Feature-directory specified')
        exit(-1)
    path = sys.argv[1]
    main(path)