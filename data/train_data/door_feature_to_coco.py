import os
import sys
import json

import numpy as np

def main(path):
    features = [f for f in os.listdir(path) if f.endswith('json')]
    img_path = 'data/train_data/images'
    feature_paths = []
    rgb_paths = []
    depth_paths = []
    seg_paths = []
    data_paths = []
    images = []
    annotations= []
    categories = []
    currentId=0


    for f in features:
        base_name = '.'.join(f.split('.')[:-1])
        feature_path = os.path.join(path, f)
        # print(base_name, feature_path)
        rgb_path = os.path.join(img_path, base_name) + '_rgb.png'
        depth_path = os.path.join(img_path, base_name) + '_depth.png'
        seg_path = os.path.join(img_path, base_name) + '_seg.png'
        in_json = open(os.path.join(path, f))
        d=json.load(in_json)
        plane_bb_min=d['object']['min']
        plane_bb_max=d['object']['max']
        handle_bb_min=d['handle']['min']
        handle_bb_max=d['handle']['max']


        image={
        'file_name': rgb_path,
        'height': 400,
        'width': 400,
        'id': 0,
        'id': currentId
        }
        images.append(image)

        annotation = {
            'id': currentId,
            'image_id': currentId,
            'bbox': [
                plane_bb_min[0],
                plane_bb_min[1],
                plane_bb_max[0],
                plane_bb_max[1]
            ],
            'area': 100,
            'iscrowd': 0,
            'category_id': 0,
            'segmentation': []
        }
        annotations.append(annotation)

        category= {
            'supercategory': "testcategory.obj",
            'id': currentId,
            'name': "door"
        }
        categories.append(category)

        currentId+=1

    all = {
        'images': images,
        'categories': categories,
        'annotations': annotations
    }

    with open('coco.json', 'w') as f:
        json.dump(all, f, indent=4)

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
