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
    current_img_id=1
    current_ann_id=1


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
        'id': current_img_id
        }

        images.append(image)

        plane_annotation = {
            'id': current_ann_id,
            'image_id': current_img_id,
            'bbox': pb_to_coco_bb(plane_bb_min, plane_bb_max),
            'area': 0,
            'iscrowd': 0,
            'category_id': 1,
            'segmentation': []
        }
        current_ann_id += 1
        annotations.append(plane_annotation)

        handle_annotation = {
            'id': current_ann_id,
            'image_id': current_img_id,
            'bbox': pb_to_coco_bb(handle_bb_min, handle_bb_max),
            'area': 0,
            'iscrowd': 0,
            'category_id': 2,
            'segmentation': []
        }
        annotations.append(handle_annotation)
        current_ann_id += 1
        current_img_id += 1


    categories.append({
        'supercategory': "Door_plane",
        'id': 1,
        'name': "door_plane"
    })
    categories.append({
        'supercategory': "Door_handle",
        'id': 2,
        'name': "door_handle"
    })

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

def pb_to_coco_bb(pb_bb_min, pb_bb_max):
    width=abs(pb_bb_min[0]-pb_bb_max[0])
    height= abs(pb_bb_min[1] - pb_bb_max[1])
    top_left_x=min(pb_bb_min[0], pb_bb_max[0])
    top_left_y = min(pb_bb_min[1], pb_bb_max[1])
    return [top_left_x, top_left_y, width, height]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No Feature-directory specified')
        exit(-1)
    path = sys.argv[1]
    main(path)
