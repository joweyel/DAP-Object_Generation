import os
import sys
import json
import png

import numpy as np

def main(path):
    features = [f for f in os.listdir(path) if f.endswith('json')]
    img_path = path.replace("/features","/images")#'../DAP/DAP-Object_Generation/data/train_data/images'
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
        seg_path = os.path.join(img_path, base_name) + '_seg.png'
        in_json = open(os.path.join(path, f))
        d=json.load(in_json)
        handle_bb_min=d['handle']['min']
        handle_bb_max=d['handle']['max']

        #Implemented according to
        im=png.Reader(seg_path)
        direct=im.asDirect()
        image_2d=np.vstack([arr for arr in direct[2]])
        labelMap=np.reshape(image_2d, (direct[1], direct[0], direct[3]['planes']))
        plane_id = np.unique(labelMap.reshape(-1, labelMap.shape[2]), axis=0)[1]
        labelMask=(labelMap == plane_id).all(axis=2)
        labelMask=np.expand_dims(labelMask, axis=2)
        labelMask=labelMask.astype('uint8')

        a=np.where(labelMask != 0)
        xmin=int(min(a[1]))
        ymin=int(np.min(a[0]))
        xmax=int(np.max(a[1]))
        ymax=int(np.max(a[0]))

        plane_bbox=[xmin, ymin, xmax-xmin, ymax-ymin]
        plane_area=int(np.sum(labelMask))

        image={
        'file_name': base_name+ '_rgb.png',
        'height': 400,
        'width': 400,
        'id': current_img_id
        }

        images.append(image)

        plane_annotation = {
            'id': current_ann_id,
            'image_id': current_img_id,
            'bbox': plane_bbox,
            'area': plane_area,
            'iscrowd': 0,
            'category_id': 1,
            'segmentation': []
        }
        current_ann_id += 1
        annotations.append(plane_annotation)

        handle_area, handle_bbox = pb_to_coco_bb(handle_bb_min, handle_bb_max)
        handle_annotation = {
            'id': current_ann_id,
            'image_id': current_img_id,
            'bbox': handle_bbox,
            'area': handle_area,
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
    return width*height, [top_left_x, top_left_y, width, height]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No Feature-directory specified')
        exit(-1)
    path = sys.argv[1]
    main(path)
