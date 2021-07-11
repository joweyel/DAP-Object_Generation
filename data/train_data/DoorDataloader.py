import os
import json
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import pandas as pd


def getAABB(img, bbox, ax=None, color='r'):
    x_min, y_min = bbox[0, :]
    x_max, y_max = bbox[1, :]
    if ax is None:
        _, ax = plt.subplots()
    rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                             linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    return ax

def plotAABB(img, ax, idx):
    ax.imshow(img)
    # plt.savefig('test/bbox_img_{}.jpg'.format(idx), dpi=200)
    plt.show()
    
def check_bbox(bbox, w, h):
    '''
        [x0 y0]
        [x1 y1]
    '''
    if bbox[0, 0] < 0 or bbox[0, 0] >= w:
        return False
    if bbox[1, 0] < 0 or bbox[1, 0] >= w:
        return False
    if bbox[0, 1] < 0 or bbox[0, 0] >= h:
        return False
    if bbox[1, 1] < 0 or bbox[0, 0] >= h:
        return False
    return True

class DoorDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, train, transforms=None):
        self.root = root
        self.transforms = transforms
        self.train = train
        # self.Imgs = None
        # self.depthImgs = None
        # self.segImgs = None
        self.img_dir = os.path.join(root, 'images')
        self.feature_dir = os.path.join(root, 'features')

        # image prefixes: '_rgb.png', '_depth.png' and '_seg.png' get appended in __getitem__
        # json-name is the same as the prefix
        self.img_prefix = sorted(['_'.join(img.split('_')[:-1]) for img in os.listdir(self.img_dir)])


    def __getitem__(self, idx, debug=False):

        ## get the base-path and append relevant endings to load
        img_base = os.path.join(self.root, 'images', self.img_prefix[idx])
        rgb_path = img_base + '_rgb.png'
        depth_path = img_base + '_depth.png'
        seg_path = img_base + '_seg.png'

        ## load rgb-image, depth-image and segmentation image
        rgb_img = plt.imread(rgb_path)
        depth_img = plt.imread(depth_path)
        seg_img = plt.imread(seg_path)

        ## get feature-information corresponding to the images
        feature_path = os.path.join(self.root, 'features', self.img_prefix[idx]) + '.json'
        with open(feature_path, 'r') as json_file:
            features = json.load(json_file)

        '''
            ## get door bounding box from segmentation-mask
            # cls = np.unique(seg_img)
            # for c in cls:
            #     print(c)
            # n_cls = len(cls)
            # print(n_cls)
            # door_cls = cls[1]
            # # y_idx, x_idx = np.where(seg_img == door_cls)
            # out = np.where(seg_img == door_cls)
            # y_idx = out[1]; x_idx = out[0]
            # x_min, x_max = x_idx.min(), x_idx.max()
            # y_min, y_max = y_idx.min(), y_idx.max()
            # bbox_door = np.array([[x_min, y_min], [x_max, y_max]])
        '''
        
        boxes = []
        labels = []

        ## get handle buonding box from saved features
        
        # door
        bbox_door = np.array([features['object']['min'], features['object']['max']])
        boxes.append(bbox_door)
        labels.append(1)

        # handle
        bbox_handle = np.array([features['handle']['min'], features['handle']['max']])
        height, width, _ = rgb_img.shape
        if (check_bbox(bbox_handle, width, height)):
            boxes.append(bbox_handle)
            labels.append(2)


        print('#bboxes = ', len(boxes), labels)
        
        ## load the axis information
        axis = features['axis']
        x0, y0 = axis[0]
        x1, y1 = axis[1]


        if debug:
            # _, ax = plt.subplots(1, 3)
            # ax[0].imshow(rgb_img); ax[0].set_title('rgb')
            # ax[1].imshow(depth_img); ax[1].set_title('depth')
            # ax[2].imshow(seg_img); ax[2].set_title('seg')
            # plt.show()

            ax = getAABB(rgb_img, bbox_door)
            ax = getAABB(rgb_img, bbox_handle, ax, color='b')
            l = mlines.Line2D([x0, x1], [y0, y1], color='b',
                              linewidth=3)
            ax.add_line(l)
            plotAABB(rgb_img, ax, idx)
            


        target = dict()
        target['rgb']   = torch.tensor(rgb_img)
        target['depth'] = torch.tensor(depth_img)
        # target['seg'] = torch.tensor(seg_img)
        target['bbox']  = torch.tensor(boxes, dtype=torch.int32)
        target['label'] = torch.LongTensor(labels)
        # target['axis']  = axis

        print(target)

        return target

    def __len__(self):
        return len(self.img_prefix)

def main():
    dds = DoorDataset(root='./', train=True, data_file=None, transforms=None)

    for i in range(dds.__len__()):
        dds.__getitem__(i, True)
    
    print('len = ', dds.__len__())

if __name__ == '__main__':
    main()