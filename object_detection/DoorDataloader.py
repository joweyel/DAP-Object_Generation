import os
import png
import json
import cv2
import numpy as np
from scipy.ndimage.measurements import label
import torch
import torch.utils.data
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import pandas as pd

from shapely.geometry import box, Polygon, LineString, Point
import shapely

## Mapping
# door: 1, handle: 2, cabinet: 3, cupboard: 4

def clipAxis(img, axis):
    # [[y0, x0]
    #  [y1, x1]]
    
    l0 = axis[0, :][::-1]; l1 = axis[1, :][::-1]
    # print('l0 = ', l0, ' | l1 = ', l1)
    img_height, img_width = img.shape[:2]
    ret, p0, p1 = cv2.clipLine((0, 0, img_width, img_height), l0, l1)
    p0 = np.array(p0); p1 = np.array(p1)
    # print(ret, p0, p1)

    if ret:
        clipped_axis = np.array([p0, p1])
        return clipped_axis
    else:
        return axis


def getAABB(img, bbox, ax=None, color='r'):
    # bbox = [[y0, x0], -> min
    #         [y1, x1]] -> max
    y_min, y_max = bbox[:, 1]
    x_min, x_max = bbox[:, 0]

    if ax is None:
        _, ax = plt.subplots()
    rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                             linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    return ax

def plotAABB(img, ax, idx, title=''):
    ax.imshow(img)
    # plt.savefig('/home/ubuntupc/HESSENBOX-DA/DAP_data/Report_Material/Synth_data_annotation/Dataloader/bbox_img_{}.png'.format(idx), dpi=200)
    plt.title(title)
    # plt.savefig('/home/ubuntupc/HESSENBOX-DA/DAP_data/Report_Material/Synth_data_annotation/Dataloader/gt_data_' + str(idx) + '.png')
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

def clip_axis(bbox, img_width, img_height):
    _, p1, p2 = cv2.clipLine((0, 0, img_width, img_height), 
                               bbox[:, 0], bbox[:, 1])
    print(p1, p2)
    # return np.array([p1])
    return bbox
    pass

def clip_to_im(bbox, w, h):
    bbox[0][0] = np.clip(bbox[0][0], 0, w)
    bbox[1][0] = np.clip(bbox[1][0], 0, w)
    bbox[0][1] = np.clip(bbox[0][1], 0, h)
    bbox[1][1] = np.clip(bbox[1][1], 0, h)
    return bbox

def get_door_data(file):
    # door: 0, handle: 1, cabine: 2, cupboard: 3
    data = os.path.basename(file).split('_')
    door_type = data[3]
    if door_type[:3] == 'dos':
        return 0
    if door_type[:3] == 'cas':
        return 2
    if door_type[:3] == 'cus':
        return 3

def numpy2torch(img):
    torch_img = torch.tensor(img)
    torch_img = torch_img.unsqueeze(0)
    print('img = ', torch_img.shape)
    torch_img = torch_img.permute(0, 3, 1, 2)
    print('torch_img = ', torch_img.shape)
    return torch_img

def torch2numpy(tensor):
    """ Convert 3D PyTorch CHW tensor to 3D numpy HWC ndarray."""
    assert (tensor.dim() == 3)
    return np.transpose(tensor.numpy(), (1, 2, 0))

def get_data_files(root, type):
    print(type)
    coco_data_path = os.path.join(root, 'coco_' + type + '.json')
    with open(coco_data_path, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    # print(json.dumps(images, indent=4))
    files = []
    for img_feature in images:
        prefix = img_feature['file_name']
        prefix = '_'.join(prefix.split('_')[:-1])
        files.append(prefix)    
    return files

## TODO: Possiblity to load batches

class DoorDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, shuffle=False, batch_size=1, transforms=None):
        self.root = root
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = lambda x: x

        self.batch_size = batch_size

        # directory to load data from (train/test)

        self.data_type = 'train' if train else 'test'
        self.img_dir = os.path.join(self.root, 'images')
        self.feature_dir = os.path.join(self.root, 'features')

        # image appendixes: '_rgb.jpeg', '_depth.png' and '_seg.png' get appended in __getitem__
        # json-name is the same as the prefix
        self.img_list = get_data_files(root, self.data_type)
        if shuffle:
            self.img_list = np.random.shuffle(self.img_list)



    def ___getitem__(self, idx, debug=False):

        # get the base-path and append relevant endings to load
        img_base = os.path.join(self.img_dir, self.img_list[idx])
        # print(idx, img_base)

        # load images from specified directory
        rgb_path   = img_base + '_rgb.jpg'
        depth_path = img_base + '_depth.png'
        seg_path   = img_base + '_seg.png'

        # load rgb-image, depth-image and segmentation image
        rgb_img = plt.imread(rgb_path)
        rgb_img = self.transforms(rgb_img)
        depth_img = plt.imread(depth_path)
        seg_img = plt.imread(seg_path)

        ## get feature-information corresponding to the images
        feature_path = os.path.join(self.feature_dir, self.img_list[idx]) + '.json'
        with open(feature_path, 'r') as json_file:
            features = json.load(json_file)
        
        boxes = []
        labels = []

        ## get handle buonding box from saved features
        
        ## door
        # bbox_door = np.array([features['object']['min'], features['object']['max']])
        im = png.Reader(seg_path)
        direct = im.asDirect()
        image_2d = np.vstack([arr for arr in direct[2]])
        labelMap = np.reshape(image_2d, (direct[1], direct[0], direct[3]['planes']))
        plane_id = np.unique(labelMap.reshape(-1, labelMap.shape[2]), axis=0)[1]
        labelMask = (labelMap == plane_id).all(axis=2)
        labelMask = np.expand_dims(labelMask, axis=2)
        labelMask = labelMask.astype('uint8')
        a = np.where(labelMask != 0)
        ymin = int(np.min(a[0])); xmin = int(np.min(a[1]))
        ymax = int(np.max(a[0])); xmax = int(np.max(a[1]))
        bbox_door = np.array([[ymin, xmin], [ymax, xmax]])
        
        boxes.append(bbox_door)
        labels.append(0)

        print('bbox_door = ', bbox_door.shape)
        print('(ymin, xmin) = ', ymin, xmin)
        print('(ymax, xmax) = ', ymax, xmax)


        ## handle
        bbox_handle = np.array([features['handle']['min'][::-1], features['handle']['max'][::-1]])
        # bbox_handle = bbox_handle[:, ::-1]

        boxes.append(bbox_handle)
        labels.append(1)

        print('bbox_handle = ', bbox_door.shape)
        print('(ymin, xmin) = ', bbox_handle[0])
        print('(ymax, xmax) = ', bbox_handle[1])


        ## axis
        axis = np.array(features['axis'])
        axis = axis[:, ::-1]
        print('axis = \n', axis, 'axis.T = \n', axis.T)


        y = axis[:, 0]; x = axis[:, 1]

        bbox_axis = np.array([[y.min(), x.min()], [y.max(), x.max()]])


        if debug:
            ax = getAABB(rgb_img, bbox_door)
            ax = getAABB(rgb_img, bbox_handle, ax, color='b')
            # ax = getAABB(rgb_img, axis, ax, color='yellow')
            # y0, x0 = axis[0, :]
            # y1, x1 = axis[1, :]
            # p0 = [x0, x1]; p1 = [y0, y1]
            l = mlines.Line2D(x, y, color='b', linewidth=3)
            ax.add_line(l)
            # plotAABB(rgb_img, ax, idx, title=self.img_list[idx])
            plotAABB(rgb_img, ax, idx)



        ## TODO: needs to be in torch-Layout  (N, C, H, W)
        target = dict()
        # pytorch-tensor not used here since preprocessing requires numpy-array

        target['rgb'] = rgb_img
        target['bbox'] = torch.FloatTensor(boxes)
        target['labels'] = labels
        target['axis'] = torch.tensor(axis.copy())
        target['bbox_axis'] = torch.tensor(bbox_axis) # [[y0, x0], [y1, x1]]
        target['axis_position'] = int(features['axis_is_right']) # 0: left, 1: right

        return target



    def __getitem__(self, idx, debug=False):

        # get the base-path and append relevant endings to load
        img_base = os.path.join(self.img_dir, self.img_list[idx])
        door_type = get_door_data(img_base)

        # load images from specified directory
        rgb_path   = img_base + '_rgb.jpg'
        depth_path = img_base + '_depth.png'
        seg_path   = img_base + '_seg.png'

        ## load rgb-image, depth-image and segmentation image
        rgb_img = plt.imread(rgb_path)
        rgb_img = self.transforms(rgb_img)        
        # rgb_img = (rgb_img / 255.).astype(np.float32)
        depth_img = plt.imread(depth_path)
        seg_img   = plt.imread(seg_path)

        ## get feature-information corresponding to the images
        feature_path = os.path.join(self.feature_dir, self.img_list[idx]) + '.json'
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
        
        ## door
        # bbox_door_original = np.array([features['object']['min'], features['object']['max']])
        im = png.Reader(seg_path)
        direct = im.asDirect()
        image_2d = np.vstack([arr for arr in direct[2]])
        labelMap = np.reshape(image_2d, (direct[1], direct[0], direct[3]['planes']))
        plane_id = np.unique(labelMap.reshape(-1, labelMap.shape[2]), axis=0)[1]
        labelMask = (labelMap == plane_id).all(axis=2)
        labelMask = np.expand_dims(labelMask, axis=2)
        labelMask = labelMask.astype('uint8')

        a = np.where(labelMask != 0)
        xmin = int(min(a[1]));    ymin = int(np.min(a[0]))
        xmax = int(np.max(a[1])); ymax = int(np.max(a[0]))
        bbox_door = np.array([[xmin, ymin], [xmax, ymax]])


        boxes.append(bbox_door)
        labels.append(0)


        ##  handle
        bbox_handle = np.array([features['handle']['min'], features['handle']['max']])
        height, width, _ = rgb_img.shape
        bbox_handle = clip_to_im(bbox_handle, width, height)
        boxes.append(bbox_handle)
        labels.append(1) 


        ## load the axis information
        axis = np.array(features['axis'])
        # axis = clip_axis_point(axis[0, :], axis[1, :], height, width)
        # print('axis = ', axis)
        x0, y0 = axis[0, :]
        x1, y1 = axis[1, :] 

        axis = clipAxis(rgb_img, axis)

        axis_min_x = axis[:, 1].min()
        axis_max_x = axis[:, 1].max()
        axis_min_y = axis[:, 0].min()
        axis_max_y = axis[:, 0].max()
        axis_bbox = np.array([[axis_min_x, axis_min_y], [axis_max_x, axis_max_y]])
    

        # axis_bbox = clip_axis(axis_bbox, width, height)
        # x0, y0 = axis_bbox[0, :]
        # x1, y1 = axis_bbox[1, :]

        if debug:

            ax = getAABB(rgb_img, bbox_door)
            ax = getAABB(rgb_img, bbox_handle, ax, color='b')
            # ax = getAABB(rgb_img, axis_bbox, ax, color='yellow')
            l = mlines.Line2D([x0, x1], [y0, y1], color='b',
                              linewidth=3)
            ax.add_line(l)
            # print(self.img_list[idx])
            plotAABB(rgb_img, ax, idx)


        ## TODO: needs to be in torch-Layout  (N, C, H, W)
        target = dict()

        # pytorch-tensor not used here since preprocessing requires numpy-array
        target['rgb'] = rgb_img
        target['depth'] = depth_img
        target['seg'] = seg_img
        bbox_arr = torch.FloatTensor(boxes)
        target['bbox'] = bbox_arr
        target['label'] = labels
        target['axis'] = torch.tensor(axis)
        target['bbox_axis'] = torch.tensor(axis_bbox) # [[x0, y0], [x1, y1]]
        target['axis_position'] = int(features['axis_is_right']) # 0: left, 1: right
        
        '''
            # target['depth'] = numpy2torch(depth_img)
            # target['seg']   = numpy2torch(seg_img)
            # target['bbox']  = torch.tensor(boxes, dtype=torch.float32)
            # target['label'] = torch.LongTensor(labels)
            # print('AXIS:\t', target['axis'])   # [[lower_point], [upper_point]]
            # print(target)
        '''

        return target

    def __len__(self):
        return len(self.img_list)

def main():

    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[141.2988, 146.1302, 153.4230],
    #                          std=[0.3984, 0.3882, 0.3610])])
    # mu = [141.2988, 146.1302, 153.4230]
    # std = [0.3984, 0.3882, 0.3610]
    # std = [s * 255. for s in std]
    # data_transform = lambda x, mean=mu, std=std: (x - mean) / std

    # root = '/home/ubuntupc/HESSENBOX-DA/DAP_data/train_data_clipped/'
    # root = '/home/ubuntupc/HESSENBOX-DA/DAP_data/Test_axis_fixed/'
    # root = '/home/ubuntupc/HESSENBOX-DA/DAP_data/dataset_handle_front_bb_test/'
    # root = '/home/ubuntupc/HESSENBOX-DA/DAP_data/dataset_larger_bbs_low_compression/'
    

    # Mean:  tensor([0.4673, 0.4957, 0.5550])  std:  tensor([0.4673, 0.4439, 0.3502])
    ## root = '/home/ubuntupc/HESSENBOX-DA/DAP_data/1500 imgs min-max notation/'
    data_transform = None
    # root = '/home/ubuntupc/HESSENBOX-DA/DAP_data/train_data_big/'
    root = '/home/ubuntupc/HESSENBOX-DA/DAP_data/dataset_unseen_large_Dataloader/dataset/'


    dds = DoorDataset(root=root, train=False, transforms=data_transform)


    for i in range(dds.__len__()):
        dds.__getitem__(i, debug=True)
    
    print('len = ', dds.__len__())

if __name__ == '__main__':
    main()