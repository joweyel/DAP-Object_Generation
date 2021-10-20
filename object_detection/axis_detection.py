import detectron2
from detectron2.utils.logger import setup_logger
from numpy.core.arrayprint import printoptions
setup_logger()
import time
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches

from shapely.geometry import box


import argparse

import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.losses.object_detection import giou_loss, iou_loss

print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.8")

import random
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import ImageList
from detectron2.structures.image_list import _as_tensor

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build
from detectron2.utils.visualizer import ColorMode

from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# loading of COCO-Datasets is especially easy
from detectron2.data.datasets import register_coco_instances

# For Trainging
from detectron2.engine import DefaultTrainer

# For saving
from detectron2.checkpoint import DetectionCheckpointer, Checkpointer

# Custom dataloader
from DoorDataloader import *

def getAABB(img, bbox, ax=None, color='r'):
    # bbox = [[y0, x0], -> min
    #         [y1, x1]] -> max
    # y_min, y_max = bbox[:, 1]
    # x_min, x_max = bbox[:, 0]
    x_min = bbox[0]; y_min = bbox[1]
    x_max = bbox[2]; y_max = bbox[3]
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

    
def get_center(bbox):
    # x1, y1, x2, y2
    # center = [x2 - x1, y_2 - y1]
    x1 = bbox[0]; y1 = bbox[1]; x2 = bbox[2]; y2 = bbox[3]
    center = torch.tensor([(x1 + x2)/2, (y1 + y2)/2])
    # print('CENTER:')
    # print(bbox, center)
    return center

def bb_to_edgepoints(bb):
    bb_min_x = min(bb[0][0], bb[1][0])
    bb_max_x = max(bb[0][0], bb[1][0])
    bb_min_y = min(bb[0][1], bb[1][1])
    bb_max_y = max(bb[0][1], bb[1][1])
    return [[bb_min_x, bb_min_y], [bb_min_x, bb_max_y], [bb_max_x, bb_max_y], [bb_max_x, bb_min_y]]

def intersection(bbox_door, bbox_handle):
    bbox_door = torch.tensor(bbox_door).reshape(2, 2)
    bbox_handle = torch.tensor(bbox_handle).reshape(2, 2)
    door_edgepoints = bb_to_edgepoints(bbox_door)
    handle_edgepoints = bb_to_edgepoints(bbox_handle)
    door_shape   = Polygon(door_edgepoints)
    handle_shape = Polygon(handle_edgepoints)

    intersection = handle_shape.intersection(door_shape).area
    handle_area = handle_shape.area
    # print('I(H && D) = ', intersection)
    # print('I(H) = ', handle_area)
    # print('I / I_HD = ', intersection / handle_area)
    I = intersection / handle_area
    return I

def dist_door_center(door_center, handle_centers):
    # TODO
    door_y = door_center[1]
    handle_ys = [hc[1] for hc in handle_centers]
    distances = torch.tensor([abs(h_y - door_y) for h_y in handle_ys])
    print('D = ', distances)
    return distances


CONFIG_PATH = './cfgs/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AxisPrediction(nn.Module):
    def __init__(self, bbox_cfg, input_size, transform=lambda x: x, model_type='heuristic'):
        super().__init__()

        ## Model 1: BBox-Prediction
        # load config
        self.bbox_cfg = bbox_cfg
        self.input_format = self.bbox_cfg.INPUT.FORMAT
        self.model_type = model_type
        self.normalizer = transform

        if self.model_type == 'heuristic':
            self.bbox_model = DefaultPredictor(self.bbox_cfg)
        elif self.model_type == 'DNN':
            self.bbox_model = build_model(self.bbox_cfg)
            DetectionCheckpointer(self.bbox_model).load(self.bbox_cfg.MODEL.WEIGHTS)

            p = 0.4
            ## Model 2: Axis-Prediction (preliminary)
            self.axis_model = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(1024, 4),
                nn.ReLU())
            self.axis_model.requires_grad_()
        else:
            print('No method specified!')
            exit(-1)
        
        self.bbox_device = torch.device(self.bbox_cfg.MODEL.DEVICE)


    def preprocess_image(self, batched_inputs):
        """
            # Based upon: detectron2/modeling/meta_arch/rcnn.py#L185
            Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def check_inside(self, bbox, center):
        x1 = bbox[0]; y1 = bbox[1]; x2 = bbox[2]; y2 = bbox[3]
        check_x = True if (x1 < center[0] and center[0] < x2) else False
        check_y = True if (x1 < center[0] and center[0] < x2) else False
        check = check_x and check_y
        print(center, ' in ', bbox, ' === >>> ', check)
        return check
    

    def process_data(self, img, pred_boxes, scores, pred_classes):
        pred_boxes = pred_boxes.to('cpu')
        scores = scores.to('cpu')
        pred_classes = pred_classes.to('cpu')
        # print('pred_boxes: ', pred_boxes)
        # print('scores: ', scores) 
        # print('pred_classes: ', pred_classes)

        # get the door-bboxes, the centers
        door_idx = np.where(pred_classes == 0)
        door_bboxes = pred_boxes[door_idx]
        door_scores = scores[door_idx]
        door_centers = [get_center(box) for box in door_bboxes]
        # print(door_centers)

        # get the handle-bboxes, the centers
        handle_idx = np.where(pred_classes == 1)
        handle_bboxes = pred_boxes[handle_idx]
        handle_scores = scores[handle_idx]
        handle_centers = [get_center(box) for box in handle_bboxes]

        # print('handle_centers: ', handle_centers)
        # print(door_idx, scores[door_idx], door_bboxes)
        # print(handle_idx, scores[handle_idx], handle_bboxes)

        # iterate over the doors
        # choose the door with the highest probability if handle is present

        door = []
        handle = []

        threshold = 0.75
        '''
        for d_center, d_box in zip(door_centers, door_bboxes):
            print(d_center, d_box)
            doable_handle_bboxes = []
            for h_box in handle_bboxes:
                if intersection(d_box, h_box) > threshold and self.check_inside(d_box, get_center(h_box)):
                    doable_handle_bboxes.append(h_box)
            handle_centers = [get_center(box) for box in doable_handle_bboxes]
            print('handle_centers = ', handle_centers)
            # now all possible handles are extracted
            # TODO: check if handle is doable

            dists = dist_door_center(d_center, handle_centers)
            if len(dists) > 0:
                min_dist = torch.argmin(dists)
                best_h_box = doable_handle_bboxes[min_dist]
                ax = getAABB(img, d_box)
                ax = getAABB(img, best_h_box, ax, color='b')
                plotAABB(img, ax, 0) #, title=str(check))

            # for h_box in doable_handle_bboxes:
            #    ax = getAABB(img, d_box)
            #    ax = getAABB(img, h_box, ax, color='b')
            #    plotAABB(img, ax, 0) #, title=str(check))


            # for h_center, h_box in zip(handle_centers, doable_handle_bboxes):

                # # check = self.check_inside(d_box, h_center)  # must be true
                # # w_I = intersection(d_box, h_box)            # must exceed 0.75
                # # dist_door_center(d_box, handle_bboxes)


                # # if check: # and w_I > 0.75:
                # ax = getAABB(img, d_box)
                # ax = getAABB(img, h_box, ax, color='b')
                # plotAABB(img, ax, 0), # title=str(check))

        '''
        
        doors = []
        handles = []

        for idx_d, (d_center, d_box, d_score) in enumerate(zip(door_centers, door_bboxes, door_scores)):
            print(d_center, d_box)
            
            d_width  = d_box[0] - d_box[2]
            d_height = d_box[1] - d_box[3]

            ratio = (d_height / d_width)
            print(ratio) 

            dists = dist_door_center(d_center, handle_centers)
            print('D = ', dists, ' | door_Center = ', d_center)

            for idx_h, (h_center, h_box, h_score) in enumerate(zip(handle_centers, handle_bboxes, handle_scores)):
                
                check = self.check_inside(d_box, h_center)  # must be true
                w_I = intersection(d_box, h_box)            # must exceed 0.75
                # dist_door_center(d_box, handle_bboxes)


                if check and w_I > 0.75:
                    # ax = getAABB(img, d_box)
                    # ax = getAABB(img, h_box, ax, color='b')
                    # plotAABB(img, ax, 0, title=str(check))
                    doors.append(d_box)
                    handles.append(h_box)
            
        return doors, handles





        ## iterate over the available doors
        # for 


    def forward(self, original_img, idx):

         # use the pretrained bbox-prediction model
        if self.model_type == 'heuristic':
            with torch.no_grad():
                outputs = self.bbox_model(original_img)['instances']
                fields = outputs.get_fields()
                pred_boxes = fields['pred_boxes']
                scores = fields['scores']
                pred_classes = fields['pred_classes']
                doors, handles = self.process_data(original_img, pred_boxes, scores, pred_classes)

                # Door and handle both present
                if len(doors) > 0 and len(handles) > 0:
                    # ax = getAABB(original_img, doors[0])
                    # ax = getAABB(original_img, handles[0], ax, color='b')
                    # plotAABB(original_img, ax, 0)

                    bbox_door = doors[0].reshape(2, 2)
                    bbox_handle = handles[0].reshape(2, 2)
                    axis_pred = detect_axis(bbox_door, bbox_handle, original_img, idx, visualize=False)

                # either door or handle not present -> not able to infer the articulation
                else:
                    axis_pred = axis_pred = torch.zeros((2, 2), dtype=torch.float32)

                '''
                    # maps class to the first instance of the class in predictions
                    # because scores are ordered descending
                    # boxes_idx = {}
                    # for i in range(2):
                    #     boxes_idx[i] = int((pred_classes==i).nonzero(as_tuple=True)[0][0])
                    # print('boxes_idx = ', boxes_idx)
                '''

                '''
                    # boxes_idx = {i: int((pred_classes==i).nonzero(as_tuple=True)[0][0]) for i in range(2) if i in pred_classes}
                    # # check if all relevant bboxes are present
                    # boxes_present = [i in pred_classes for i in range(2)]
                    # if all(boxes_present):
                    #     door_idx = boxes_idx[0] # int((pred_classes==0).nonzero(as_tuple=True)[0][0])
                    #     handle_idx = boxes_idx[1] # int((pred_classes==1).nonzero(as_tuple=True)[0][0])
                    #     bbox_door = pred_boxes[door_idx].tensor.reshape(2, 2)
                    #     bbox_handle = pred_boxes[handle_idx].tensor.reshape(2, 2)
                    #     axis_pred = detect_axis(bbox_door.cpu(), bbox_handle.cpu(), original_img)
                    # else:
                    #     axis_pred = axis_pred = torch.zeros((2, 2), dtype=torch.float32)
                '''

                # returns axis, all prediction data + indices of best predictions
                # return axis_pred, outputs, boxes_idx
                return outputs, axis_pred, doors, handles

        if self.model_type == 'DNN':
            self.bbox_model.eval()

            with torch.no_grad():
                if self.input_format == 'RGB':
                    original_img = original_img[:, :, ::-1]

                height, width = original_img.shape[:2]
                img = original_img.copy()
                image = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))
                inputs = [{'image': image, 'height': height, 'width': width}]
    
                ## TODO: Inference on trained model
                images = [x['image'].to(self.bbox_device) for x in inputs]
                images = [self.normalizer(x) for x in images]
                images = ImageList.from_tensors(images, self.bbox_model.backbone.size_divisibility)
    
                # manually execute the main parts of the detectron model
                features = self.bbox_model.backbone(images.tensor)
                proposals, _ = self.bbox_model.proposal_generator(images, features) # somehow ImageList is needed here
                instances, _ = self.bbox_model.roi_heads(images, features, proposals)
                outputs = instances[0]

                p6 = features['p6']
                print('p6 = ', p6.shape)
    
                
                # img = images.tensor.squeeze().detach().cpu().numpy()
                img = original_img.copy()
                print('....................... ', img.shape)
                # images = np.transpose(img, (2, 1, 0))
                images = img
                print('----------------------- ', images.shape)
                v = Visualizer(images[:, :, ::-1], scale=1.0)
                # ## v = Visualizer(im[:, :, ::-1], metadata=test_dataset_metadata, scale=0.5)
                out = v.draw_instance_predictions(instances[0].to("cpu"))
                out_img = out.get_image()[:, :, ::-1]
                # # splt.imsave('prediction_output/' + str(idx) + '_prediction.jpg', out_img)
                plt.imshow(out_img)
                # # cv2.imshow(str(idx) + '_prediction', out_img)
                
                plt.show()


            return outputs


    def freeze_bbox_model(self, parameter_list=[]):
        if len(parameter_list) > 0:
            params = self.bbox_model.parameters()
        else:
            params = parameter_list

        for parameter in params:
            parameter.requires_grad = False

def print_bboxes(img, outputs, metadata):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs.to("cpu"))
    out_img = out.get_image()[:, :, ::-1]
    plt.imshow(out_img)
    plt.show()

def detect_axis(bbox_door, bbox_handle, img, idx, visualize=False):
    door_center = bbox_door.mean(axis=0)
    handle_center = bbox_handle.mean(axis=0)

    x_min_d, y_min_d = bbox_door[0, :]
    x_max_d, y_max_d = bbox_door[1, :]    
    
    x_min_h, y_min_h = bbox_handle[0, :]
    x_max_h, y_max_h = bbox_handle[1, :]
    
    # check if axis is on left or right of door
    # left = 0, right = 1
    if handle_center[0] > door_center[0]: # handle-center is right of door-center
        axis_pos = 0
    else:
        axis_pos = 1

    # get the sides of the door
    left_edge = torch.tensor([[x_min_d, y_max_d], 
                              [x_min_d, y_min_d]])
    right_edge = torch.tensor([[x_max_d, y_max_d], 
                               [x_max_d, y_min_d]])
    edges = torch.stack((left_edge, right_edge), axis=0)
    edge = edges[axis_pos]

    if visualize:
        _, ax = plt.subplots()
        rect_door = patches.Rectangle((x_min_d, y_min_d), x_max_d-x_min_d, y_max_d-y_min_d,
                                        linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_door)
        rect_handle = patches.Rectangle((x_min_h, y_min_h), x_max_h-x_min_h, y_max_h-y_min_h,
                                        linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect_handle)
        door_circle = patches.Circle(door_center, radius=3, color='r')
        handle_circle = patches.Circle(handle_center, radius=3, color='b')
        ax.add_patch(door_circle)
        ax.add_patch(handle_circle)
        
        axis_line = mlines.Line2D(edge[:, 0], edge[:, 1], color='yellow', linewidth=4)
        ax.add_line(axis_line)
        ax.imshow(img)
        plt.show()

    return edge

def plot_prediction_heuristic(img, bbox_door, bbox_handle, axis, idx, save=False):
    _, ax = plt.subplots(1)
    l_pred = mlines.Line2D([axis[0, 0], axis[1, 0]],
                               [axis[0, 1], axis[1, 1]],
                               color='yellow', linewidth=3)
    ax.add_line(l_pred)
    bbox_door = bbox_door.reshape(2, 2)
    bbox_handle = bbox_handle.reshape(2, 2)

    print('bbox_door = ', bbox_door.shape)
    y_min_d, y_max_d = bbox_door[:, 1]
    x_min_d, x_max_d = bbox_door[:, 0]
    rect_door = patches.Rectangle((x_min_d, y_min_d), x_max_d-x_min_d, 
                                  y_max_d-y_min_d, linewidth=3, edgecolor='red', 
                                  facecolor='none')
    ax.add_patch(rect_door)

    y_min_h, y_max_h = bbox_handle[:, 1]
    x_min_h, x_max_h = bbox_handle[:, 0]
    rect_handle = patches.Rectangle((x_min_h, y_min_h), x_max_h-x_min_h, 
                                    y_max_h-y_min_h, linewidth=3, edgecolor='blue', 
                                    facecolor='none')
    ax.add_patch(rect_handle)

    ax.imshow(img)
    plt.savefig('/home/ubuntupc/HESSENBOX-DA/DAP_data/detectron2/Tutorial/prediction_output/pred_synth_axis/prediction_' + str(idx) + '.png')
    # plt.show()

    del ax

def plot_prediction(img, outputs, bbox_idx, axis, save=False):

    bbox_door = outputs.pred_boxes[bbox_idx[0]].tensor.reshape(2, 2)
    bbox_door = bbox_door.cpu()
    bbox_handle = outputs.pred_boxes[bbox_idx[1]].tensor.reshape(2, 2)
    bbox_handle = bbox_handle.cpu()
    
    _, ax = plt.subplots(1)
    l_pred = mlines.Line2D([axis[0, 0], axis[1, 0]],
                               [axis[0, 1], axis[1, 1]],
                               color='r', linewidth=3)
    ax.add_line(l_pred)

    y_min_d, y_max_d = bbox_door[:, 1]
    x_min_d, x_max_d = bbox_door[:, 0]
    rect_door = patches.Rectangle((x_min_d, y_min_d), x_max_d-x_min_d, 
                                  y_max_d-y_min_d, linewidth=4, edgecolor='yellow', 
                                  facecolor='none')
    ax.add_patch(rect_door)

    y_min_h, y_max_h = bbox_handle[:, 1]
    x_min_h, x_max_h = bbox_handle[:, 0]
    rect_handle = patches.Rectangle((x_min_h, y_min_h), x_max_h-x_min_h, 
                                    y_max_h-y_min_h, linewidth=4, edgecolor='blue', 
                                    facecolor='none')
    ax.add_patch(rect_handle)

    ax.imshow(img); plt.show()

    del ax
    
def axis_loss(output, target):    
    val = torch.mean((output - target)**2)
    return val

def main():
    
    '''
        DOOR_DATA = os.environ['DOORDATA']
        train_json = DOOR_DATA + 'train_data_big/coco_train.json'
        test_json  = DOOR_DATA + 'train_data_big/coco_test.json'
        image_root = DOOR_DATA + 'train_data_big/images/'
        image_stat = DOOR_DATA + 'train_data_big/data_stats.json'
        register_coco_instances(name='door_data_train', 
                                metadata={}, json_file=train_json, 
                                image_root=image_root)
        register_coco_instances(name='door_data_test', 
                                metadata={}, 
                                json_file=test_json, 
                                image_root=image_root)

        # get the registered dataset to use by name
        train_dataset = DatasetCatalog.get('door_data_train')
        train_dataset_metadata = MetadataCatalog.get('door_data_train')
        test_dataset = DatasetCatalog.get('door_data_test')
        test_dataset_metadata = MetadataCatalog.get('door_data_test')
    '''

    cfg_file = os.path.join(CONFIG_PATH, 'cfg_bbox_mode.yaml')
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    print(cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

    ## DNN-Version
    feature_size = 12544 # Dimension of flattened optput of p6-part of ResNet50
    feature_size += 8    # p6 + 2 * BBox
    
    # standardization of images
    mu = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1).to(device)
    data_transform = lambda x, mean=mu, std=std: (x - mean) / std
    model_type = 'heuristic'

    net = AxisPrediction(bbox_cfg=cfg, 
                         input_size=feature_size, 
                         transform=data_transform, 
                         model_type=model_type)

    net.to(device)


    #####################
    ##### real data #####
    #####################
    real_path = os.environ['REAL_DOORS']

    real_imgs = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith('jpg')]
    for idx, img_path in enumerate(real_imgs):
        print(img_path)
        real_img = plt.imread(img_path)
        plt.imshow(real_img); plt.show()
        outputs, axis_pred, doors, handles = net(real_img, idx)

        # visualizing the predictions
        v = Visualizer(real_img[:, :, ::-1], scale=1.0)
        out = v.draw_instance_predictions(outputs.to("cpu"))
        out_img = out.get_image()[:, :, ::-1]
        plt.imshow(out_img)
        plt.show()

        # visualize the "best choice" of door and handle
        if len(doors) > 0 and len(handles) > 0:
            bbox_door = doors[0].reshape(2, 2)
            bbox_handle = handles[0].reshape(2, 2)

            x_min_d, y_min_d = bbox_door[0, :]
            x_max_d, y_max_d = bbox_door[1, :]    

            x_min_h, y_min_h = bbox_handle[0, :]
            x_max_h, y_max_h = bbox_handle[1, :]

            _, ax = plt.subplots()
            rect_door = patches.Rectangle((x_min_d, y_min_d), x_max_d-x_min_d, y_max_d-y_min_d,
                                            linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_door)
            rect_handle = patches.Rectangle((x_min_h, y_min_h), x_max_h-x_min_h, y_max_h-y_min_h,
                                            linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect_handle)
            axis_line = mlines.Line2D(axis_pred[:, 0], axis_pred[:, 1], color='yellow', linewidth=4)
            ax.add_line(axis_line)
            ax.imshow(real_img)
            plt.show()

    return

    
    ######################
    ### synthetic data ###
    ######################

    # data_path = '/home/ubuntupc/HESSENBOX-DA/DAP_data/train_data_big/'
    data_path = '/home/ubuntupc/HESSENBOX-DA/DAP_data/dataset_unseen_large_Dataloader/dataset/'
    trainloader = DoorDataset(root=data_path, train=True)
    testloader = DoorDataset(root=data_path, train=False)

    # compute error on test-data
    # N = len(testloader)
    # cost = 0
    # costs = np.zeros(N)
    criterion = axis_loss
    criterionGIoU = giou_loss

    running_loss=0
    losses = []
    time_list = []

    for i, data in enumerate(testloader, 0):

        rgb_img = data['rgb']
        axis = data['axis']
        bbox_gt = data['bbox']
        gt_lbl = data['label']
        bbox_axis = data['bbox_axis']
        axis_position = data['axis_position'] # 0: left, 1: right
        print(bbox_axis.shape)

        if model_type == 'heuristic':
            outputs, axis_pred, doors, handles = net(rgb_img)
            if len(doors) < 1 or len(handles) < 1: # doors or handles == 0 -> no axis detectable
                print('Not able to predict axis!')
                continue
            plot_prediction_heuristic(rgb_img, doors[0], handles[0], axis_pred, i)
        
        if model_type == 'DNN':
            outputs = net(rgb_img)
            print(outputs.get_fields())
            fields = outputs.get_fields()
            pred_boxes = fields['pred_boxes']
            scores = fields['scores']
            pred_classes = fields['pred_classes']

            print(pred_boxes)
            print(scores)
            print(pred_classes)

        continue 

        # loss to be evaluated!
        print(axis_pred.shape,'\n')
        loss = criterion(axis_pred.flatten(), axis.flatten())
        loss = np.linalg.norm(axis_pred, axis, 2)
        
        pred_GIoU = axis_pred.flatten().unsqueeze(0)
        target_GIoU = bbox_axis.flatten().unsqueeze(0)
        lossGIoU = criterionGIoU(pred_GIoU, target_GIoU)
        
        print('MES: ', loss, ' | GIoU: ', lossGIoU)

        running_loss += loss.item()

        loss_i = running_loss / len(testloader)
        print(i, 'Loss: ', loss_i)
        losses.append(loss_i)

        # if i >= 1000:
        #     break
    
    plt.plot(losses)
    with open('losses.txt', 'w') as f:
        for item in losses:
            f.write("%s\n" % item)

    
    # net.freeze_bbox_model()
    # 
    # criterion = giou_loss # axis_loss
    # optimizer = optim.SGD(net.axis_model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    main()