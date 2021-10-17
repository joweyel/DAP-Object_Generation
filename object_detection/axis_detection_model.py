import detectron2
from detectron2.utils.logger import setup_logger
from numpy.core.arrayprint import printoptions
setup_logger()
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches

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

CONFIG_PATH = './cfgs/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numpy2torch(array):
    """ Converts 3D numpy HWC ndarray to 3D PyTorch CHW tensor."""
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    elif array.ndim == 2:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array)


def torch2numpy(tensor):
    """ Convert 3D PyTorch CHW tensor to 3D numpy HWC ndarray."""
    assert (tensor.dim() == 3)
    return np.transpose(tensor.numpy(), (1, 2, 0))


class AxisPrediction(nn.Module):
    def __init__(self, bbox_cfg, input_size, normalizer=lambda x: x, bbox_model_weights='output/model_final.pth'):
        super().__init__()

        ## Model 1: BBox-Prediction

        # load config and model
        self.bbox_cfg = bbox_cfg
        self.bbox_model = build_model(bbox_cfg) # build pretrained model from config
        self.bbox_device = torch.device(self.bbox_cfg.MODEL.DEVICE)
        DetectionCheckpointer(self.bbox_model).load(self.bbox_cfg.MODEL.WEIGHTS)

        # image preprocessing-parameters for the DNN-Approach
        self.input_format = self.bbox_cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.normalizer = normalizer # (x - pixel_mean) / pixel_std ## added later
        self.aug = T.ResizeShortestEdge(
            [self.bbox_cfg.INPUT.MIN_SIZE_TEST, 
             self.bbox_cfg.INPUT.MIN_SIZE_TEST], 
             self.bbox_cfg.INPUT.MAX_SIZE_TEST)


        p = 0.4
        ## Model 2: Axis-Prediction (preliminary)
        self.axis_model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1024, 4),
            nn.ReLU()
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 4),   # (x0,y0,x1,y1) Axis Position
            # nn.ReLU()
        )
        self.axis_model.requires_grad_()


    def freeze_bbox_model(self, parameter_list=[]):
        if len(parameter_list) > 0:
            params = self.bbox_model.parameters()
        else:
            params = parameter_list

        for parameter in params:
            parameter.requires_grad = False


    def preprocess_image(self, batched_inputs):
        """
            # Based upon: detectron2/modeling/meta_arch/rcnn.py#L185
            Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def axis_prediction(self, original_image, bbox_gt, gt_lbl):
        self.bbox_model.eval()

        with torch.no_grad():
            pass


    def forward(self, original_image, bbox_gt, gt_lbl):
        ## Preprocessing for backbone-Input
        # https://github.com/facebookresearch/detectron2/blob/4bd27960e1c56eb8950b04f24c60b845d5840d8a/detectron2/modeling/meta_arch/rcnn.py#L185
        ## Inference prcedure for a single image
        # DefaultPredictor and __call__ from detectron2/engine/defaults.py

        '''
            # Based upon "__call__" of "DefaultPredictor" in defaults.py
            Args:
                input_image (np.ndarray) of shape (H, W, C) and with BGR-Channels
        '''
        self.bbox_model.eval() # use the pretrained bbox-prediction model

        # print('bbox_gt = ', bbox_gt.shape, bbox_gt.dtype, type(bbox_gt), bbox_gt)
        with torch.no_grad():   # Image-Preprocessing
            if self.input_format == 'RGB':
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)


            image = original_image.copy()
            image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            inputs = [{'image': image, 'height': height, 'width': width}]


            # TODO: Inference on trained model
            # https://github.com/facebookresearch/detectron2/blob/4bd27960e1c56eb8950b04f24c60b845d5840d8a/detectron2/modeling/meta_arch/rcnn.py#L185
            # preprocessing images (backbone-input)
            images = [x['image'].to(self.bbox_device) for x in inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.bbox_model.backbone.size_divisibility)


            # manually execute the main parts of the detectron model
            features = self.bbox_model.backbone(images.tensor)
            proposals, _ = self.bbox_model.proposal_generator(images, features) # somehow ImageList is needed here
            instances, _ = self.bbox_model.roi_heads(images, features, proposals)


            # get latest features from the FPN-Part of the Resnet-50
            # p6 = features['p6']

            # bbox-predictions: like "predictions = self.model([inputs])[0]"
            outputs = instances[0]

            # all outputs 
            fields = outputs.get_fields()            
            pred_boxes = fields['pred_boxes']
            scores = fields['scores']
            pred_classes = fields['pred_classes']
            
            print('pred_classes = ', pred_classes)
            '''
                print(pred_classes.dtype)
                all_cls_idx = []
                for cls in [0, 1]:
                    num_oc = (pred_classes == cls).sum()
                    if cls in pred_classes:
                        # print('Class: {} occured {} times'.format(cls, num_oc))
                        cls_idx = torch.where(pred_classes == cls)[0][0]
                        # print('cls_idx = ', cls_idx)
                        all_cls_idx.append(int(cls_idx.detach().cpu()))
                        # print('CLS = ', cls_idx, ' | argmax CLS = ', scores[cls_idx])

                print('all_cls_idx = ', all_cls_idx)
                if len(all_cls_idx) > 1:
                    pred_boxes = pred_boxes[all_cls_idx]
                    scores = scores[all_cls_idx]
                    pred_classes = pred_classes[all_cls_idx]
                    outputs.set('num_instances', len(all_cls_idx))
                    outputs.set('pred_boxes', pred_boxes)
                    outputs.set('scores', scores)
                    outputs.set('pred_classes', pred_classes)
                print('pred_boxes => ', pred_boxes)
                print('scores => ', scores)
                print('pred_classes => ', pred_classes)
            '''
    
            # check if bbox present, otherwise add gt-bbox
            # TODO: testing if the Instances-Class version works better
            # @staticmethod
            # def cat(instance_lists: List["Instances"]) -> "Instances":
            #    ...

            
            # gt_added = False
            # for i in range(2):
            #     if i not in pred_classes:
            #         # print('class [{}] was not found!'.format(i))
            #         gt_added = True
            #         pred_classes = torch.cat((pred_classes, torch.tensor([i]).to(device)))
            #         scores = torch.cat((scores, torch.tensor([1.0]).to(device)))
            #         handle_box = bbox_gt[i, :, :].to(device)
            #         handle_box = handle_box.reshape(-1, 4)
            #         pred_boxes.tensor = torch.cat((pred_boxes.tensor, handle_box), axis=0)
            #         outputs.set('pred_boxes', pred_boxes)
            #         outputs.set('scores', scores)
            #         outputs.set('pred_classes', pred_classes)
            
        
        ## end of `torch.no_grad()`


        # '''
        print('Visualize!')
        v = Visualizer(original_image[:, :, ::-1], scale=1.0)
        ## v = Visualizer(im[:, :, ::-1], metadata=test_dataset_metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs.to("cpu"))
        out_img = out.get_image()[:, :, ::-1]
        # plt.imsave('prediction_output/' + str(idx) + '_prediction.jpg', out_img)
        # if gt_added:
        #     plt.title('added gt-data')    
        plt.imshow(out_img)
        # cv2.imshow(str(idx) + '_prediction', out_img)
        plt.show()
        ## plt.pause(0.5)
        # '''

    
        ## Part 2: make line predictions
        # TODO: pass the informations in axis prediction model
        # currently: 2-FCN Layer as placeholder for the architecture

        # get the predictions for every class by score
        door_idx = int((pred_classes==0).nonzero(as_tuple=True)[0][0])
        handle_idx = int((pred_classes==1).nonzero(as_tuple=True)[0][0])
        # feature_inputs = torch.cat((p6.flatten(),
        #                             pred_boxes[door_idx].tensor.flatten(),     # TODO: use only gt-boxes to test whats up
        #                             pred_boxes[handle_idx].tensor.flatten()))

        ## feature_inputs = torch.cat((p6.flatten(),
        ##                             bbox_gt[gt_lbl[0]].flatten().to(device),     # TODO: use only gt-boxes to test whats up
        ##                             bbox_gt[gt_lbl[1]].flatten().to(device)))





        # x = torch.flatten(feature_inputs.to(device))
        # x = self.axis_model(x)

        # for params in self.axis_model.parameters():
        #     print(params)
        ## print('x = ', x.shape)

        # return x

        bbox_door = pred_boxes[door_idx].tensor.reshape(2, 2)
        bbox_handle = pred_boxes[handle_idx].tensor.reshape(2, 2)
        axis = detect_axis(bbox_door, bbox_handle, original_image)

        return axis


    def _forward(self, images):
        # print('FORWARD: ', images.shape, images.dtype)
        # img = images.squeeze().permute(1, 2, 0); img = img.cpu().detach().numpy(); plt.imshow(img); plt.show()
        
        '''
          To see how variable sized input into partial execution of the 
          backbone woks see: ImageList is crucial for padding of odd-sized images
          https://github.com/facebookresearch/detectron2/issues/852,
          https://github.com/facebookresearch/detectron2/issues/5#issuecomment-693273320
          https://github.com/facebookresearch/detectron2/blob/4bd27960e1c56eb8950b04f24c60b845d5840d8a/detectron2/modeling/meta_arch/rcnn.py#L185
        '''
        norm_images = [self.normalizer(im) for im in images]
        images_list = ImageList.from_tensors(norm_images, self.bbox_model.backbone.size_divisibility)

        ## Part 1: use bbox-prediction model
        # manually do the last steps of pretrained model to get access to
        # intermediate representations

        self.bbox_model.eval() # use the pretrained bbox-prediction model

        with torch.no_grad():
            im = images_list.tensor
            features = self.bbox_model.backbone(im) # access features to use

            proposals, _ = self.bbox_model.proposal_generator(images_list, features) # somehow ImageList is needed here
            instances, _ = self.bbox_model.roi_heads(im, features, proposals)
        
            # print('Features: ', type(features), features.keys())
            # print('Proposals: ', type(proposals), proposals)
            plt.imshow(torch2numpy(im.detach().cpu().squeeze())); plt.show()
            # print('Instances: ', type(instances[0]), instances[0])
            # print('Im: ', im.shape)
            

        print('>>> Visualizer <<<')
        
        # img = images.squeeze().detach().cpu().numpy()
        # print('....................... ', img.shape)
        # images = np.transpose(img, (2, 1, 0))
        # print('----------------------- ', images.shape)
        # v = Visualizer(images[:, :, ::-1], scale=1.0)
        # ## v = Visualizer(im[:, :, ::-1], metadata=test_dataset_metadata, scale=0.5)
        # out = v.draw_instance_predictions(instances[0].to("cpu"))
        # out_img = out.get_image()[:, :, ::-1]
        # # splt.imsave('prediction_output/' + str(idx) + '_prediction.jpg', out_img)
        # plt.imshow(out_img)
        # # cv2.imshow(str(idx) + '_prediction', out_img)
        # plt.show()

        p6 = features['p6']
        # print('p6 = ', p6.shape)



        # return im, features, proposals
        
        ## Part 2: make line predictions
        # TODO: pass the informations in axis prediction model
        # currently: 2-FCN Layer as placeholder for the architecture
        # x = 'features and bbox-input as vector?!'
        # x = p6 #.to(device)
        x = torch.flatten(p6)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        # print('x = ', x.shape)
        return x

def detect_axis(bbox_door, bbox_handle, img):
    door_center =  bbox_door.mean(axis=0)
    handle_center =  bbox_handle.mean(axis=0)

    # print('min(door) = ', bbox_door[0, :], ' | max(door) = ', bbox_door[1, :])
    # print('center(door) = ', door_center, '\n')
    # print('min(handle) = ', bbox_handle[0, :], ' | max(handle) = ', bbox_handle[1, :])
    # print('center(handle) = ', handle_center, '\n\n')
    
    _, ax = plt.subplots()

    x_min_d, y_min_d = bbox_door[0, :]
    x_max_d, y_max_d = bbox_door[1, :]
    # rect_door = patches.Rectangle((x_min_d, y_min_d), x_max_d-x_min_d, y_max_d-y_min_d,
    #                                 linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect_door)
    
    x_min_h, y_min_h = bbox_handle[0, :]
    x_max_h, y_max_h = bbox_handle[1, :]
    # rect_handle = patches.Rectangle((x_min_h, y_min_h), x_max_h-x_min_h, y_max_h-y_min_h,
    #                                 linewidth=1, edgecolor='b', facecolor='none')
    # ax.add_patch(rect_handle)

    
    # door_circle = patches.Circle(door_center, radius=3, color='r')
    # handle_circle = patches.Circle(handle_center, radius=3, color='b')
    # ax.add_patch(door_circle)
    # ax.add_patch(handle_circle)


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

    # print(edge)
    # axis_line = mlines.Line2D(edge[:, 0], edge[:, 1], color='b', linewidth=3)
    # ax.add_line(axis_line)
    # ax.imshow(img)
    # plt.show()

    return edge

def axis_detection(original_image, bbox_gt, gt_lbl):
    pass



## TODO: define loss for axis-regression
def axis_loss(output, target):    
    val = torch.mean((output - target)**2)
    return val


def main():
    # cfg = 'config for detectron bbox-prediction model'
    cfg_file = os.path.join(CONFIG_PATH, 'cfg_bbox_mode.yaml')
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    # print(cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6


    feature_size = 12544 # Dimension of flattened optput of p6-part of ResNet50
    feature_size += 8    # p6 + 2 * BBox


    # standardization of images
    mu = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1).to(device)
    data_transform = lambda x, mean=mu, std=std: (x - mean) / std

    net = AxisPrediction(bbox_cfg=cfg, input_size=feature_size, normalizer=data_transform)
    net.freeze_bbox_model()
    net.to(device)
    criterion = giou_loss # axis_loss
    optimizer = optim.SGD(net.axis_model.parameters(), lr=0.001, momentum=0.9)

    # TODO: write custom dataloader to load the axis data
    # also COCO? -> Points of bbox can be used as points for
    # line prediction

    '''
        # data_path = '/home/ubuntupc/HESSENBOX-DA/DAP_data/train_data_clipped/'
        # data_path = '/home/ubuntupc/HESSENBOX-DA/DAP_data/dataset_with_axis_right/'
        # data_path = '/home/ubuntupc/HESSENBOX-DA/DAP_data/dataset_larger_bbs_low_compression/'
    '''
    data_path = '/home/ubuntupc/HESSENBOX-DA/DAP_data/train_data_big/'
    
    trainloader = DoorDataset(root=data_path, train=True)
    testloader = DoorDataset(root=data_path, train=False)
    

    # for i, data in enumerate(trainloader, 0):
        # rgb_img = data['rgb']
        # axis    = data['axis']
        # bbox_gt = data['bbox']
        # cls_lbl = data['label']
        # bbox_axis = data['bbox_axis']
        # axis_position = data['axis_position'] # 0: left, 1: right

        # axis = net(rgb_img, bbox_gt[0], bbox_gt[1])
        # print(axis)


    # return 

    ## TODO: Train the network
    n_epochs = 3
    
    losses = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # print('epoch = ', epoch, i)
            # needed data: rgb-img, axis
            rgb_img = data['rgb']
            axis = data['axis']
            bbox_gt = data['bbox']
            cls_lbl = data['label']
            bbox_axis = data['bbox_axis']
            axis_position = data['axis_position'] # 0: left, 1: right
            # axis = torch.flatten(labels).to(device)
        
            bbox_door = bbox_gt[cls_lbl[0]]
            bbox_handle = bbox_gt[cls_lbl[1]]
            detect_axis(bbox_door, bbox_handle, rgb_img)



            '''
                print(axis,'\n', bbox_axis)
                out_img = rgb_img # torch2numpy(rgb_img.squeeze())
                _, ax = plt.subplots(1)
                # y's = labels[:, 1]; x's = labels[:, 0]
                l = mlines.Line2D(axis[:, 1], axis[:, 0], color='b',
                                    linewidth=3)
                x_min, x_max = bbox_axis[:, 0]
                y_min, y_max = bbox_axis[:, 1]

                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                 linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.add_line(l)
                ax.imshow(out_img); plt.show()

                # np_img = rgb_img # torch2numpy(rgb_img.squeeze())
                # print(np_img.shape)
                # plt.imshow(np_img); plt.show()
                # plt.imshow(rgb_img); plt.show()
            '''

            bbox_axis = torch.flatten(bbox_axis).to(device)

            optimizer.zero_grad()
            # rgb_img = rgb_img.to(device)
            outputs = net(rgb_img, bbox_gt, cls_lbl)
            # print('axis_prediction = ', outputs)
            
            outputs = outputs.unsqueeze(0).cpu()
            bbox_axis = bbox_axis.unsqueeze(0).cpu()
            print('outputs: ', outputs)
            print('axis: ', axis)

            loss = criterion(outputs, bbox_axis)
            # print('L = ', loss.item(), '\n\n')
            losses.append(loss)


            loss.backward()
            optimizer.step()

            '''
            out_line = outputs.detach().cpu().reshape(2, 2)
            print('out_line = ', out_line, out_line.shape)
            out_img = rgb_img.copy()
            _, ax = plt.subplots(1)
            l = mlines.Line2D(labels[:, 1], labels[:, 0], color='b',
                                linewidth=3)
            l_pred = mlines.Line2D([out_line[0, 0], out_line[0, 1]],
                                   [out_line[1, 0], out_line[1, 1]], 
                                   color='r', linewidth=3)
            ax.add_line(l)
            ax.add_line(l_pred)
            ax.imshow(out_img); plt.show()
            '''

            epoch_loss += outputs.shape[0] * loss.item()
            running_loss += loss.item()
            if i % 10 == 1:
                print('\n')
                print('Loss = ', running_loss / 10)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                print('[]')
                running_loss = 0.0
                print('\n')


    # TODO: save the trained model
    # out_path = './path/to/save/model.pth'
    # torch.save(net.state_dict(), out_path)

if __name__ == "__main__":
    main()