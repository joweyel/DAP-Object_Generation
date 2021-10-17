import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import get_mean_std as img_stats

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.8")

import random
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# loading of COCO-Datasets is especially easy
from detectron2.data.datasets import register_coco_instances

# For Trainging
from detectron2.engine import DefaultTrainer

# For saving
from detectron2.checkpoint import DetectionCheckpointer, Checkpointer


# Paths to data and configs
# DOOR_DATA = os.environ['DOORDATA']
detectron2_models = os.environ['detectron2_models']
CONFIG_PATH = './cfgs/'


def png4to3(img):
    if img.shape[-1] == 4:  # if 4-channel image (png)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = (img - img.min()) / (img.max() - img.min())
        img = (255. * img).astype(np.uint8)
    return img

def visualize_data(dataset, metadata, n_samples=10):
    for d in random.sample(dataset, n_samples):
        img = plt.imread(d['file_name'])
        img = png4to3(img)
        ## visualizer = Visualizer(img, metadata=metadata, scale=0.5) 
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0) 
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()

def bbox_prediction(args, resume=False):

    model_path = args.model
    data_path = args.data
    mode = args.mode

    ## Load dataset
    train_json = os.path.join(data_path, 'dataset/train/coco_train.json')
    test_json  = os.path.join(data_path, 'dataset/unseen_test/coco_test.json')
    image_root_train = os.path.join(data_path, 'dataset/train/images/')
    image_root_test = os.path.join(data_path, 'dataset/unseen_test/images/')
    image_stat = os.path.join(data_path, 'dataset/data_stats.json')

    # generate dataloader for coco-feature data
    register_coco_instances(name='door_data_train', 
                            metadata={}, json_file=train_json, 
                            image_root=image_root_train)

    register_coco_instances(name='door_data_test', 
                            metadata={}, 
                            json_file=test_json, 
                            image_root=image_root_test)


    # get the registered dataset to use by name
    train_dataset = DatasetCatalog.get('door_data_train')
    train_dataset_metadata = MetadataCatalog.get('door_data_train')
    test_dataset = DatasetCatalog.get('door_data_test')
    test_dataset_metadata = MetadataCatalog.get('door_data_test')

    # R101-FPN -> faster_rcnn_R_101_FPN_3x.yaml
    ## initializing the model-config and loading a model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    # cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))    
    cfg.DATASETS.TRAIN = ('door_data_train',)
    cfg.DATASETS.TEST = ('door_data_test',)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # retain Images with no features

    ## load or resume to use an old checkpoint
    if resume and model_path is not None:
        print('Loading pretrained weights!')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

        if not os.path.exists(cfg.MODEL.WEIGHTS):  # if nothing is found
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
            # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    else:
        print('Start from scratch!')
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    print(cfg.MODEL.WEIGHTS)


    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000 # 1400 # 1500 # 2000
    cfg.SOLVER.STEPS = []
    # cfg.TEST.EVAL_PERIOD = 20



    ## Load a model (faster_rcnn_R_50_FPN_3x)
    # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
    # After changing the number of classes, certain layers in a pre-trained
    # model will become incompatible and therefore cannot be loaded to the 
    # new model. This is expected, and loading such pre-trained models will 
    # produce warnings about such layers.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    
    # get the data statistics to normalize data
    if os.path.exists(image_stat):
        with open(image_stat, 'r') as data:
            img_data = json.load(data)
            mean = img_data['mean']
            std = img_data['std']
    else:
        # mean, std = img_stats.get_data_stats(image_root)
        mean, std = img_stats.get_data_stats(image_root_train)
        mean = mean.tolist()
        std = std.tolist()

    print('Mean = ', mean, ' | std = ', std)
    cfg.MODEL.PIXEL_MEAN = mean
    cfg.MODEL.PIXEL_STD = std

    ####################################### TRAIN #############################################
    ## Train a model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    if mode == 'train':
        trainer.train()
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model (we just trained)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9                          # set a custom testing threshold

    print('cfg.OUTPUT_DIR: ', cfg.OUTPUT_DIR)
    print(cfg)

    ####################################### TRAIN #############################################


    ## load a bbox-prediction model with the constructed config
    predictor = DefaultPredictor(cfg)

    with open(CONFIG_PATH + 'cfg_bbox_mode.yaml', 'w') as f:
        f.write(cfg.dump())   # save config to file

    # for idx, d in enumerate(random.sample(test_dataset, 10)):
    for idx, d in tqdm(enumerate(test_dataset), desc='Prediction on Test-Data: '):
        im = plt.imread(d['file_name'])
        im = png4to3(im)
        outputs = predictor(im)
        
        # print(outputs)
        v = Visualizer(im[:, :, ::-1], metadata=train_dataset_metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_img = out.get_image()[:, :, ::-1]
        plt.imsave('prediction_output/' + str(idx) + '_prediction.jpg', out_img)
        # plt.imshow(out_img)
        plt.show()

    ## evaluate the trained model with COCO-Metrics for Bounding-Boxes
    evaluator = COCOEvaluator('door_data_test', tasks=('bbox',), 
                          distributed=False, output_dir='./output/')
    val_loader = build_detection_test_loader(cfg, 'door_data_test')
    print(inference_on_dataset(trainer.model, val_loader, evaluator))




def _bbox_prediction(args, resume=False):

    model_path = args.model
    data_path = args.data
    mode = args.mode

    ## Load dataset
    train_json = os.path.join(data_path, 'dataset/train/coco_train.json')
    test_json  = os.path.join(data_path, 'dataset/unseen_test/coco_test.json')
    image_root_train = os.path.join(data_path, 'dataset/train/images/')
    image_root_test = os.path.join(data_path, 'dataset/unseen_test/images/')
    image_stat = os.path.join(data_path, 'dataset/data_stats.json')


    '''
        ## Load dataset
        train_json = DOOR_DATA + 'dataset_unseen_large_/dataset/train/coco_train.json'
        test_json  = DOOR_DATA + 'dataset_unseen_large_/dataset/unseen_test/coco_test.json'
        image_root_train = DOOR_DATA + 'dataset_unseen_large_/dataset/train/images/'
        image_root_test = DOOR_DATA + 'dataset_unseen_large_/dataset/unseen_test/images/'
        image_stat = DOOR_DATA + 'dataset_unseen_large_/dataset/data_stats.json'
        print(os.path.exists(train_json))
        print(os.path.exists(test_json))
        print(os.path.exists(image_root_train))
        print(os.path.exists(image_root_test))
    '''

    # generate dataloader for coco-feature data
    register_coco_instances(name='door_data_train', 
                            metadata={}, json_file=train_json, 
                            image_root=image_root_train)

    register_coco_instances(name='door_data_test', 
                            metadata={}, 
                            json_file=test_json, 
                            image_root=image_root_test)


    # get the registered dataset to use by name
    train_dataset = DatasetCatalog.get('door_data_train')
    train_dataset_metadata = MetadataCatalog.get('door_data_train')
    test_dataset = DatasetCatalog.get('door_data_test')
    test_dataset_metadata = MetadataCatalog.get('door_data_test')

    visualize_data(train_dataset, train_dataset_metadata, 10)


    # R101-FPN -> faster_rcnn_R_101_FPN_3x.yaml
    ## initializing the model-config and loading a model
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))    
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))    
    cfg.DATASETS.TRAIN = ('door_data_train',)
    cfg.DATASETS.TEST = ('door_data_test',)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # retain Images with no features


    ## load or resume to use an old checkpoint
    if resume:
        print('Loading pretrained weights!')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

        if not os.path.exists(cfg.MODEL.WEIGHTS):  # if nothing is found
            # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    else:
        print('Start from scratch!')
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    print(cfg.MODEL.WEIGHTS)

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000 # 1400 # 1500 # 2000
    cfg.SOLVER.STEPS = []
    # cfg.TEST.EVAL_PERIOD = 20



    ## Load a model (faster_rcnn_R_50_FPN_3x)
    # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
    # After changing the number of classes, certain layers in a pre-trained
    # model will become incompatible and therefore cannot be loaded to the 
    # new model. This is expected, and loading such pre-trained models will 
    # produce warnings about such layers.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2


    # get the data statistics to normalize data
    if os.path.exists(image_stat):
        with open(image_stat, 'r') as data:
            img_data = json.load(data)
            mean = img_data['mean']
            std = img_data['std']
    else:
        # mean, std = img_stats.get_data_stats(image_root)
        mean, std = img_stats.get_data_stats(image_root_train)
        mean = mean.tolist()
        std = std.tolist()

    print('Mean = ', mean, ' | std = ', std)
    cfg.MODEL.PIXEL_MEAN = mean
    cfg.MODEL.PIXEL_STD = std
    print(cfg)

    ## Train a model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    print('cfg.OUTPUT_DIR: ', cfg.OUTPUT_DIR)


    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9                          # set a custom testing threshold
    print(cfg)

    ## load a bbox-prediction model with the constructed config
    predictor = DefaultPredictor(cfg)

    with open(CONFIG_PATH + 'cfg_bbox_mode.yaml', 'w') as f:
        f.write(cfg.dump())   # save config to file

    # for idx, d in enumerate(random.sample(test_dataset, 10)):
    for idx, d in tqdm(enumerate(test_dataset), desc='Prediction on Test-Data: '):
        im = plt.imread(d['file_name'])
        im = png4to3(im)
        outputs = predictor(im)
        
        # print(outputs)
        v = Visualizer(im[:, :, ::-1], metadata=train_dataset_metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_img = out.get_image()[:, :, ::-1]
        plt.imsave('prediction_output/' + str(idx) + '_prediction.jpg', out_img)
        # plt.imshow(out_img)
        plt.show()

    ## evaluate the trained model with COCO-Metrics for Bounding-Boxes
    evaluator = COCOEvaluator('door_data_test', tasks=('bbox',), 
                          distributed=False, output_dir='./output/')
    val_loader = build_detection_test_loader(cfg, 'door_data_test')
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='which model to use', type=str, default=None)
    parser.add_argument('--mode', help='train or evaluate', type=str, required=True)
    parser.add_argument('--data', help='path to dataset', type=str, required=True)
    args = parser.parse_args()
    print(parser.parse_args())
    bbox_prediction(args, resume=False)