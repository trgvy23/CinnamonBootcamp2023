import os
from MOT import utils
from MOT.utils import get_names
import torch
from .utils.yolov8_utils import prepare_input, process_output
import numpy as np
import warnings
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import attempt_load_one_weight
from PIL import Image

class YOLOv8Detector:
    def __init__(self,
                 weights=None,
                 use_cuda=True):

        self.device = 'cuda' if use_cuda else 'cpu'

        # Load Model
        self.model = self.load_model(use_cuda, weights)

    def load_model(self, use_cuda, weights):

        model, ckpt = attempt_load_one_weight(weights)
        model = AutoBackend(model, fp16=False, dnn=False).to(self.device)
        model.float()
        return model

    def detect(self, image: list,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes: bool = None,
               agnostic_nms: bool = True,
               return_image=False
               ) -> list:

        # Preprocess input image and also copying original image for later use
        original_image = image.copy()
        processed_image = prepare_input(
            image, input_shape, 32, True)

        processed_image = torch.from_numpy(processed_image).to(self.device)
        # Change image floating point precision if fp16 set to true
        processed_image = processed_image.float()

        with torch.no_grad():
            prediction = self.model(processed_image, augment=False)
                
        detection = []
        # Postprocess prediction
        
        detection = process_output(prediction,
                                original_image.shape[:2],
                                processed_image.shape[2:],
                                conf_thres,
                                iou_thres,
                                agnostic=agnostic_nms,
                                max_det=max_det)

        image_info = {
            'width': original_image.shape[1],
            'height': original_image.shape[0],
        }

        if filter_classes:
            class_names = get_names()

            filter_class_idx = []
            if filter_classes:
                for _class in filter_classes:
                    if _class.lower() in class_names:
                        filter_class_idx.append(
                            class_names.index(_class.lower()))
                    else:
                        warnings.warn(
                            f"class {_class} not found in model classes list.")

            detection = detection[np.in1d(
                detection[:, 5].astype(int), filter_class_idx)]

        if return_image:
            return detection, original_image
        else: 
            return detection, image_info
        