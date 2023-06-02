from MOT.utils import get_names
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import attempt_load_one_weight
import numpy as np
import warnings
from ultralytics.yolo.utils import ops
from ultralytics.yolo.data.augment import LetterBox

class YOLOv8Detector:
    """
    YOLOv8Detector is a class that encapsulates the functionality of a YOLOv8 object detector.

    Args:
        weights (str): Path to the weights file for the YOLOv8 model.
        use_cuda (bool): Flag indicating whether to use CUDA for GPU acceleration. Default is True.

    Attributes:
        device (str): Device to use for model inference ('cuda' or 'cpu').
        model (nn.Module): YOLOv8 model.

    """

    def __init__(self, weights=None, use_cuda=True):
        self.device = 'cuda' if use_cuda else 'cpu'
        self.model = self.load_model(use_cuda, weights)

    def load_model(self, use_cuda, weights):
        """
        Loads the YOLOv8 model.

        Args:
            use_cuda (bool): Flag indicating whether to use CUDA for GPU acceleration.
            weights (str): Path to the weights file for the YOLOv8 model.

        Returns:
            nn.Module: Loaded YOLOv8 model.

        """

        model, ckpt = attempt_load_one_weight(weights)
        model = AutoBackend(model, fp16=False, dnn=False).to(self.device)
        model.float()
        return model

    def detect(self, image,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes=None,
               agnostic_nms: bool = True,
               return_image=False,
               display=False,
               output_dir=None,
               filename=None,
               save_result=True,
               fps=1
               ) -> list:

        """
        Detects objects in an input image using YOLOv8.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.
            input_shape (tuple): Input shape of the image. Default is (640, 640).
            conf_thres (float): Confidence threshold for object detection. Default is 0.25.
            iou_thres (float): IoU threshold for non-maximum suppression. Default is 0.45.
            max_det (int): Maximum number of detections. Default is 1000.
            filter_classes (list): List of classes to filter the detections. Default is None.
            agnostic_nms (bool): Flag indicating whether to use agnostic NMS. Default is True.
            return_image (bool): Flag indicating whether to return the image with detections. Default is False.
            display (bool): Flag indicating whether to display the image with detections. Default is False.
            output_dir (str): Directory to save the image with detections. Default is None.
            filename (str): Filename to save the image with detections. Default is None.
            save_result (bool): Flag indicating whether to save the image with detections. Default is True.
            fps (int): Frames per second for video output. Default is 1.

        Returns:
            tuple: A tuple containing the detection results and image information.
                - detection (list): List of detected objects.
                - image_info (dict): Dictionary containing the width and height of the original image.

        """

        # Make a copy of the original image
        original_image = image.copy()

        # Prepare the input image for inference
        processed_image = self._prepare_input(image, input_shape, 32, True)
        processed_image = torch.from_numpy(processed_image).to(self.device).float()

        # Perform inference on the processed image
        with torch.no_grad():
            prediction = self.model(processed_image, augment=False)

        # Process the output prediction to obtain the detection results
        detection = self._process_output(prediction, original_image.shape[:2], processed_image.shape[2:], 
                                         conf_thres, iou_thres, agnostic=agnostic_nms, max_det=max_det)

        # Extract image information
        image_info = {
            'width': original_image.shape[1],
            'height': original_image.shape[0],
        }

        # Filter the detections based on the specified classes
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

        # Return the detection results and image information
        if return_image:
            return detection, original_image
        else:
            return detection, image_info

    def _prepare_input(self, image, input_shape, stride, pt):
        """
        Prepare the input tensor by resizing and normalizing the image.

        Args:
            image (numpy.ndarray): The input image.
            input_shape (tuple): The desired input shape (height, width).
            stride (int): The stride value for resizing the image.
            pt (bool): Whether to use PyTorch style resizing.

        Returns:
            numpy.ndarray: The prepared input tensor.
        """
        letter_box = LetterBox(input_shape, auto=pt, stride=stride)
        input_tensor = letter_box(image=image)
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[::-1]  # Convert from HWC to CHW, BGR to RGB
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)  # Ensure contiguous memory layout
        input_tensor /= 255.0  # Normalize pixel values to the range 0.0 - 1.0
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # Add a batch dimension
        return input_tensor

    def _process_output(self, detections, ori_shape, input_shape, conf_threshold, iou_threshold, classes=None, agnostic=False, max_det=300):
        """
        Process the model's output detections.

        Args:
            detections (torch.Tensor): The output detections from the model.
            ori_shape (tuple): The original image shape (height, width).
            input_shape (tuple): The input shape used for inference (height, width).
            conf_threshold (float): The confidence threshold for filtering detections.
            iou_threshold (float): The IoU threshold for non-maximum suppression.
            classes (list, optional): The list of class labels. Defaults to None.
            agnostic (bool, optional): Whether to perform class-agnostic detections. Defaults to False.
            max_det (int, optional): The maximum number of detections to keep. Defaults to 300.

        Returns:
            numpy.ndarray: The processed output detections.
        """
        detections = ops.non_max_suppression(detections, conf_thres=conf_threshold, iou_thres=iou_threshold, 
                                             classes=classes, agnostic=agnostic, max_det=max_det)

        for i in range(len(detections)):
            # Scale and round the predicted bounding boxes
            detections[i][:, :4] = ops.scale_boxes(input_shape, detections[i][:, :4], ori_shape).round()

        return detections[0].cpu().numpy()