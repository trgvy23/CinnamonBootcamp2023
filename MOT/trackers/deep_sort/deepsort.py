import numpy as np
import os
from MOT import utils
from .tracker import build_tracker

class DeepSort:
    def __init__(self, detector, weights=None, use_cuda=True):
        """
        Initializes the DeepSort object tracker.

        Args:
            detector: Object detector used for detecting objects in images.
            weights (str): Path to the DeepSort model weights file.
            use_cuda (bool): Whether to use CUDA for GPU acceleration.
        """
        if weights is None:
            # Use default path for weights file if not provided
            weights = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tracker/deep/checkpoint/ckpt.t7"
            )

        # DeepSort configuration parameters
        cfg = {
            'MAX_DIST': 0.2,
            'MIN_CONFIDENCE': 0.3,
            'NMS_MAX_OVERLAP': 0.5,
            'MAX_IOU_DISTANCE': 0.7,
            'MAX_AGE': 70,
            'N_INIT': 3,
            'NN_BUDGET': 100
        }

        # Build the DeepSort tracker
        self.tracker = build_tracker(weights, cfg, use_cuda=use_cuda)
        self.detector = detector

        # Get the input shape for the detector model
        try:
            self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        except AttributeError as e:
            self.input_shape = (640, 640)

    def detect_and_track(self, image: np.ndarray, config: dict) -> tuple:
        """
        Performs object detection and tracking on the input image.

        Args:
            image (np.ndarray): Input image.
            config (dict): Configuration parameters for object detection.

        Returns:
            tuple: A tuple containing the bounding boxes, object IDs, and class IDs of the detected and tracked objects.
        """

        # Perform object detection using the detector
        detection_results, image_info = self.detector.detect(image, **config)
        image_info['im0'] = image

        class_ids = []
        ids = []
        bounding_boxes = []
        scores = []

        if isinstance(detection_results, np.ndarray) and len(detection_results) > 0:
            # Extract class IDs from detections
            class_ids = detection_results[:, -1].tolist()

            # Update the tracker with the detections
            bounding_boxes, ids, class_ids = self._tracker_update(detection_results, image_info)

        return bounding_boxes, ids, [], class_ids

    def _tracker_update(self, detection_results: np.ndarray, image_info: dict):
        """
        Updates the DeepSort tracker with the detected bounding boxes.

        Args:
            detection_results (np.ndarray): Detected bounding boxes in the format [x1, y1, x2, y2, confidence, class_id].
            image_info (dict): Information about the input image.

        Returns:
            tuple: A tuple containing the updated bounding boxes, object IDs, and class IDs.
        """

        updated_bounding_boxes = []
        ids = []
        object_id = []

        if detection_results is not None:
            # Convert bounding box coordinates to xywh format
            dets_xywh = np.array([np.array(utils.xyxy_to_xywh(det)) for det in detection_results[:, :4]])

            # Update the tracker with the converted bounding boxes
            outputs = self.tracker.update(
                dets_xywh, detection_results[:, -2].tolist(), detection_results[:, -1].tolist(), image_info['im0'])

            if len(outputs) > 0:
                updated_bounding_boxes = outputs[:, :4]
                ids = outputs[:, -2]
                object_id = outputs[:, -1]

        return updated_bounding_boxes, ids, object_id
