from MOT.trackers.deep_sort.deepsort import DeepSort

class Tracker:
    def __init__(self, tracker_type: int, object_detector: object, use_cuda=True):
        """
        Initializes the object tracker.

        Args:
            tracker_type (int): The type of object tracker to use.
            object_detector: The object detection model used for detecting objects in images.
            use_cuda (bool): Whether to use CUDA for GPU acceleration.
        """
        self.available_trackers = {
            '0': DeepSort,
            # '1': StrongSort,
        }
        self.tracker = self._select_tracker(tracker_type, object_detector, use_cuda=use_cuda)

    def _select_tracker(self, tracker_type, object_detector, use_cuda):
        """
        Selects and initializes the specified object tracker.

        Args:
            tracker_type (int): The type of object tracker.
            object_detector: The object detection model used for detecting objects in images.
            use_cuda (bool): Whether to use CUDA for GPU acceleration.

        Returns:
            object: The initialized object tracker.
        """
        selected_tracker = self.available_trackers.get(str(tracker_type), None)

        if selected_tracker is not None:
            if selected_tracker is DeepSort:
                return selected_tracker(object_detector, use_cuda=use_cuda)
            else:
                return selected_tracker(object_detector)
        else:
            raise ValueError(f'Invalid tracker type: {tracker_type}')

    def detect_and_track(self, image, detection_config: dict):
        """
        Performs object detection and tracking on the input image.

        Args:
            image: The input image.
            detection_config (dict): Configuration parameters for object detection.

        Returns:
            tuple: A tuple containing the detected bounding boxes, object IDs, and class IDs.
        """
        return self.tracker.detect_and_track(image, detection_config)

    def get_tracker(self):
        """
        Returns the initialized object tracker.

        Returns:
            object: The initialized object tracker.
        """
        return self.tracker
