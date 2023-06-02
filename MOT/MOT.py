import cv2
import time
import copy
from loguru import logger
from MOT.utils.draw import *
from MOT.detectors.detector import Detector
from MOT.trackers.tracker import Tracker
from MOT.utils.default_cfg import config

class MultiTracking:
    def __init__(self, tracker=0, use_cuda=False):
        """
        Initializes a MultiTracking object with a specified tracker and CUDA usage.

        Args:
            tracker (int): Index of the tracker type to be used. Use -1 to disable tracking.
            use_cuda (bool): Flag indicating whether to use CUDA for acceleration.
        """
        self.use_cuda = use_cuda

        # Get the detector object
        self.detector = self._get_detector()

        if tracker == -1:
            self.tracker = None
            return

        self.tracker = self._get_tracker(tracker)

    def _get_detector(self):
        """
        Returns the detector object.

        Returns:
            object: The detector object.
        """
        detector = Detector(use_cuda=self.use_cuda).get_detector()
        return detector
    
    def _get_tracker(self, tracker):
        """
        Returns the tracker object.

        Args:
            tracker (int): Index of the tracker type.

        Returns:
            object: The tracker object.
        """
        tracker = Tracker(tracker, self.detector, use_cuda=self.use_cuda).get_tracker()
        return tracker

    def _update_args(self, kwargs):
        """
        Updates the configuration arguments based on the provided keyword arguments.

        Args:
            kwargs (dict): Keyword arguments to update the configuration.

        Returns:
            dict: Updated configuration.
        """
        for key, value in kwargs.items():
            if key in config.keys():
                config[key] = value
            else:
                print(f'"{key}" argument not found! Valid args: {list(config.keys())}')
                exit()

        return config

    def track_video(self, video_path, **kwargs):
        """
        Tracks objects in a video and yields the bounding box and frame details.

        Args:
            video_path (str): Path to the video file.
            **kwargs: Additional keyword arguments for configuration.

        Yields:
            tuple: Bounding box details and frame details.
        """
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            yield bbox_details, frame_details

    def _start_tracking(self, stream_path, config):
        """
        Starts the object tracking process on the specified video stream.

        Args:
            stream_path (str): Path to the video stream.
            config (dict): Configuration dictionary.

        Yields:
            tuple: Bounding box details and frame details.
        """
        if not self.tracker:
            print("No tracker is selected. Use the detect() function to perform detection or pass a tracker.")
            exit()

        fps = config['fps']
        class_names = config['filter_classes']

        cap = cv2.VideoCapture(stream_path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        frame_id = 1
        tic = time.time()

        prev_time = 0

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            im0 = copy.deepcopy(frame)

            bboxes_xyxy, ids, scores, class_ids = self.tracker.detect_and_track(frame, config)
            elapsed_time = time.time() - start_time

            logger.info('frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count), elapsed_time * 1000))

            im0 = draw_boxes(im0, bboxes_xyxy, class_ids, identities=ids, class_names=class_names)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.line(im0, (20, 25), (127, 25), (85, 45, 255), 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            frame_id += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            yield (im0, ids, scores, class_ids), (frame, frame_id - 1, fps)

        tac = time.time()
        print(f'Total Time Taken: {tac - tic:.2f}')


if __name__ == '__main__':
    mot = MultiTracking()

    mot.start_tracking('../data/sample_vids/test.mp4')