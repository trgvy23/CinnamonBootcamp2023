# from MOT.trackers import ByteTrack
# from MOT.trackers import NorFair
from MOT.trackers.deep_sort.deepsort import DeepSort
# from MOT.trackers import Motpy

class Tracker:
    def __init__(self, tracker: int, detector: object, use_cuda=True) -> None:
        
        self.trackers = {
            '0': DeepSort,
            # '1': ByteTrack,
            # '2': NorFair,
            # '3': Motpy
        }

        self.tracker = self._select_tracker(tracker, detector, use_cuda=use_cuda)

    def _select_tracker(self, tracker, detector, use_cuda):
        _tracker = self.trackers.get(str(tracker), None)

        if _tracker is not None:
            if _tracker is DeepSort:
                return _tracker(detector, use_cuda=use_cuda)
            else:
                return _tracker(detector)
        else:
            raise ValueError(f'Invalid tracker: {tracker}')

    def detect_and_track(self, image, config: dict):
        
        return self.tracker.detect_and_track(image, config)

    def get_tracker(self):
        return self.tracker