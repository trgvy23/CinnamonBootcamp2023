import cv2
import os

from MOT.detectors.yolov8.yolov8_detector import YOLOv8Detector


class Detector:
    def __init__(self, use_cuda):
        
        self.model = self._select_detector(use_cuda)
    def _select_detector(self, cuda):
        _detector = YOLOv8Detector(weights=os.path.join('yolov8\weights\yolov8n.pt'), 
                                   use_cuda=cuda)
        return _detector

    def get_detector(self):
        return self.model

    def detect(self,
               image: list,
               **kwargs: dict):
        return self.model.detect(image, **kwargs)


if __name__ == '__main__':

    result = Detector(use_cuda=False)
    img = cv2.imread("data/sample_images/test.jpeg")
    pred = result.detect(img)
    print(pred)
