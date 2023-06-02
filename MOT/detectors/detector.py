import cv2
import os
from MOT.detectors.yolov8.yolov8_detector import YOLOv8Detector
from MOT.utils import get_names

class Detector:
    def __init__(self, use_cuda):
        self.model = self._select_detector(use_cuda)
    
    def _select_detector(self, cuda):
        _detector = YOLOv8Detector(weights=os.path.join('yolov8\weights\yolov8n.pt'), use_cuda=cuda)
        return _detector

    def get_detector(self):
        return self.model

    def detect(self, image, **kwargs: dict):
        return self.model.detect(image, **kwargs)


if __name__ == '__main__':
    # Create the detector instance
    detector = Detector(use_cuda=False)
    
    # Get the detector model
    model = detector.get_detector()

    # Load and detect objects in an image
    img = cv2.imread("../../data/sample_images/test.jpeg")
    detection, image_info = model.detect(img)

    # Get class names
    class_names = get_names()

    # Print the detection results with class names
    for detect in detection:
        class_id = int(detect[5])
        class_name = class_names[class_id]
        print("Class:", class_name)
        print("Coordinates:", detect[:4])
        print("Confidence:", detect[4])
        print()

    print("Image Info:", image_info)
