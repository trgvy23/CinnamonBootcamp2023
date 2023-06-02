import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from MOT.trackers.deep_sort.tracker.deep.model import Net

class Extractor:
    def __init__(self, model_path, use_cuda=True):
        """
        Initialize the FeatureExtractor.

        Args:
            model_path (str): Path to the pre-trained model.
            use_cuda (bool): Whether to use CUDA for GPU acceleration.
        """
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        # Load the pre-trained model
        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict)

        # Set up logging
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))

        # Move the model to the appropriate device
        self.net.to(self.device)

        # Define the image size and normalization transform
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess_images(self, image_crops):
        """
        Preprocess the input image crops.

        Steps:
            1. Convert the images to float and scale them from 0 to 1.
            2. Resize the images to the target size of 64x128 pixels.
            3. Concatenate the images into a numpy array.
            4. Convert the numpy array to a torch Tensor.
            5. Apply normalization to the Tensor.

        Args:
            image_crops (List[ndarray]): List of image crops as numpy arrays.

        Returns:
            torch.Tensor: Preprocessed image batch.
        """
        def _resize_image(image, target_size):
            return cv2.resize(image.astype(np.float32) / 255., target_size)

        processed_images = torch.cat([self.norm(_resize_image(image, self.size)).unsqueeze(0)
                                      for image in image_crops], dim=0).float()
        return processed_images

    def __call__(self, image_crops):
        """
        Extract features from the input image crops.

        Args:
            image_crops (List[ndarray]): List of image crops as numpy arrays.

        Returns:
            ndarray: Extracted features as a numpy array.
        """
        preprocessed_images = self._preprocess_images(image_crops)

        with torch.no_grad():
            preprocessed_images = preprocessed_images.to(self.device)
            features = self.net(preprocessed_images)

        return features.cpu().numpy()