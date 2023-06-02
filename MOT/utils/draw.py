import cv2
from numpy import random
import numpy as np
from MOT.utils.classes import get_names
from MOT.utils.colors import color_generator

names = get_names()
data_deque = {}

import cv2
import random


def draw_bounding_box(bbox_coords, image, label=None, color=None, line_thickness=None):
    """
    Draws a bounding box on the input image.

    Args:
        bbox_coords (list): List containing the coordinates of the bounding box in the format [x1, y1, x2, y2].
        image (np.ndarray): The input image.
        label (str): The label associated with the bounding box.
        color (list): List containing the RGB color values for the bounding box.
        line_thickness (int): The thickness of the bounding box lines.

    Returns:
        np.ndarray: The image with the drawn bounding box.
    """
    thickness = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    pt1, pt2 = (int(bbox_coords[0]), int(bbox_coords[1])), (int(bbox_coords[2]), int(bbox_coords[3]))

    # Draw the bounding box rectangle
    cv2.rectangle(image, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_AA)

    if label:
        text_thickness = max(thickness - 1, 1)
        text_size = cv2.getTextSize(str(label), 0, fontScale=thickness / 3, thickness=text_thickness)[0]

        # Draw a background rectangle for the label text
        image = draw_custom_box(image, (pt1[0], pt1[1] - text_size[1] - 3),
                                          (pt1[0] + text_size[0], pt1[1] + 3), color, 1, 8, 2)

        # Draw the label text
        cv2.putText(image, str(label), (pt1[0], pt1[1] - 2), 0, thickness / 3,
                    [225, 255, 255], thickness=text_thickness, lineType=cv2.LINE_AA)
    
    return image

def draw_custom_box(image, top_left, bottom_right, color, thickness, radius, gap):
    """
    Draws a custom box with rounded corners on the image.

    Args:
        image (numpy.ndarray): The image on which to draw the custom box.
        top_left (tuple): Top-left coordinates of the custom box.
        bottom_right (tuple): Bottom-right coordinates of the custom box.
        color (tuple): Color of the custom box in BGR format.
        thickness (int): Thickness of the custom box and lines.
        radius (int): Radius of the rounded corners.
        gap (int): Gap between the rounded corners and the lines.

    Returns:
        numpy.ndarray: The image with the custom box drawn.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw rounded corners
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    # Draw straight lines
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)

    # Draw rectangles to complete the custom box
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius - gap), color, -1, cv2.LINE_AA)

    # Draw circles at the corners
    cv2.circle(image, (x1 + radius, y1 + radius), 2, color, 12)
    cv2.circle(image, (x2 - radius, y1 + radius), 2, color, 12)
    cv2.circle(image, (x1 + radius, y2 - radius), 2, color, 12)
    cv2.circle(image, (x2 - radius, y2 - radius), 2, color, 12)

    return image

def draw_boxes(image, bounding_boxes, class_ids, identities=None, offset=(0, 0), class_names=None):
    """
    Draws bounding boxes on the input image.

    Args:
        image (numpy.ndarray): The input image.
        bounding_boxes (list): List of bounding box coordinates in the format [x1, y1, x2, y2].
        class_ids (list): List of class IDs for each bounding box.
        identities (list, optional): List of identities for each bounding box. Defaults to None.
        offset (tuple, optional): Offset to apply to the bounding box coordinates. Defaults to (0, 0).
        class_names (list, optional): List of class names. Defaults to None.

    Returns:
        numpy.ndarray: The image with bounding boxes drawn on it.
    """
    height, width, _ = image.shape

    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        identity = int(identities[i]) if identities is not None else None

        color = color_generator(int(class_ids[i]))

        if class_names:
            object_name = class_names[int(class_ids[i])]
        else:
            object_name = names[int(class_ids[i])]

        label = f'{object_name} {identity}' if identity is not None else object_name

        draw_bounding_box(box, image, label=label, color=color, line_thickness=2)

    return image
