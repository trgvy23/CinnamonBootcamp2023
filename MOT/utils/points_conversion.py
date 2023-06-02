def xyxy_to_xywh(xyxy):
    """
    Converts bounding box coordinates from absolute pixel values to relative values.

    Args:
        xyxy (list): List of bounding box coordinates in the format [x1, y1, x2, y2].

    Returns:
        list: List of bounding box coordinates in the format [x_c, y_c, w, h].
    """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return [x_c, y_c, w, h]