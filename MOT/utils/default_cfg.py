from MOT.utils.classes import get_names

names = get_names()

config = {
    "output_dir": "data/results",
    "filename": None,
    "fps": None,
    "filter_classes": names,
    "input_shape" : (640, 640),
    "conf_thres": 0.25,
    "iou_thres" : 0.45,
    "max_det" : 1000,
    "agnostic_nms" : False,
    "use_conda": False,
}
