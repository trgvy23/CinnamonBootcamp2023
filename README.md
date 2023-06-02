<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>
CinnamonBootcamp2023
</h1>

<h3 align="center">🚀 Developed with the software and tools below:</h3>

<p align="center">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=for-the-badge&logo=SciPy&logoColor=white" alt="SciPy" />
<img src="https://img.shields.io/badge/JPEG-8A8A8A.svg?style=for-the-badge&logo=JPEG&logoColor=white" alt="JPEG" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white" alt="NumPy" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style=for-the-badge&logo=Markdown&logoColor=white" alt="Markdown" />
</p>

## 🤖 Overview
This is a Tracking System application that performs object tracking in videos using YOLOv8 for detection and DeepSort algorithm for tracking. It allows you to upload a video file, select the tracking classes, and adjust the confidence threshold for object detection.

![demo-gif](https://github.com/trgvy23/CinnamonBootcamp2023/blob/main/results.gif)

## 🔮 Prerequisites

- Python 3.6 or later
- Make sure to install MS Build tools 
- Make sure to install `GPU` driver if you want to use `GPU`

You can install the required dependencies by running the following command:

## 🚀 Installation

1. Clone the CinnamonBootcamp2023 repository:

```
git clone https://github.com/trgvy23/CinnamonBootcamp2023.git && cd CinnamonBootcamp2023
```

2. Create conda environment 

```
conda create -n cinnamon pip
conda activate cinnamon
```

3. Install the required dependencies

```
# for CPU
pip install torch torchvision

# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
or
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt
pip install typing-extensions --upgrade
```

## 💻 Usage

### 🪄 GPU Setting

If you want to run the application with Cuda, simply change the configuration in `MOT/utils/default_cfg.py`:

```python
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
```

Change `use_conda` key from `False` to `True`. The default value is `False`

### 👩‍💻 Running Multi Object Tracking

To start the Tracking System, run the following command:

```
streamlit run main.py
```

Once the application is running, you can access it in your local web browser.

1. Select the desired settings in the sidebar:
   - Adjust the confidence threshold using the slider.
   - Choose the classes you want to track by selecting from the multi-select dropdown.
   - Upload a video file (supported formats: MP4, AVI).

2. The object tracking process will begin automatically.

3. The dashboard will display the video frames with tracked objects being highlighted.

4. The number of the current frame, the number of objects being tracked, and the frames per second (FPS) will be displayed in the text area below the video.

5. Once the tracking is complete, a results.mp4 file will be saved in the `data/results` directory.

## ⚙️ Project Structure

```
repo
├── MOT
│   ├── detectors
│       ├── yolov8
│           ├── weights
│               └── yolov8n.pt
│           └── yolov8_detector.py
│       └── detector.py
│   ├── trackers
│       ├── deep_sort
│           ├── tracker
│               ├── deep
│                   ├── checkpoint
│                       ├── ckpt.t7
│                   ├── feature_extractor.py
│                   └── model.py
│               ├── sort
│                   ├── detection.py
│                   ├── iou_matching.py
│                   ├── kalman_filter.py
│                   ├── linear_assignment.py
│                   ├── nn_matching.py
│                   ├── track.py
│                   └── tracker.py
│               ├── __init__.py
│               └── deep_sort.py
│           └── deepsort.py
│       └── tracker.py
│   ├── utils
│       ├── __init.py__
│       ├── classes.py
│       ├── colors.py
│       ├── default_cfg.py
│       ├── draw.py
│       └── points_conversion.py
│   ├── MOT.py
│   └── __init__.py
├── data
│   ├── results
│      └── results.mp4
│   ├── sample_images
│   └── sample_vids
├── README.md
├── main.py
├── requirements.txt
└──results.gif
```

## 🙏 Acknowledgments

I would like to express my gratitude to the following open-source projects:

- [Ultralytics YOLOv8](https://github.com/airockchip/ultralytics_yolov8): This repository provides the YOLOv8 implementation, which is used for object detection in the Tracking Dashboard. 

- [YOLOv5 DeepSort OSNet](https://github.com/mikel-brostrom/Yolov5_DeepSort_OSNet): This repository by Mikel Broström serves as the foundation for the object detection and tracking pipeline used in the Tracking Dashboard. The combination of YOLOv5, DeepSort, and OSNet results in real-time and accurate multi-object tracking. 

Please note that while we have utilized these repositories as references, the Tracking Dashboard is a standalone application with its own specific implementation and features.
