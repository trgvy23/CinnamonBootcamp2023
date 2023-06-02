import streamlit as st
import MOT
from MOT import MultiTracking
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import time
from MOT.utils.default_cfg import config

USE_CONDA = config['use_conda']

TRACKING_CLASSES = [
    # List of tracking classes
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def main():
    # Streamlit app configuration
    st.set_page_config(page_title="Tracking Dashboard", layout="wide")
    st.title("Tracking Dashboard")

    st.sidebar.header("Setting")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
    tracking_class = st.sidebar.multiselect("Class to select:", TRACKING_CLASSES)

    uploaded_video = st.sidebar.file_uploader("Upload video:", type=["mp4", "avi"])
    if uploaded_video is not None:
        temporary_file = tempfile.NamedTemporaryFile(delete=False)
        temporary_file.write(uploaded_video.read())
        video_path = temporary_file.name
    else:
        video_path = None

    if video_path is not None:
        tracker = MOT.DEEPSORT
    else:
        tracker = -1

    detecting_object = MultiTracking(
        tracker=tracker,
        use_cuda=USE_CONDA
    )

    # Start tracking the video with the specified configurations
    track_action = detecting_object.track_video(video_path, 
                                                conf_thres=confidence_threshold,
                                                iou_thres=0.5,
                                                filter_classes=tracking_class if tracking_class else None)
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = 'data/results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_placeholder = st.empty()
    text_container = st.container()

    with text_container:
        st.markdown(
            """
            <style>
                .info {
                    font-size: 20px;
                    font-weight: bold;
                    color: #4f4f65
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        num_frame_placeholder = st.empty()
        num_obj_placeholder = st.empty()
        fps_placeholder = st.empty()
    
    while True:
        start_time = time.time()
        ret, frame = video_capture.read()
        if not ret:
            break
        bbox_detail, frame_detail = next(track_action)
        im0, ids, scores, class_ids = bbox_detail
        frame_num = frame_detail[1]

        frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame_placeholder.image(frame, width=800, output_format="PNG")
        end_time = time.time()
        processing_time = end_time - start_time
        actual_fps = int(1/processing_time)
        num_frame_placeholder.markdown(
            f'<p class="info">Frame number: {frame_num}</p>',
            unsafe_allow_html=True
        )
        num_obj_placeholder.markdown(
            f'<p class="info">Number of Object being tracked: {len(ids)}</p>',
            unsafe_allow_html=True
        )
        fps_placeholder.markdown(
            f'<p class="info">FPS: {actual_fps}</p>',
            unsafe_allow_html=True
        )

        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame)

    video_capture.release()
    out.release()

if __name__ == '__main__':
    main()
