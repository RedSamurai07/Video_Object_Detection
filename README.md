# Automated Video Object Detection and Person Tracking

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
This project implements a computer vision pipeline designed to download video content from external sources (such as YouTube) and perform real-time object detection using the ImageAI library. The system is specifically configured to identify and track "persons" across every frame of a video, providing detailed telemetry on their location (bounding boxes) and the model's confidence levels. It utilizes a pre-trained deep learning environment (Torch and Torchvision) to process high-resolution video files and output a new video with visual detection overlays.

### Executive Summary
The project successfully demonstrates a complete end-to-end workflow for video analytics. By integrating yt-dlp for media acquisition and ImageAI for detection, the system processed a video containing over 11,442 frames. During execution, the model consistently identified multiple individuals per frame, often with 99-100% confidence. The system is optimized for stability, including a detection timeout mechanism and automated dependency management to ensure it can run in various Python environments (like Anaconda or Jupyter).

### Goal

- Automate Media Acquisition: Use command-line tools to programmatically download high-quality MP4 video files for analysis.

- Precision Person Detection: Configure the detection model to specifically target "person" objects, ignoring irrelevant classes to improve processing efficiency.

- Frame-by-Frame Analytics: Implement a callback function (per_frame_function) to extract and print real-time data, including the number of persons detected and their exact coordinates (Box Points) for every frame.

- Visual Output Generation: Produce a processed version of the input video that includes bounding boxes and labels for all detected individuals.

- Model Reliability: Maintain a high detection standard by enforcing a 40% minimum probability threshold, ensuring that only high-confidence detections are recorded.

### Data Structure 

[Video link](https://rr1---sn-5jucgv5qc5oq-cagy.googlevideo.com/videoplayback?expire=1772045808&ei=kPGeaanpHc3PkucPx6232A8&ip=82.26.208.222&id=o-APwWhXE0OUG6ZSjgDJpIDAq76DpH2i0qrxwcvodwbXzr&itag=18&source=youtube&requiressl=yes&xpc=EgVo2aDSNQ%3D%3D&rms=au%2Cau&bui=AVNa5-yResFzwEHYx15DrPDmAJyvBip5oO2MGkIhQ5-bvuAo9-Ut6yhAZZm-vyURKcTbCLad270M5CZl&spc=6dlaFIfWCePQLd5B0nsoofsT7HJXl9J5QUcTcmsqdJrmvINMOvZW9plcuyBRyBFbeGc&vprv=1&svpuc=1&mime=video%2Fmp4&rqh=1&cnr=14&ratebypass=yes&dur=84.195&lmt=1762410982655185&fexp=51552689,51565115,51565681,51580968&c=ANDROID&txp=5538534&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cxpc%2Cbui%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Crqh%2Ccnr%2Cratebypass%2Cdur%2Clmt&sig=AJEij0EwRAIgYAtuil5iwnM36msdaz0obCTw9Yd6C7NdKFzOdjscQDgCIHRSP9xCJhkl092Vt1O_OhdfQdIvOGAUISocB18N5f_c&redirect_counter=1&rm=sn-q4fe6y7e&rrc=104&req_id=d02b9bb370a1a3ee&cms_redirect=yes&cmsv=e&cps=0&ipbypass=yes&met=1772024215,&mh=l8&mip=49.207.137.152&mm=31&mn=sn-5jucgv5qc5oq-cagy&ms=au&mt=1772023753&mv=m&mvi=1&pl=21&lsparams=cps,ipbypass,met,mh,mip,mm,mn,ms,mv,mvi,pl,rms&lsig=APaTxxMwRgIhAMO1Bm4CTgmGtLW4gNCD6TmuJy10SevLFF0uIIh3XVYDAiEAxPHjxmIKvEVCfdjk5RmcJ9lfALCj3eSL8Mgr8JdyeAA%3D)

### Tools

Python: VS code, 

### Analysis
**Process of Object detection in video**

First,  we downlaod the video with good resolution then we download and upload the pre trained model `yolov3.pt` under the models folder. We run the code as mentioned below.

We install the necessary packages for object detetcion in video.

``` python
pip install imageai yt-dlp opencv-python torch torchvision
```
!yt-dlp -f "best[ext=mp4]" -o "input_video.mp4" "https://youtu.be/9zW_CKWb3oM?si=hn1AJ9ikzB1Ltou2"

We access the link where we weanted to perfrom object detetcion

``` python
!yt-dlp -f "best[ext=mp4]" -o "input_video.mp4" "https://youtu.be/9zW_CKWb3oM?si=hn1AJ9ikzB1Ltou2"
```

<img width="1633" height="142" alt="image" src="https://github.com/user-attachments/assets/412162ee-1a82-4e95-bf8a-2cc1f1617623" />

We perfrom the object detection using the below mentioned code snippet.

``` python
import os
import cv2
from imageai.Detection import VideoObjectDetection

# Configuratinon
MODEL_PATH  = r"D:\Github repos\Projects\Video_Object_Detection\models\yolov3.pt"
INPUT_VIDEO = "input_video.mp4"
OUTPUT_PATH = "output/detected_video"

cap = cv2.VideoCapture(INPUT_VIDEO)
VIDEO_FPS    = int(cap.get(cv2.CAP_PROP_FPS))          # ✅ Reads real FPS from your video
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
DURATION_SEC = TOTAL_FRAMES // VIDEO_FPS
cap.release()

print(f"Video Info: {VIDEO_FPS} FPS | {TOTAL_FRAMES} frames | ~{DURATION_SEC} seconds")

os.makedirs("output", exist_ok=True)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(MODEL_PATH)
detector.loadModel()

# Custom Objects (Person detection only) 
custom_objects = detector.CustomObjects(person=True)

# --- 4. CALLBACK FUNCTION ---
def per_frame_function(frame_number, output_array, output_count):
    print(f"Frame {frame_number}/{TOTAL_FRAMES} | Persons: {output_count}")
    for obj in output_array:
        print(f"  -> {obj['name']} | Confidence: {obj['percentage_probability']:.1f}% | Box: {obj['box_points']}")

# Execution
print("\nStarting full video detection...")
print(f"Processing ALL {TOTAL_FRAMES} frames at {VIDEO_FPS} FPS...\n")

video_path = detector.detectObjectsFromVideo(
    input_file_path=INPUT_VIDEO,
    output_file_path=OUTPUT_PATH,
    custom_objects=custom_objects,
    frames_per_second=VIDEO_FPS,        
    per_frame_function=per_frame_function,
    minimum_percentage_probability=40,
    log_progress=True,
    detection_timeout= 360         
)

print(f"\nDetection complete! Full video saved at: {video_path}")import os
import cv2
from imageai.Detection import VideoObjectDetection

# Configuratinon
MODEL_PATH  = r"D:\Github repos\Projects\Video_Object_Detection\models\yolov3.pt"
INPUT_VIDEO = "input_video.mp4"
OUTPUT_PATH = "output/detected_video"

cap = cv2.VideoCapture(INPUT_VIDEO)
VIDEO_FPS    = int(cap.get(cv2.CAP_PROP_FPS))          # ✅ Reads real FPS from your video
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
DURATION_SEC = TOTAL_FRAMES // VIDEO_FPS
cap.release()

print(f"Video Info: {VIDEO_FPS} FPS | {TOTAL_FRAMES} frames | ~{DURATION_SEC} seconds")

os.makedirs("output", exist_ok=True)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(MODEL_PATH)
detector.loadModel()

# Custom Objects (Person detection only) 
custom_objects = detector.CustomObjects(person=True)

# --- 4. CALLBACK FUNCTION ---
def per_frame_function(frame_number, output_array, output_count):
    print(f"Frame {frame_number}/{TOTAL_FRAMES} | Persons: {output_count}")
    for obj in output_array:
        print(f"  -> {obj['name']} | Confidence: {obj['percentage_probability']:.1f}% | Box: {obj['box_points']}")

# Execution
print("\nStarting full video detection...")
print(f"Processing ALL {TOTAL_FRAMES} frames at {VIDEO_FPS} FPS...\n")

video_path = detector.detectObjectsFromVideo(
    input_file_path=INPUT_VIDEO,
    output_file_path=OUTPUT_PATH,
    custom_objects=custom_objects,
    frames_per_second=VIDEO_FPS,        
    per_frame_function=per_frame_function,
    minimum_percentage_probability=40,
    log_progress=True,
    detection_timeout= 360         
)

print(f"\nDetection complete! Full video saved at: {video_path}")
```
<img width="509" height="298" alt="image" src="https://github.com/user-attachments/assets/201ca7ce-8714-4173-9710-a278193620d6" />
<img width="501" height="478" alt="image" src="https://github.com/user-attachments/assets/15de1181-7c48-4ce0-91b1-f57734dc3e4a" />



**Video output:**




### Insights

- High Confidence in Dense Scenarios: The model (YOLOv3) demonstrates exceptional precision even in crowded frames. For example, in Frame 705, the system successfully identified 14 distinct persons, with many detections reaching a 100% confidence score.

- Temporal Detection Gaps: The logs show significant sequences where no objects are detected (e.g., Frames 1 through 69). This suggests the video may have a long intro or "clean" background, indicating that a pre-analysis phase could skip these empty segments to save resources.

- Processing Bottle-neck: With 11,442 frames to process at 29 FPS, the system is handling a massive computational load. Processing every single frame at this high frequency is likely overkill for standard tracking and contributes to long execution times.

- Effective Object Filtering: By using CustomObjects(person=True), the system effectively reduces noise by ignoring other common YOLO classes (cars, dogs, etc.), focusing all computational power on person-tracking telemetry.

### Recommendations

1. Upgrade the Detection Architecture
The current project uses YOLOv3, which is now several generations behind.

- Transition to YOLOv8 or YOLOv10. These newer models offer a superior speed-to-accuracy ratio and are optimized for real-time video inference, potentially reducing processing time by 50-70%.

2. Implement Frame Skipping
Processing every frame at 29 FPS is computationally expensive and often redundant for person tracking.

- Modify the logic to process every 3rd or 5th frame. In a 29 FPS video, processing 10 frames per second is usually sufficient to maintain accurate tracklets while cutting processing time by roughly 65%.

3. Formalize Data Output
Currently, detection data (coordinates and confidence) is only printed to the console via the per_frame_function.

- Update the callback function to write this data into a structured CSV or JSON file. This allows for post-processing tasks such as:

- Heatmapping: Visualizing where people spend the most time.

- Path Tracking: Drawing the movement trajectories of individuals over time.

4. Enable Hardware Acceleration Check
Video processing is highly dependent on the GPU.

- Add a code block to verify if CUDA (NVIDIA GPU support) is active. If the system is defaulting to the CPU, it will take hours to finish 11,000 frames. Ensuring torch.cuda.is_available() returns True is critical for a production-level pipeline.

- Install PhantomJS or a similar runtime in the environment to ensure stable video fetching from various external sources.
