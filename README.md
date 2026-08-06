# Real-Time Person Tracking & Face Recognition Line Counter
<img width="438" height="240" alt="demonstration" src="https://github.com/user-attachments/assets/f46ec8c2-85e3-4d9c-a76a-53172412d6b7" />


A computer vision pipeline combining YOLOv8 object detection, ByteTRACK multi-object tracking, and DeepFace (ArcFace) facial recognition to track, count, and identify individuals crossing a virtual boundary in real time.

![Project Demo](demo.gif)

## Key Features
- **Object Tracking:** Uses ByteTRACK to retain persistent object IDs across frames.
- **Directional Counter:** Vectorized X-coordinate line crossing algorithm prevents duplicate counts.
- **Biometric Identification:** Embeds ArcFace facial recognition with 512-D vectors for identity verification.
- **FPS Optimization:** Executes face matching every 30 frames and caches identities to run smoothly on CPU.

## Tech Stack
- **Detection & Tracking:** Ultralytics YOLOv8, ByteTRACK
- **Facial Recognition:** DeepFace (ArcFace Model, OpenCV Backend)
- **Image Processing:** OpenCV, NumPy

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/ShadiAbbas/yolov8-deepface-tracking-counter.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the application:
   ```bash
   python person_face_seg.py
