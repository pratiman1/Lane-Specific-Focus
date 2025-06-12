
This project implements real-time lane detection using YOLOv8 segmentation and Intel RealSense depth camera to calculate 3D world coordinates of lane center points. The system processes video streams, detects lanes, and outputs spatial coordinates with timestamps to a CSV file.

Key Features
🚦 Real-time lane detection using YOLOv8 segmentation

📏 Reference line generation at configurable intervals

📍 Central point calculation within detected lanes

📊 3D world coordinate calculation using depth data

⏱️ Timestamped CSV output with IST timezone

📈 FPS monitoring and performance display

🎨 Visual annotation with configurable transparency

Requirements
Hardware
Intel RealSense depth camera (D400 series recommended)

NVIDIA GPU (for optimal YOLOv8 performance)

Software
Python 3.8+

OpenCV

PyTorch

Ultralytics YOLOv8

PyRealSense2

NumPy

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/lane-detection-3d.git
cd lane-detection-3d
Install dependencies:

bash
pip install -r requirements.txt
Download the pre-trained YOLOv8 lane detection model:

bash
wget https://github.com/yourusername/models/releases/download/v1.0/best_lab.pt -O models/best_lab.pt
Configuration
Modify the configurable variables at the top of lane_detection.py:

python
# --- Configurable Variables ---
MODEL_PATH = 'models/best_lab.pt'  # Path to YOLO model
CSV_FILE_PATH = 'output/lane_data.csv'  # Output CSV path
DESIRED_CLASS_ID = 0  # Class ID for 'Lane'
REFERENCE_LINE_INTERVAL = 40  # Pixel interval for reference lines
TOLERANCE = 5  # Boundary detection tolerance
TRANSPARENCY = 0.5  # Annotation transparency
Usage
Run the detection script:

bash
python lane_detection.py
Controls:

Press 'q' to quit

Adjust camera position for optimal depth sensing

Ensure proper lighting for lane detection

Outputs
The system generates:

Real-time visualization window with:

Detected lane boundaries

Reference lines (green)

Central points (orange)

3D coordinates display

Area calculation

FPS counter

CSV file containing:

Timestamp (IST)

X, Y, Z coordinates

Frame number

Example CSV output:

text
Time,X,Y,Z,Frame
2024-05-15 14:30:45.123456,0.254,-0.142,1.873,42
2024-05-15 14:30:45.134567,0.261,-0.138,1.869,43

Customization
To adapt this system:

Train your own YOLOv8 model for different lane types

Adjust reference line intervals for different resolutions

Modify coordinate calculation for different cameras

Add additional output formats (JSON, database)

Implement ROS integration for robotic systems

Performance
Typical performance on NVIDIA RTX 3060:

640x480 resolution: 25-30 FPS

Higher resolutions may require model optimization

Troubleshooting
Common issues:

No depth data: Ensure camera is properly connected and calibrated

Poor lane detection:

Verify model matches your lane types

Adjust lighting conditions

Clean camera lens

Low FPS:

Reduce input resolution

Use smaller YOLO model variant

Enable GPU acceleration

