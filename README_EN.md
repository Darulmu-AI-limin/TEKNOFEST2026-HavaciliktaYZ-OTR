<div align="center">

# ✈️ BlackScope: Artificial Intelligence in Aviation System
### TEKNOFEST 2026 — Artificial Intelligence in Aviation Competition
#### **Team:** Darülmu-AI-limin | **Team ID:** #754906 | **Application ID:** #4983561

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11%20%7C%20v8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TEKNOFEST](https://img.shields.io/badge/TEKNOFEST-2026-red.svg)](https://www.teknofest.org/)

**[🇹🇷 Türkçe](README.md) | [🇬🇧 English](README_EN.md)**

<p align="center">
  <b>A multitask artificial intelligence and computer vision system developed for the TEKNOFEST 2026 Artificial Intelligence in Aviation Competition, processing video streams from Unmanned Aerial Vehicle (UAV) cameras completely offline and in real-time.</b>
</p>

</div>

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Team Structure](#-team-structure)
- [Core Capabilities and Task Solutions](#-core-capabilities-and-task-solutions)
- [System Architecture](#-system-architecture)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Server Communication Protocol and JSON Schema](#-server-communication-protocol-and-json-schema)
- [License](#-license)

---

## 📌 Project Overview

In the competition scenario, the system receives a 7.5 FPS video stream along with synchronized telemetry data from the UAV and transmits responses in standard JSON format over the local network (Ethernet) to the competition server at a processing rate of at least 1 frame per second (FPS).

BlackScope unifies three independent competition tasks into an integrated, single-pass pipeline:
1. **Object Detection, Motion Analysis, and Landing Zone Feasibility**
2. **Visual-Based Position Estimation (Visual Odometry)**
3. **Undefined / Reference Object Matching**

---

## 👥 Team Structure

| Role | Responsibilities |
|:---|:---|
| **Advisor (Academic Coordinator)** | Providing academic guidance and supervision in software and artificial intelligence. |
| **Team Captain** | System integration, server communication, project management, dataset curation, annotation tools, and validation test scenarios. |
| **Member 1** | YOLOv8/v11 model training, data annotation, motion analysis. |
| **Member 2** | Optical flow, position estimation algorithm. |
| **Member 3** | Feature extraction, template matching, adaptive recognition. |

---

## 🚀 Core Capabilities and Task Solutions

### 1. Task 1: Object Detection, Motion Analysis, and Landing Feasibility
* **YOLOv11-Based Detection:** Pretrained on COCO and VisDrone2019 datasets and fine-tuned on custom UAV aerial imagery, detecting 4 distinct classes:
  * `0`: Vehicle
  * `1`: Human
  * `2`: UAP (Flying Car Parking Area)
  * `3`: UAİ (Flying Ambulance Landing Area)
* **Camera Ego-Motion Compensation:** To prevent stationary vehicles from being falsely detected as moving (false positives) due to the UAV's flight motion, an inter-frame homography matrix ($H$) is estimated using ORB features and RANSAC. Once camera motion is compensated, the true ground motion status of vehicles (`is_moving`: 1 / 0) is determined.
* **Landing Area Feasibility:** An Intersection over Union (IoU) overlap analysis is performed between detected UAP/UAİ zones and obstacles/vehicles. If an obstacle is detected on the area, landing feasibility is reported as `0` (unsuitable); if the zone is clear, it is reported as `1` (suitable).

### 2. Task 2: Position Estimation via Visual Odometry
* **GPS Outage Management:** The visual odometry module is engaged automatically whenever telemetry reports `gps_health_status = 0`.
* **Lucas-Kanade Pyramidal Optical Flow:** Keypoints to track are identified using the Shi-Tomasi algorithm, and optical flow is tracked between consecutive video frames.
* **Pose Estimation and Metric Scaling:** Camera rotation matrix ($R$) and translation vector are estimated using Essential Matrix ($E$) estimation with RANSAC filtering. Using the camera intrinsic matrix ($K$) and altitude ($Z$) data, relative pixel displacements are converted into metric spatial coordinates ($\Delta X, \Delta Y, \Delta Z$).
* **Drift Reset:** Once valid GPS packets recover (`health_status = 1`), the system resynchronizes its absolute position using the ground truth server telemetry data.

### 3. Task 3: Undefined / Reference Object Matching
* **Adaptive and Hierarchical Matching:** A hybrid strategy is utilized to detect target reference images provided at the beginning of the competition session during flight:
  * **Primary (ORB):** Fast, computationally efficient feature extraction and matching.
  * **Fallback (SIFT):** Automatically engaged under challenging conditions such as extreme perspective angles, scale variations, or RGB-to-thermal modality differences.
* **Homography and Bounding Box Verification:** Following Brute-Force matching and Lowe's Ratio Test, RANSAC homography estimation is applied to extract the precise bounding box of the target object within image coordinates.

---

## 🏗️ System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │              UAV Video Stream & Telemetry              │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │       Preprocessing (CLAHE, Histogram Equalization)    │
       └───────────────────────────┬────────────────────────────┘
                                   │
         ┌─────────────────────────┼────────────────────────┐
         │                         │                        │
         ▼                         ▼                        ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│      Task 1       │    │      Task 2       │    │      Task 3       │
│ YOLOv11m Detection│    │  Visual Odometry  │    │ Adaptive Matching │
│  (Vehicle, Human, │    │  (GPS=0 Status)   │    │   (ORB / SIFT)    │
│     UAP, UAİ)     │    │   Lucas-Kanade    │    │                   │
│         │         │    │  Essential Matrix │    │ Brute-Force +     │
│         ▼         │    │  Metric ΔX,ΔY,ΔZ  │    │ Homography        │
│   Ego-Motion &    │    └─────────┬─────────┘    └─────────┬─────────┘
│ Landing Feasibility              │                        │
└────────┬──────────┘              │                        │
         │                         │                        │
         └─────────────────────────┼────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │    Integrated JSON Output (TEKNOFEST Specification)    │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │      Local Competition Server (HTTP POST /results)     │
       └────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```text
BlackScope/
├── configs/
│   ├── server_config.json      # Server IP, port, and camera intrinsic parameters
│   └── visdrone.yaml           # Dataset class definitions
├── models/
│   └── yolo11m_visdrone.pt     # Trained object detection model weights
├── results/                    # Test plots and benchmark outputs
│   ├── detection_plot.png
│   ├── matching_plot.png
│   └── odometry_plot.png
├── scripts/
│   ├── run_pipeline.py         # Main script launching competition mode or video test
│   ├── benchmark.py            # Module-based performance and speed benchmarks
│   ├── augment_*.py            # Thermal, night, and rain simulation scripts
│   └── train_model.py          # YOLO model fine-tuning training script
├── src/
│   ├── detection/
│   │   ├── ego_motion.py       # Homography-based camera ego-motion compensation
│   │   ├── landing_analysis.py # Obstacle and landing feasibility analysis for UAP / UAİ
│   │   ├── preprocessing.py    # CLAHE and contrast equalization steps
│   │   └── train_yolo.py       # Model loading and inference
│   ├── integration/
│   │   ├── pipeline.py         # Main pipeline running all 3 tasks in a single pass
│   │   └── server_comm.py      # Server communication protocol (GET /frame, POST /results)
│   ├── matching/
│   │   ├── adaptive_match.py   # Hybrid ORB/SIFT matching manager
│   │   ├── orb_matcher.py      # ORB feature extraction and matching
│   │   └── sift_matcher.py     # SIFT feature extraction and matching
│   ├── odometry/
│   │   ├── optical_flow.py     # Lucas-Kanade pyramidal optical flow
│   │   └── position_est.py     # Relative pose and metric position estimation
│   └── utils/
│       ├── metrics.py          # IoU computation and evaluation metrics
│       └── visualization.py    # Bounding box and telemetry drawing utilities
├── tests/                      # Automated unit tests
│   ├── test_detection.py
│   ├── test_matching.py
│   ├── test_odometry.py
│   └── test_server.py
├── LICENSE                     # MIT License
├── requirements.txt            # Python library dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/username/BlackScope.git
cd BlackScope
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The system is engineered to operate strictly offline during the competition. Ensure all pip dependencies and model weight files are downloaded and verified in your local environment prior to arriving at the competition venue.

---

## 🚦 Usage

### 1. Competition Mode (Live Server Connection)
Connects to the competition server via Ethernet, fetches frames, processes them, and transmits results in real time:
```bash
python scripts/run_pipeline.py --mode competition --server http://192.168.1.100:8080
```

### 2. Offline Video Test Mode
To test the pipeline on a recorded UAV flight video and save predictions as JSON:
```bash
python scripts/run_pipeline.py --mode video --video test_flight.mp4 --output results/
```

### 3. Automated Unit Tests
To verify the correctness of all system components:
```bash
python -m unittest discover tests/
```

### 4. Performance Benchmarks
To profile the FPS and inference latency of individual modules:
```bash
python scripts/benchmark.py
```

---

## 📡 Server Communication Protocol and JSON Schema

In accordance with competition rules, the system fetches telemetry-accompanied image data via `GET /frame` and sends predictions to the `POST /results` endpoint formatted as follows:

```json
{
  "detected_objects": [
    {
      "cls": "0",
      "landing_status": "-1",
      "motion_status": "1",
      "top_left_x": 120,
      "top_left_y": 80,
      "bottom_right_x": 260,
      "bottom_right_y": 190
    },
    {
      "cls": "2",
      "landing_status": "1",
      "motion_status": "-1",
      "top_left_x": 300,
      "top_left_y": 150,
      "bottom_right_x": 480,
      "bottom_right_y": 330
    }
  ],
  "detected_translations": [
    {
      "translation_x": 1.4523,
      "translation_y": -0.8921,
      "translation_z": 12.4012
    }
  ],
  "detected_undefined_objects": [
    {
      "object_id": "target_1",
      "top_left_x": 210,
      "top_left_y": 140,
      "bottom_right_x": 310,
      "bottom_right_y": 240
    }
  ]
}
```

* `cls`: `0` (Vehicle), `1` (Human), `2` (UAP), `3` (UAİ)
* `landing_status`: `1` (suitable) or `0` (occupied) for UAP/UAİ; `-1` for all other classes
* `motion_status`: `1` (moving) or `0` (stationary) for Vehicle; `-1` for all other classes
* `detected_translations`: Generated via visual odometry strictly when `gps_health_status = 0`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Refer to the `LICENSE` file for details.
