# 🎯 Mouth Open Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)

**Real-time mouth open detection using multiple AI methods with web interface**

[✨ Features](#-features) • [🚀 Quick Start](#-quick-start) • [🎮 Usage](#-usage) • [🗂️ Database Schema](#️-database-schema) • [🛠️ Development](#️-development)

</div>

![Mouth Open Detection Demo](./demo/demo.gif)
---

## ✨ Features

### 🎯 **Multi-Method Detection**
- **ONNX Model**: Deep learning-based detection model
- **MediaPipe**: Facial landmark-based detection
- **Dlib**: Facial landmark-based detection

### 🌐 **Real-time Web Interface**
- Streamlit UI with real-time video streaming
- User authentication system (login/register)
- Adjustable thresholds for each method

### 📊 **Data Logging**
- Automatic database logging of detection results

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- macOS (Apple Silicon supported)
- Webcam access

### 1. Clone & Setup
```bash
git clone https://github.com/lintsao/enable-ai-assessment.git
cd enable-ai-assessment
```

### 2. Install Dependencies
```bash
./setup.sh
source mouth_open_detector/bin/activate
```

### 3. Download Models
```bash
mkdir -p model_weights
```

- Download Dlib model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and follow the instructions to extract the file

- Download ONNX model from https://github.com/iglaweb/HippoYD/blob/master/out_epoch_80_full/yawn_model_80.onnx

- ```shape_predictor_68_face_landmarks.dat.bz2``` and ```yawn_model_80.onnx``` should be in ```model_weights/```

### 4. Run Application
```bash
make run
```

### 5. Access Web Interface
Open your browser and go to: **http://localhost:8501**

---

## 🎮 Usage

### 1. **Authentication**
- Register with your email and password
- Or sign in if you already have an account

### 2. **Select Detection Method**
Choose from three available methods:
- **ONNX**
- **MediaPipe**
- **Dlib**

### 3. **Adjust Parameters**
- **ONNX**: Confidence threshold (0.0-1.0)
- **MediaPipe/Dlib**: MAR threshold (0.0-1.0)

### 4. **Start Detection**
- Click "Start" to begin real-time detection
- Results are displayed on video feed
- Database automatically logs state changes

---

## 🗂️ Database Schema
```sql
mouth_state (
    id, timestamp, email, state, mar, 
    source, frame_idx, threshold, 
    detection_method, camera_info
)
```

---

## 🛠️ Development

### Project Structure
```
EnableAI/
├── model_weights/          # AI models
│   ├── yawn_model_80.onnx
│   └── shape_predictor_68_face_landmarks.dat
├── src/                    # Source code
│   ├── constants.py        # Configuration
│   ├── db_utils.py         # Database utilities
│   ├── dlib_mouth_detector.py
│   ├── mouth_open_detector.py
│   ├── mouth_open_detector_local.py
│   └── mouth_open_detector_streamlit_combined.py
├── requirements.txt        # Dependencies
├── Makefile               # Development commands
└── README.md
```

### Available Commands
```bash
make run        # Start application
make format     # Format code (black + isort)
make lint       # Run linting (flake8)
make check      # Format + lint
make clean      # Clean generated files
```

---

<div align="center">

[⬆️ Back to Top](#-mouth-open-detection-system)

</div> 
