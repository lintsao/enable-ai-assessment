# 🎯 Mouth Open Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)

**Real-time mouth open detection using multiple AI methods with beautiful web interface**

[🚀 Quick Start](#-quick-start) • [🔧 Features](#-features) • [📦 Installation](#-installation) • [🎮 Usage](#-usage) • [🛠️ Development](#️-development)

</div>

---

## ✨ Features

### 🎯 **Multi-Method Detection**
- **ONNX Model**: Deep learning-based detection model
- **MediaPipe**: Facial landmark-based detection
- **Dlib**: Facial landmark-based detection

### 🌐 **Real-time Web Interface**
- Beautiful Streamlit UI with real-time video streaming
- User authentication system (login/register)
- Live detection results overlay
- Adjustable thresholds for each method

### 📊 **Data Logging**
- Automatic database logging of detection results
- User session management
- Detection method and camera information tracking
- Historical data analysis

### 🎨 **Modern UI/UX**
- Responsive design with sidebar controls
- Real-time FPS and confidence display
- Color-coded detection states
- Professional styling

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- macOS (Apple Silicon supported)
- Webcam access

### 1. Clone & Setup
```bash
git clone <repository-url>
cd EnableAI
```

### 2. Install Dependencies
```bash
./setup.sh
source "mouth_open_detector/bin/activate"
pip install -r requirements.txt
```

### 3. Download Models
```bash
# Create models directory
mkdir -p model_weights

# Download Dlib model
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat model_weights/

# Ensure ONNX model is in place
# yawn_model_80.onnx should be in model_weights/
```

### 4. Run Application
```bash
make run
```

### 5. Access Web Interface
Open your browser and go to: **http://localhost:8501**

---

## 🎮 Usage Guide

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

## 🔧 Database Schema
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

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **MediaPipe** for facial landmark detection
- **Dlib** for facial landmark detection
- **Streamlit** for the beautiful web interface
- **ONNX Runtime** for model inference

---

<div align="center">

[⬆️ Back to Top](#-mouth-open-detection-system)

</div> 