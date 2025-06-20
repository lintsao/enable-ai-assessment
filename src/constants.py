# ----------- ONNX Model Settings -----------
MODEL_PATH = "./model_weights/yawn_model_80.onnx"
MAX_IMAGE_WIDTH = 100
MAX_IMAGE_HEIGHT = 100
IMAGE_PAIR_SIZE = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)

# ----------- Database Settings -----------
DB_PATH = "mouth_state.db"

# ----------- WebRTC Configuration -----------
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ----------- Video Processing Settings -----------
VIDEO_SCALE_FACTOR = 0.5
VIDEO_WIDTH_IDEAL = 640
VIDEO_HEIGHT_IDEAL = 480

# ----------- Text Display Settings -----------
FONT_SCALE = 0.6
FONT_THICKNESS = 1
TEXT_POSITION_X = 10
TEXT_POSITION_Y_STATE = 25
TEXT_POSITION_Y_CONFIDENCE = 45
TEXT_POSITION_Y_FPS = 65

# ----------- Color Settings -----------
COLOR_OPEN = (0, 0, 255)  # Red for mouth open
COLOR_CLOSED = (0, 255, 0)  # Green for mouth closed
COLOR_UNKNOWN = (128, 128, 128)  # Gray for unknown state
COLOR_WHITE = (255, 255, 255)  # White for text

# ----------- Threshold Defaults -----------
DEFAULT_ONNX_THRESHOLD = 0.5
DEFAULT_MEDIAPIPE_THRESHOLD = 0.2
DEFAULT_DLIB_THRESHOLD = 0.3

# ----------- Session State Keys -----------
SESSION_KEYS = {
    "login_email": "",
    "login_state": False,
    "detection_method": "onnx",
    "current_state": "Unknown",
    "current_confidence": 0.0,
    "current_fps": 0.0,
    "current_frame": 0,
}

# ----------- Detection Method Names -----------
DETECTION_METHODS = ["onnx", "mediapipe", "dlib"]

# ----------- Slider Settings -----------
SLIDER_MIN = 0.0
SLIDER_MAX = 1.0
SLIDER_STEP = 0.01

# ----------- DLib Settings -----------
DLIB_PREDICTOR_PATH = "model_weights/shape_predictor_68_face_landmarks.dat"
DLIB_DOWNLOAD_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
