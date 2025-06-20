import datetime
import os
import time

import av
import cv2
import numpy as np
import onnxruntime as rt
import streamlit as st
from streamlit_webrtc import (RTCConfiguration, VideoProcessorBase, WebRtcMode,
                              webrtc_streamer)

from constants import (COLOR_CLOSED, COLOR_OPEN, COLOR_UNKNOWN, COLOR_WHITE,
                       DB_PATH, DEFAULT_DLIB_THRESHOLD,
                       DEFAULT_MEDIAPIPE_THRESHOLD, DEFAULT_ONNX_THRESHOLD,
                       DETECTION_METHODS, DLIB_DOWNLOAD_URL,
                       DLIB_PREDICTOR_PATH, FONT_SCALE, FONT_THICKNESS,
                       IMAGE_PAIR_SIZE, MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH,
                       MODEL_PATH, RTC_CONFIGURATION, SESSION_KEYS, SLIDER_MAX,
                       SLIDER_MIN, SLIDER_STEP, TEXT_POSITION_X,
                       TEXT_POSITION_Y_CONFIDENCE, TEXT_POSITION_Y_FPS,
                       TEXT_POSITION_Y_STATE, VIDEO_HEIGHT_IDEAL,
                       VIDEO_SCALE_FACTOR, VIDEO_WIDTH_IDEAL)
from db_utils import (check_user, init_db, insert_state, register_user,
                      user_exists, get_camera_info_from_frame)

# Check if model file exists and load it
onnx_sess = None
input_name = None
label_name = None

if os.path.exists(MODEL_PATH):
    try:
        onnx_sess = rt.InferenceSession(MODEL_PATH)
        input_name = onnx_sess.get_inputs()[0].name
        label_name = onnx_sess.get_outputs()[0].name
    except Exception as e:
        st.warning(f"Failed to load ONNX model: {e}")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

DB_PATH = "mouth_state.db"
init_db(DB_PATH)


def get_timestamp_ms():
    return int(
        (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
        * 1000
    )


def prepare_input_blob(im):
    if im.shape[0] != MAX_IMAGE_WIDTH or im.shape[1] != MAX_IMAGE_HEIGHT:
        im = cv2.resize(im, IMAGE_PAIR_SIZE)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def detect_image_onnx(frame):
    gray_img = prepare_input_blob(frame)
    time_start = get_timestamp_ms()
    image_frame = gray_img[:, :, np.newaxis]
    image_frame = image_frame / 255.0
    image_frame = np.expand_dims(image_frame, 0).astype(np.float32)
    pred = onnx_sess.run([label_name], {input_name: image_frame})[0]
    pred = np.squeeze(pred)
    pred = round(float(pred), 2)
    time_diff = get_timestamp_ms() - time_start
    print(f"ONNX Prediction: {pred:.2f}; time: {time_diff} ms")
    return pred


# Try to import MediaPipe detector
try:
    from mouth_open_detector import MouthOpenDetector

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning(
        "MediaPipe detector is not available. Please ensure all dependencies are installed."
    )

# Try to import DLib detector
try:
    from dlib_mouth_detector import DLibMouthDetector

    DLIB_AVAILABLE = True
    # Check if DLib predictor file exists
    if not os.path.exists(DLIB_PREDICTOR_PATH):
        st.warning(f"DLib predictor file not found: {DLIB_PREDICTOR_PATH}")
        st.info(f"Please download it from: {DLIB_DOWNLOAD_URL}")
        DLIB_AVAILABLE = False
except ImportError:
    DLIB_AVAILABLE = False
    st.warning("DLib detector is not available. Please install dlib: pip install dlib")


class MouthOpenONNXProcessor(VideoProcessorBase):
    def __init__(
        self, email, threshold=DEFAULT_ONNX_THRESHOLD, detection_method="onnx"
    ):
        self.threshold = threshold
        self.detection_method = detection_method
        self.prev_time = time.time()
        self.fps = 0.0
        self.prev_state = None
        self.frame_idx = 0
        self.email = email
        self.camera_info = "WebRTC Camera - Initializing..."

        # Initialize detectors
        if detection_method == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self.detector = MouthOpenDetector(threshold=self.threshold)
        elif detection_method == "dlib" and DLIB_AVAILABLE:
            self.detector = DLibMouthDetector(threshold=self.threshold)
        elif detection_method == "onnx" and onnx_sess is not None:
            self.detector = None
        else:
            st.error(f"Detection method {detection_method} is not available")
            self.detector = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Update camera info from actual frame
        if self.frame_idx == 0:
            self.camera_info = get_camera_info_from_frame(img)

        # Resize video to smaller size
        height, width = img.shape[:2]
        new_width = int(width * VIDEO_SCALE_FACTOR)
        new_height = int(height * VIDEO_SCALE_FACTOR)
        img = cv2.resize(img, (new_width, new_height))

        # Process based on detection method
        if self.detection_method == "mediapipe" and self.detector:
            img, state, mar = self.detector.process(img)
            confidence = mar
            color = (
                COLOR_OPEN
                if state == "Open"
                else COLOR_CLOSED if state == "Closed" else COLOR_UNKNOWN
            )
        elif self.detection_method == "dlib" and self.detector:
            img, state, mar = self.detector.process(img)
            confidence = mar
            color = (
                COLOR_OPEN
                if state == "Open"
                else COLOR_CLOSED if state == "Closed" else COLOR_UNKNOWN
            )
        elif self.detection_method == "onnx" and onnx_sess is not None:
            pred = detect_image_onnx(img)
            is_mouth_opened = pred >= self.threshold
            state = "Open" if is_mouth_opened else "Closed"
            confidence = pred
            color = COLOR_OPEN if state == "Open" else COLOR_CLOSED
        else:
            # Default state
            state = "Unknown"
            confidence = 0.0
            color = COLOR_UNKNOWN

        # Adjust text size and position for smaller window
        cv2.putText(
            img,
            f"Mouth: {state}",
            (TEXT_POSITION_X, TEXT_POSITION_Y_STATE),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            FONT_THICKNESS,
        )

        # Display different metrics based on detection method
        if self.detection_method in ["mediapipe", "dlib"]:
            cv2.putText(
                img,
                f"MAR: {confidence:.3f}",
                (TEXT_POSITION_X, TEXT_POSITION_Y_CONFIDENCE),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE * 0.8,
                COLOR_WHITE,
                FONT_THICKNESS,
            )
        else:
            cv2.putText(
                img,
                f"Confidence: {confidence:.2f}",
                (TEXT_POSITION_X, TEXT_POSITION_Y_CONFIDENCE),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE * 0.8,
                COLOR_WHITE,
                FONT_THICKNESS,
            )

        # FPS calculation
        now = time.time()
        self.fps = 1.0 / (now - self.prev_time) if now != self.prev_time else 0.0
        self.prev_time = now
        cv2.putText(
            img,
            f"FPS: {self.fps:.1f}",
            (TEXT_POSITION_X, TEXT_POSITION_Y_FPS),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE * 0.7,
            COLOR_WHITE,
            FONT_THICKNESS,
        )

        # Database logging
        if state != self.prev_state:
            insert_state(
                self.email,
                state,
                confidence,
                f"webcam-streamlit-{self.detection_method}",
                self.frame_idx,
                self.threshold,
                detection_method=self.detection_method,
                camera_info=self.camera_info,
                db_path=DB_PATH,
            )
            self.prev_state = state
        self.frame_idx += 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def processor_factory(threshold: float, email: str, detection_method: str):
    def factory():
        proc = MouthOpenONNXProcessor(
            email=email, threshold=threshold, detection_method=detection_method
        )
        proc.threshold = threshold
        if proc.detector:
            proc.detector.threshold = threshold
        return proc

    return factory


def main():
    # Always initialize session_state keys
    for k, v in SESSION_KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.header("Mouth Open Detection with Live Webcam")

    # Login form
    if not st.session_state["login_state"]:
        st.sidebar.header("Sign In")
        email = st.sidebar.text_input("Email").strip().lower()
        password = st.sidebar.text_input("Password", type="password")
        login_btn = st.sidebar.button("Sign In")
        signup_btn = st.sidebar.button("Sign Up")

        if login_btn:
            if not email or not password:
                st.sidebar.warning("Please enter your email and password.")
            elif user_exists(email, DB_PATH):
                if check_user(email, password, DB_PATH):
                    st.session_state["login_state"] = True
                    st.session_state["login_email"] = email
                    st.sidebar.success(f"Signed in as: {email}")
                    st.rerun()
                    return
                else:
                    st.sidebar.error("Incorrect password.")
            else:
                st.sidebar.error("This email is not registered. Please sign up.")

        if signup_btn:
            if not email or not password:
                st.sidebar.warning("Please enter your email and password.")
            elif user_exists(email, DB_PATH):
                st.sidebar.error("This email is already registered. Please sign in.")
            else:
                ok = register_user(email, password, DB_PATH)
                if ok:
                    st.session_state["login_state"] = True
                    st.session_state["login_email"] = email
                    st.sidebar.success(f"Registered and signed in as: {email}")
                    st.rerun()
                    return
                else:
                    st.sidebar.error("Registration failed. Please try again.")
    else:
        st.sidebar.header(f"Signed in as: {st.session_state.get('login_email', '')}")
        if st.sidebar.button("Sign Out"):
            st.session_state["login_state"] = False
            st.session_state["login_email"] = ""
            st.rerun()
            return

        # Detection method selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Detection Settings")

        detection_method = st.sidebar.selectbox(
            "Detection Method",
            options=DETECTION_METHODS,
            index=(
                0
                if st.session_state.get("detection_method") == "onnx"
                else 1 if st.session_state.get("detection_method") == "mediapipe" else 2
            ),
            help="Choose the detection method to use",
        )

        # Show availability status
        if detection_method == "onnx":
            if onnx_sess is not None:
                st.sidebar.success("✅ ONNX model loaded")
            else:
                st.sidebar.error("❌ ONNX model not available")
                detection_method = (
                    "mediapipe"
                    if MEDIAPIPE_AVAILABLE
                    else "dlib" if DLIB_AVAILABLE else None
                )

        if detection_method == "mediapipe":
            if MEDIAPIPE_AVAILABLE:
                st.sidebar.success("✅ MediaPipe detector available")
            else:
                st.sidebar.error("❌ MediaPipe detector not available")
                detection_method = (
                    "onnx"
                    if onnx_sess is not None
                    else "dlib" if DLIB_AVAILABLE else None
                )

        if detection_method == "dlib":
            if DLIB_AVAILABLE:
                st.sidebar.success("✅ DLib detector available")
            else:
                st.sidebar.error("❌ DLib detector not available")
                detection_method = (
                    "onnx"
                    if onnx_sess is not None
                    else "mediapipe" if MEDIAPIPE_AVAILABLE else None
                )

        # Threshold setting
        if detection_method == "onnx":
            threshold = st.sidebar.slider(
                "Confidence Threshold",
                SLIDER_MIN,
                SLIDER_MAX,
                DEFAULT_ONNX_THRESHOLD,
                SLIDER_STEP,
            )
        elif detection_method == "dlib":
            threshold = st.sidebar.slider(
                "MAR Threshold",
                SLIDER_MIN,
                SLIDER_MAX,
                DEFAULT_DLIB_THRESHOLD,
                SLIDER_STEP,
            )
        else:
            threshold = st.sidebar.slider(
                "MAR Threshold",
                SLIDER_MIN,
                SLIDER_MAX,
                DEFAULT_MEDIAPIPE_THRESHOLD,
                SLIDER_STEP,
            )

        # Update session state
        st.session_state["detection_method"] = detection_method

        # Display technical information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Technical Information")
        if detection_method == "onnx":
            st.sidebar.markdown(f"- **Model:** ONNX Runtime")
            st.sidebar.markdown(
                f"- **Input Size:** {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"
            )
        elif detection_method == "dlib":
            st.sidebar.markdown(f"- **Model:** DLib + Facial Landmarks")
            st.sidebar.markdown(f"- **Metric:** MAR (Mouth Aspect Ratio)")
        else:
            st.sidebar.markdown(f"- **Model:** MediaPipe + Facial Landmarks")
            st.sidebar.markdown(f"- **Metric:** MAR (Mouth Aspect Ratio)")

        # Only show streamer when detection method is available
        if detection_method:
            webrtc_streamer(
                key=f"mouth-open-{st.session_state.get('login_email', '')}-{detection_method}-{threshold}",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=processor_factory(
                    threshold, st.session_state.get("login_email", ""), detection_method
                ),
                frontend_rtc_configuration=RTCConfiguration(RTC_CONFIGURATION),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": VIDEO_WIDTH_IDEAL},
                        "height": {"ideal": VIDEO_HEIGHT_IDEAL},
                    },
                    "audio": False,
                },
            )
        else:
            st.error(
                "No detection method available. Please ensure all necessary dependencies are installed."
            )


if __name__ == "__main__":
    main()
