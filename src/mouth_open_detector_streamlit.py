import streamlit as st
import av
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from src.mouth_open_detector import MouthOpenDetector
from src.db_utils import insert_state, init_db, register_user, check_user, user_exists

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

DB_PATH = "mouth_state.db"
init_db(DB_PATH)

class MouthOpenStreamlitProcessor(VideoProcessorBase):
    def __init__(self, email, threshold=0.2):
        self.threshold = threshold
        self.detector = MouthOpenDetector(threshold=self.threshold)
        self.prev_time = time.time()
        self.fps = 0.0
        self.prev_state = None
        self.frame_idx = 0
        self.email = email
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, state, mar = self.detector.process(img)
        color = (0,0,255) if state=="Open" else (0,255,0) if state=="Closed" else (128,128,0)
        cv2.putText(img, f"Mouth: {state}", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"MAR: {mar:.3f}", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        now = time.time()
        self.fps = 1.0/(now-self.prev_time) if now!=self.prev_time else 0.0
        self.prev_time = now
        cv2.putText(img, f"FPS: {self.fps:.1f}", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # 寫入資料庫（只在狀態變化時）
        if state != self.prev_state:
            insert_state(self.email, state, mar, "webcam-streamlit", self.frame_idx, self.threshold, db_path=DB_PATH)
            self.prev_state = state
        self.frame_idx += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def processor_factory(threshold: float, email: str):
    def factory():
        proc = MouthOpenStreamlitProcessor(email=email, threshold=threshold)
        proc.threshold = threshold
        proc.detector.threshold = threshold
        return proc
    return factory

def main():
    # Always initialize session_state keys
    for k, v in {
        "login_email": "",
        "login_state": False
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.title("Mouth Open Detection with Live Webcam")
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
                    st.experimental_rerun()
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
                    st.experimental_rerun()
                    return
                else:
                    st.sidebar.error("Registration failed. Please try again.")
    else:
        st.sidebar.header(f"Signed in as: {st.session_state.get('login_email', '')}")
        if st.sidebar.button("Sign Out"):
            st.session_state["login_state"] = False
            st.session_state["login_email"] = ""
            st.experimental_rerun()
            return
        threshold = st.sidebar.slider("MAR Threshold", 0.0, 1.0, 0.2, 0.01)
        webrtc_streamer(
            key=f"mouth-open-{st.session_state.get('login_email', '')}-{threshold}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=processor_factory(threshold, st.session_state.get('login_email', '')),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

if __name__ == "__main__":
    main()
