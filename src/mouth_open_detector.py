import cv2
import mediapipe as mp
import numpy as np

LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,
              178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,
              42,183,78]
LIPS_INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,
              88,95,61]

def compute_mar(landmarks, w:int, h:int) -> float:
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    v = np.linalg.norm(pt(13) - pt(14))
    h_dist = np.linalg.norm(pt(78) - pt(308)) + 1e-6
    return v / h_dist

def compute_mar_v2(landmarks, w: int, h: int) -> float:
    def pt(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    # 多點平均
    upper_idxs = [13, 82, 312]
    lower_idxs = [14, 87, 317]
    upper_mean = np.mean([pt(i) for i in upper_idxs], axis=0)
    lower_mean = np.mean([pt(i) for i in lower_idxs], axis=0)
    v = np.linalg.norm(upper_mean - lower_mean)
    h_dist = np.linalg.norm(pt(78) - pt(308)) + 1e-6
    return v / h_dist

class MouthOpenDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        state, mar = "Not Detected", 0.0
        color = (128,128,0)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            mar = compute_mar_v2(lm, w, h)
            state = "Open" if mar > self.threshold else "Closed"
            color = (0,0,255) if state=="Open" else (0,255,0)
            outer = [(int(lm[i].x*w), int(lm[i].y*h)) for i in LIPS_OUTER]
            inner = [(int(lm[i].x*w), int(lm[i].y*h)) for i in LIPS_INNER]
            for pts in (outer, inner):
                for i in range(len(pts)):
                    cv2.line(frame, pts[i], pts[(i+1)%len(pts)], color, 2)
        return frame, state, mar