import argparse
import time
import cv2
import mediapipe as mp
import numpy as np
import os

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321,
              375, 291, 308, 324, 318, 402, 317, 14, 87,
              178, 88, 95, 185, 40, 39, 37, 0, 267, 269,
              270, 409, 415, 310, 311, 312, 13, 82, 81,
              42, 183, 78]

LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310,
              415, 308, 324, 318, 402, 317, 14, 87, 178,
              88, 95, 61]

def compute_mar(landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    v = np.linalg.norm(pt(13) - pt(14))
    h_dist = np.linalg.norm(pt(78) - pt(308))
    return v / (h_dist + 1e-6)

def main(source, threshold):
    is_webcam = source is None
    cap = cv2.VideoCapture(0 if is_webcam else source, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {'webcam' if is_webcam else f'video: {source}'}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        state = "No Face"
        mar = 0.0
        draw_color = (0, 255, 0)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            mar = compute_mar(lm, w, h)
            state = "Open" if mar > threshold else "Closed"
            draw_color = (0, 0, 255) if state == "Open" else (0, 255, 0)

            # Draw lips
            outer_pts = [(int(lm[idx].x * w), int(lm[idx].y * h)) for idx in LIPS_OUTER]
            for i in range(len(outer_pts)):
                cv2.line(frame, outer_pts[i], outer_pts[(i + 1) % len(outer_pts)], draw_color, 2)

            inner_pts = [(int(lm[idx].x * w), int(lm[idx].y * h)) for idx in LIPS_INNER]
            for i in range(len(inner_pts)):
                cv2.line(frame, inner_pts[i], inner_pts[(i + 1) % len(inner_pts)], draw_color, 2)

        # Text
        cv2.putText(frame, f"Mouth: {state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Mouth State Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to input video file. If not set, webcam is used.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for MAR to detect open mouth")
    args = parser.parse_args()
    main(args.video, args.threshold)
