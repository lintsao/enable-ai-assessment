import argparse
import time

import cv2

from mouth_open_detector import MouthOpenDetector


def main():
    parser = argparse.ArgumentParser(
        description="Detect mouth existence and state (open/closed)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file; if omitted, use webcam",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="MAR threshold")
    args = parser.parse_args()

    detector = MouthOpenDetector(threshold=args.threshold)
    is_cam = args.video is None
    cap = cv2.VideoCapture(0 if is_cam else args.video, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {'webcam' if is_cam else args.video}")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, state, mar = detector.process(frame)
        color = (
            (0, 0, 255)
            if state == "Open"
            else (0, 255, 0) if state == "Closed" else (128, 128, 0)
        )
        cv2.putText(
            frame, f"Mouth: {state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        now = time.time()
        fps = 1.0 / (now - prev) if now != prev else 0.0
        prev = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Mouth Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
