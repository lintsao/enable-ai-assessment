import os

import cv2
import dlib
import numpy as np

from constants import (COLOR_CLOSED, COLOR_OPEN, COLOR_UNKNOWN, COLOR_WHITE,
                       DLIB_PREDICTOR_PATH, FONT_SCALE, FONT_THICKNESS,
                       TEXT_POSITION_X, TEXT_POSITION_Y_CONFIDENCE,
                       TEXT_POSITION_Y_FPS, TEXT_POSITION_Y_STATE)


class DLibMouthDetector:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.detector = None
        self.predictor = None
        self.prev_time = None
        self.fps = 0.0
        self.frame_idx = 0

        # Initialize DLib face detector and facial landmark predictor
        try:
            self.detector = dlib.get_frontal_face_detector()
            # Try to load the shape predictor file
            if os.path.exists(DLIB_PREDICTOR_PATH):
                self.predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
            else:
                print(
                    f"Warning: {DLIB_PREDICTOR_PATH} not found. Please download it from:"
                )
                print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        except Exception as e:
            print(f"Error initializing DLib: {e}")

    def calculate_mar(self, landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR)
        MAR = (A + B) / (2 * C)
        where A, B are vertical distances and C is horizontal distance
        """
        # Mouth landmarks (0-67): 48-67 are mouth points
        # Vertical distances
        A = np.linalg.norm(
            landmarks[51] - landmarks[57]
        )  # Upper lip to lower lip (vertical)
        B = np.linalg.norm(
            landmarks[52] - landmarks[56]
        )  # Upper lip to lower lip (vertical)

        # Horizontal distance
        C = np.linalg.norm(landmarks[48] - landmarks[54])  # Left corner to right corner

        if C > 0:
            mar = (A + B) / (2 * C)
        else:
            mar = 0.0

        return mar

    def process(self, frame):
        """
        Process frame and detect mouth state
        Returns: (frame, state, mar)
        """
        if self.detector is None or self.predictor is None:
            return frame, "Unknown", 0.0

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        if len(faces) == 0:
            # No face detected
            cv2.putText(
                frame,
                "",
                (TEXT_POSITION_X, TEXT_POSITION_Y_STATE),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                COLOR_UNKNOWN,
                FONT_THICKNESS,
            )
            return frame, "Unknown", 0.0

        # Process the first face found
        face = faces[0]

        # Get facial landmarks
        landmarks = self.predictor(gray, face)

        # Convert landmarks to numpy array
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Calculate MAR
        mar = self.calculate_mar(points)

        # Determine mouth state based on MAR threshold
        if mar > self.threshold:
            state = "Open"
            color = COLOR_OPEN
        else:
            state = "Closed"
            color = COLOR_CLOSED

        # Draw face rectangle
        cv2.rectangle(
            frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2
        )

        # Draw mouth landmarks (points 48-67)
        for i in range(48, 68):
            cv2.circle(frame, (points[i][0], points[i][1]), 2, color, -1)

        # Draw mouth state text
        cv2.putText(
            frame,
            f"Mouth: {state}",
            (TEXT_POSITION_X, TEXT_POSITION_Y_STATE),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            FONT_THICKNESS,
        )

        # Draw MAR value
        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (TEXT_POSITION_X, TEXT_POSITION_Y_CONFIDENCE),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE * 0.8,
            COLOR_WHITE,
            FONT_THICKNESS,
        )

        # Calculate and draw FPS
        if self.prev_time is not None:
            import time

            now = time.time()
            self.fps = 1.0 / (now - self.prev_time) if now != self.prev_time else 0.0
            self.prev_time = now
        else:
            import time

            self.prev_time = time.time()

        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (TEXT_POSITION_X, TEXT_POSITION_Y_FPS),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE * 0.7,
            COLOR_WHITE,
            FONT_THICKNESS,
        )

        self.frame_idx += 1

        return frame, state, mar
