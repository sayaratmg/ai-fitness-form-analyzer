import cv2
import mediapipe as mp
import numpy as np


class FitnessTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # setting up mediapipe pose with decent confidence thresholds
        # I tried lower values but tracking was too shaky
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # these three variables carry all the state I need
        self.rep_counter = 0
        self.current_stage = None       # "up" or "down"
        self.coach_feedback = "Get ready!"

    # standard 3-point angle formula using arctan2
    # a = first joint, b = middle joint (the vertex), c = end joint
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = (
            np.arctan2(c[1] - b[1], c[0] - b[0])
            - np.arctan2(a[1] - b[1], a[0] - b[0])
        )
        angle = np.abs(radians * 180.0 / np.pi)
        # clamp to 0-180 range
        if angle > 180.0:
            angle = 360 - angle
        return angle

    # just a small helper so I don't repeat landmark[x].x, landmark[x].y everywhere
    def get_coords(self, landmarks, landmark_enum):
        lm = landmarks[landmark_enum.value]
        return [lm.x, lm.y]

    # I check which hip is more visible to decide which side the person is facing
    # this way the tracker works whether you stand on the left or right of the camera
    def detect_side(self, landmarks):
        left_vis  = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility
        right_vis = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
        return "LEFT" if left_vis > right_vis else "RIGHT"

    # squat logic: I measure the knee angle (hip -> knee -> ankle)
    # below 90 deg = bottom of squat, above 160 deg = standing back up = 1 rep
    def track_squat(self, landmarks, frame):
        side = self.detect_side(landmarks)

        if side == "LEFT":
            hip   = self.get_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
            knee  = self.get_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
            ankle = self.get_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        else:
            hip   = self.get_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
            knee  = self.get_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
            ankle = self.get_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)

        angle = self.calculate_angle(hip, knee, ankle)

        # convert normalized coords to pixel position so I can draw the angle on screen
        pixel_node = tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int))

        if angle < 90:
            self.current_stage = "down"
            self.coach_feedback = "Good depth! Drive up."

        if angle > 160:
            if self.current_stage == "down":
                self.rep_counter += 1
                self.coach_feedback = "Great rep!"
            self.current_stage = "up"

        return angle, pixel_node

    # push-up logic: same idea but I measure the elbow angle (shoulder -> elbow -> wrist)
    # below 90 deg = chest close to ground, above 160 deg = arms extended = 1 rep
    def track_pushup(self, landmarks, frame):
        side = self.detect_side(landmarks)

        if side == "LEFT":
            shoulder = self.get_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow    = self.get_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist    = self.get_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        else:
            shoulder = self.get_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
            elbow    = self.get_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
            wrist    = self.get_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)

        angle = self.calculate_angle(shoulder, elbow, wrist)
        pixel_node = tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int))

        if angle < 90:
            self.current_stage = "down"
            self.coach_feedback = "Low enough! Push up."

        if angle > 160:
            if self.current_stage == "down":
                self.rep_counter += 1
                self.coach_feedback = "Excellent form!"
            self.current_stage = "up"

        return angle, pixel_node

    # main function called every frame from app.py
    # converts to RGB (mediapipe needs RGB), runs pose detection, then calls the right tracker
    def process_frame(self, frame, mode):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # draw the skeleton lines and joint dots on the frame
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            try:
                landmarks = results.pose_landmarks.landmark

                if mode == "SQUAT":
                    angle, pixel_node = self.track_squat(landmarks, frame)
                elif mode == "PUSHUP":
                    angle, pixel_node = self.track_pushup(landmarks, frame)
                else:
                    return frame

                # show the live joint angle next to the tracked joint
                cv2.putText(
                    frame,
                    f"{int(angle)} deg",
                    pixel_node,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            except Exception:
                # sometimes landmarks aren't fully detected mid-frame, just skip it
                pass

        return frame