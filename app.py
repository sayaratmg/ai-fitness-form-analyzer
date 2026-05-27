import cv2
import mediapipe as mp
import numpy as np

# Set up MediaPipe drawing and pose tracking tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Trigonometry helper function for tracking angles
def calculate_angle(a, b, c):
    a = np.array(a) # Hip
    b = np.array(b) # Knee
    c = np.array(c) # Ankle
    
    # Calculate radians and convert to absolute degrees
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

rep_counter = 0
current_stage = None
form_feedback = "Get Ready!"

# Start the webcam stream
camera = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Webcam disconnected.")
            break
            
        # Flip colors to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Tracking Left side coordinates (Hip, Knee, Ankle)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
            
                angle = calculate_angle(hip, knee, ankle)
                
                # Get actual pixel mapping to render the angle text neatly on screen
                pixel_knee = tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int))
                cv2.putText(frame, f"{int(angle)} deg", pixel_knee, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                
                if angle < 90:
                    current_stage = "down"
                    form_feedback = "Good depth! Now push up."
                
                if angle > 160:
                    if current_stage == "down":
                        rep_counter += 1  # Successfully completed one full rep!
                        form_feedback = "Perfect rep!"
                    current_stage = "up"
                    
                if current_stage == "up" and angle > 110 and angle < 140:
                    form_feedback = "Go lower! Aim for 90 degrees."
            except Exception as e:
                pass
            
        # --- UI Dashboard Setup ---
        cv2.rectangle(frame, (10, 10), (350, 110), (40, 40, 40), -1)
        
        # Display the Repetition Count
        cv2.putText(frame, f"REPS: {rep_counter}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the Current Stage (UP/DOWN tracking)
        cv2.putText(frame, f"STAGE: {str(current_stage).upper()}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            
        # Display the Live Coaching Feedback
        cv2.putText(frame, f"COACH: {form_feedback}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('AI Fitness Coach', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()