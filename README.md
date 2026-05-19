# Project Name: FlexiForm-AI (Real-Time AI Yoga & Squat Form Analyzer)

## Project Introduction
FlexiForm-AI is a real-time Computer Vision application designed to act as an automated personal fitness coach. The primary goal of this project is to assist users in maintaining correct posture and execution while performing physical exercises, specifically focusing on Squats and select Yoga poses.

By utilizing a standard webcam feed, the system leverages deep-learning-based human body pose estimation to capture coordinates of key skeletal landmarks. The application continuously applies biomechanical calculations to measure exact angles at critical joints (such as hips, knees, and ankles) to evaluate the user's movement form and automatically count correct repetitions.

### Key Features to Implement:
1. **Real-Time Body Tracking:** Extract 33 distinct anatomical skeletal keypoints using MediaPipe Pose.
2. **Geometric Joint Angle Calculation:** Implement vector trigonometry algorithms using NumPy to determine knee and hip angles in real-time.
3. **Dynamic Posture Feedback Engine:** Provide immediate visual alerts and color-coded feedback on the screen when posture drops below safety thresholds.
4. **State-Based Automated Repetition Counting:** Build a software state-machine logic to track exercise phases (e.g., descending vs. ascending in a squat) to increment counts securely without duplication.

### Tech Stack
- **Programming Language:** Python
- **Core Libraries:** OpenCV, MediaPipe, NumPy, Matplotlib