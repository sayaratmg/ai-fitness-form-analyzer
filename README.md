# AI-Driven Fitness Form Tracker & Rep Counter

## About the Project
With the rise of home workouts and remote fitness, it's easy for people to injure themselves or perform exercises inefficiently without professional guidance. This project focuses on building a real-time, interactive virtual fitness coach that monitors workout posture right from a standard laptop webcam. 

The main goal is to create a lightweight Python application that tracks a user's body landmarks while they do exercises like squats or specific yoga poses. By analyzing the structural relationships between different joints, the app will give immediate, on-screen advice to help the user correct their form and will automatically count valid repetitions.

### What I'm Building:
1. **Real-Time Landmark Detection:** Using MediaPipe Pose to smoothly track key skeletal joints across every video frame without requiring a heavy external GPU.
2. **Biomechanics & Geometric Analysis:** Applying vector geometry to measure live movement angles at the hips, knees, and ankles to evaluate posture accuracy.
3. **Smart Repetition Tracking:** Implementing a robust state-machine logic (tracking "Up" vs. "Down" phases) so the counter only increments when a full, safe range of motion is achieved, completely avoiding double-counting errors.
4. **Interactive Visual Overlay:** Creating a clean OpenCV-based user interface that displays a dynamic skeleton mesh, live angle metrics, and color-coded real-time warnings when posture is incorrect.

### Tools & Technologies
- **Language:** Python
- **Libraries:** OpenCV, MediaPipe, NumPy, Matplotlib