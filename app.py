import cv2
from pose_module import FitnessTracker

# start in squat mode by default, user can switch with keyboard
tracker = FitnessTracker()
mode = "SQUAT"

# 0 = default webcam, change to 1 or 2 if you have multiple cameras
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam. Check your camera index.")
    exit()

print("Fitness tracker started!")
print("Controls:  [S] Squats   [P] Push-ups   [R] Reset counter   [Q] Quit")


def draw_hud(frame, mode, tracker):
    # dark panel background so text is always readable regardless of background
    cv2.rectangle(frame, (10, 10), (440, 145), (20, 20, 20), -1)

    # thin green border around the panel, looks cleaner than no border
    cv2.rectangle(frame, (10, 10), (440, 145), (0, 200, 100), 1)

    # show current mode and available keys as a reminder
    cv2.putText(
        frame,
        f"MODE: {mode}   [S] Squats  [P] Push-ups",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # rep count in bigger green text so it's easy to glance at while working out
    cv2.putText(
        frame,
        f"REPS: {tracker.rep_counter}",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 100),
        2,
        cv2.LINE_AA
    )

    # coach feedback in cyan, smaller so it doesn't crowd the rep number
    cv2.putText(
        frame,
        f"COACH: {tracker.coach_feedback}",
        (20, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 230, 255),
        1,
        cv2.LINE_AA
    )


while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        print("Lost camera feed, stopping.")
        break

    # flip horizontally so it acts like a mirror — much easier to follow your form
    frame = cv2.flip(frame, 1)

    # run pose detection and rep counting for the current mode
    frame = tracker.process_frame(frame, mode)

    # draw the info panel on top of the processed frame
    draw_hud(frame, mode, tracker)

    cv2.imshow("Smart AI Fitness Coach", frame)

    # check for key presses every 10ms
    key = cv2.waitKey(10) & 0xFF

    if key == ord("q"):
        print("Quitting...")
        break

    elif key == ord("s"):
        mode = "SQUAT"
        tracker.coach_feedback = "Squat mode — let's go!"

    elif key == ord("p"):
        mode = "PUSHUP"
        tracker.coach_feedback = "Push-up mode — let's go!"

    elif key == ord("r"):
        # reset everything so you can start a fresh set
        tracker.rep_counter = 0
        tracker.current_stage = None
        tracker.coach_feedback = "Counter reset!"


camera.release()
cv2.destroyAllWindows()