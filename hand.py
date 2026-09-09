import cv2
import mediapipe as mp

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

detector = HandLandmarker.create_from_options(options)

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

timestamp = 0


# -------------------------------
# Finger counting function
# -------------------------------
def count_fingers(hand, handedness):

    fingers = 0

    # Thumb
    # Left hand aur right hand ka thumb opposite direction mein hota hai
    if handedness == "Right":
        if hand[4].x < hand[3].x:
            fingers += 1
    else:
        if hand[4].x > hand[3].x:
            fingers += 1

    # Index, Middle, Ring, Pinky
    # Tip upar hai aur base se aage hai = open
    finger_tips = [8, 12, 16, 20]
    finger_joints = [6, 10, 14, 18]

    for tip, joint in zip(finger_tips, finger_joints):

        if hand[tip].y < hand[joint].y:
            fingers += 1

    return fingers


# -------------------------------
# Main loop
# -------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera nahi mil raha!")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 33

    # Agar hand detect hua
    if result.hand_landmarks:

        for i, hand in enumerate(result.hand_landmarks):

            # Handedness
            handedness = result.handedness[i][0].category_name

            # Count fingers
            fingers = count_fingers(hand, handedness)

            # Draw points
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Display count
            cv2.putText(
                frame,
                f"{handedness} Hand: {fingers} Fingers",
                (20, 50 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Finger Counter", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()