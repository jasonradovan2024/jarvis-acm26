import cv2
import mediapipe as mp

# Set up mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#open cam
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Mediapipe wants RGB but open cv gives BGR so flip it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # Print coords every 30 frames using first detected hand
        if frame_count % 30 == 0:
            hand = results.multi_hand_landmarks[0]
            print(f"\n--- Frame {frame_count} ---")
            for i, landmark in enumerate(hand.landmark):
                print(f"{LANDMARK_NAMES[i]:12s}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")

    #mirror frame so looks normal
    frame = cv2.flip(frame, 1)
    cv2.imshow("Hand Tracking", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()