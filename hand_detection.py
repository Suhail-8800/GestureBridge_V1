import cv2
import mediapipe as mp
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# For controlled printing
last_print_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip image (mirror fix)
    img = cv2.flip(img, 1)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands.process(img_rgb)

    # If hands detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks (63 values)
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Print once per second (no spam)
            current_time = time.time()
            if len(landmark_list) == 63 and (current_time - last_print_time > 1):
                print("63 OK")
                last_print_time = current_time

    # Show output
    cv2.imshow("Hand Detection", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()