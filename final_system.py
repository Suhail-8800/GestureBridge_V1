import cv2
import mediapipe as mp
import numpy as np
import joblib

# ========== LOAD MODELS ==========
alpha_model = joblib.load("model.pkl")
alpha_encoder = joblib.load("label_encoder.pkl")

num_model = joblib.load("model_numbers_full.pkl")
num_encoder = joblib.load("label_encoder_numbers_full.pkl")

word_model = joblib.load("model_words.pkl")
word_encoder = joblib.load("label_encoder_words.pkl")

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ========== MODE ==========
mode = "ALPHABET"

# Stability
prev_prediction = ""
counter = 0
prediction = ""

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)

print("Press 'M' → Toggle Alphabet/Number")
print("Press 'W' → Word Mode")
print("Press 'Q' → Quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    key = cv2.waitKey(1) & 0xFF

    # MODE SWITCH
    if key == ord('m'):
        if mode == "ALPHABET":
            mode = "NUMBER"
        else:
            mode = "ALPHABET"
        print(f"Switched to {mode} mode")

    if key == ord('w'):
        mode = "WORD"
        print("Switched to WORD mode")

    if results.multi_hand_landmarks:

        all_landmarks = []

        # SORT HANDS (CRITICAL)
        hands_sorted = sorted(
            results.multi_hand_landmarks,
            key=lambda h: h.landmark[0].x
        )

        for hand_landmarks in hands_sorted:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                all_landmarks.extend([lm.x, lm.y, lm.z])

        # ==========================
        # ALPHABET MODE
        # ==========================
        if mode == "ALPHABET" and len(all_landmarks) >= 63:

            data = np.array(all_landmarks[:63]).reshape(21, 3)

            base_x, base_y, base_z = data[0]

            normalized = []
            for x, y_val, z in data:
                normalized.extend([x - base_x, y_val - base_y, z - base_z])

            normalized = np.array(normalized).reshape(1, -1)

            pred = alpha_model.predict(normalized)
            current_pred = alpha_encoder.inverse_transform(pred)[0]

        # ==========================
        # NUMBER MODE
        # ==========================
        elif mode == "NUMBER":

            # ONE HAND (0–5)
            if len(results.multi_hand_landmarks) == 1:

                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label

                single_hand = []

                for lm in hand_landmarks.landmark:
                    x, y, z = lm.x, lm.y, lm.z

                    # Convert LEFT → RIGHT format
                    if handedness == "Left":
                        x = 1 - x

                    single_hand.extend([x, y, z])

                combined = single_hand + single_hand

                data = np.array(combined).reshape(42, 3)

                base_x, base_y, base_z = data[0]
                normalized = []
                for x, y_val, z in data:
                    normalized.extend([x - base_x, y_val - base_y, z - base_z])

                normalized = np.array(normalized).reshape(1, -1)

                pred = num_model.predict(normalized)
                current_pred = num_encoder.inverse_transform(pred)[0]

            # TWO HANDS (6–9)
            elif len(all_landmarks) == 126:

                data1 = np.array(all_landmarks).reshape(42, 3)

                base_x, base_y, base_z = data1[0]
                norm1 = []
                for x, y_val, z in data1:
                    norm1.extend([x - base_x, y_val - base_y, z - base_z])
                norm1 = np.array(norm1).reshape(1, -1)

                pred1 = num_model.predict(norm1)
                pred1_label = num_encoder.inverse_transform(pred1)[0]

                hand1 = all_landmarks[:63]
                hand2 = all_landmarks[63:]

                reversed_combined = hand2 + hand1

                data2 = np.array(reversed_combined).reshape(42, 3)

                base_x, base_y, base_z = data2[0]
                norm2 = []
                for x, y_val, z in data2:
                    norm2.extend([x - base_x, y_val - base_y, z - base_z])
                norm2 = np.array(norm2).reshape(1, -1)

                pred2 = num_model.predict(norm2)
                pred2_label = num_encoder.inverse_transform(pred2)[0]

                # FINAL DECISION
                if pred1_label == pred2_label:
                    current_pred = pred1_label
                else:
                    current_pred = max(pred1_label, pred2_label)

            else:
                current_pred = ""

        # ==========================
        # WORD MODE
        # ==========================
        elif mode == "WORD":

            if len(all_landmarks) == 126:
                combined = all_landmarks
            elif len(all_landmarks) == 63:
                combined = all_landmarks + all_landmarks
            else:
                combined = None

            if combined is not None:

                data = np.array(combined).reshape(42, 3)

                base_x, base_y, base_z = data[0]

                normalized = []
                for x, y_val, z in data:
                    normalized.extend([x - base_x, y_val - base_y, z - base_z])

                normalized = np.array(normalized).reshape(1, -1)

                pred = word_model.predict(normalized)
                current_pred = word_encoder.inverse_transform(pred)[0]

            else:
                current_pred = ""

        else:
            current_pred = ""

        # STABILITY
        if current_pred == prev_prediction and current_pred != "":
            counter += 1
        else:
            counter = 0

        prev_prediction = current_pred

        if counter >= 6:
            prediction = current_pred

    # DISPLAY
    cv2.putText(img, f"Mode: {mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(img, f"Output: {prediction}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("GestureBridge System", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()