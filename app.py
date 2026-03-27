import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ... [Keep your CSS section exactly as it is] ...

@st.cache_resource
def load_models():
    # Using your specific filenames
    return (
        joblib.load("model.pkl"), 
        joblib.load("label_encoder.pkl"),
        joblib.load("model_numbers_full.pkl"), 
        joblib.load("label_encoder_numbers_full.pkl"),
        joblib.load("model_words.pkl"), 
        joblib.load("label_encoder_words.pkl")
    )

# Load models once
try:
    alpha_model, alpha_encoder, num_model, num_encoder, word_model, word_encoder = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Model Load Error: {e}")

class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mode = "ALPHABET" # Default
        self.prev_pred = ""
        self.counter = 0
        self.prediction = "—"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        current_pred = ""

        if results.multi_hand_landmarks:
            all_landmarks = []
            hands_sorted = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)

            for hand_landmarks in hands_sorted:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    all_landmarks.extend([lm.x, lm.y, lm.z])

            try:
                # --- ALPHABET MODE ---
                if self.mode == "ALPHABET" and len(all_landmarks) >= 63:
                    data = np.array(all_landmarks[:63]).reshape(21, 3)
                    base = data[0]
                    norm = np.array([pt - base for pt in data]).flatten().reshape(1, -1)
                    current_pred = alpha_encoder.inverse_transform(alpha_model.predict(norm))[0]

                # --- NUMBER MODE ---
                elif self.mode == "NUMBER":
                    if len(results.multi_hand_landmarks) == 1:
                        handedness = results.multi_handedness[0].classification[0].label
                        data = np.array(all_landmarks[:63]).reshape(21, 3)
                        if handedness == "Left": data[:, 0] = 1 - data[:, 0]
                        combined = np.vstack((data, data)).flatten().reshape(1, -1)
                    else:
                        data = np.array(all_landmarks[:126]).reshape(42, 3)
                        base = data[0]
                        combined = np.array([pt - base for pt in data]).flatten().reshape(1, -1)
                    current_pred = num_encoder.inverse_transform(num_model.predict(combined))[0]

                # --- WORD MODE ---
                elif self.mode == "WORD" and len(all_landmarks) >= 63:
                    if len(all_landmarks) < 126: 
                        all_landmarks = all_landmarks + all_landmarks
                    data = np.array(all_landmarks[:126]).reshape(42, 3)
                    base = data[0]
                    norm = np.array([pt - base for pt in data]).flatten().reshape(1, -1)
                    current_pred = word_encoder.inverse_transform(word_model.predict(norm))[0]

            except Exception:
                pass

            # Stability logic
            if current_pred == self.prev_pred and current_pred != "":
                self.counter += 1
            else:
                self.counter = 0
            self.prev_pred = current_pred
            if self.counter >= 6:
                self.prediction = str(current_pred)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI LOGIC ---
mode_choice = st.selectbox("Select Mode", ["ALPHABET", "NUMBER", "WORD"])

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="gesture-bridge",
    video_processor_factory=GestureProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.mode = mode_choice
    st.markdown(f"""
        <div class="output-box">
            <div class="output-label">Detected Sign ({mode_choice})</div>
            <div class="output-value">{ctx.video_processor.prediction}</div>
        </div>
    """, unsafe_allow_html=True)