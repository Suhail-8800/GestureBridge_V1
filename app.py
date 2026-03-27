import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="GestureBridge",
    page_icon="🤟",
    layout="centered"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

.stApp {
    background: #0d0d0d;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

.title-block {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

.title-block h1 {
    font-size: 3rem;
    color: #00ff99;
    margin-bottom: 0.2rem;
}

.title-block p {
    color: #888;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.output-box {
    background: #1a1a1a;
    border: 2px solid #00ff99;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1.5rem 0;
}

.output-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
}

.output-value {
    font-size: 4rem;
    font-weight: 800;
    color: #00ff99;
    line-height: 1;
    text-shadow: 0 0 30px rgba(0,255,153,0.4);
}

.mode-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 999px;
    padding: 0.25rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00ff99;
    letter-spacing: 2px;
    text-transform: uppercase;
}

div[data-testid="stSelectbox"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

div[data-testid="stSelectbox"] > div > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
}

.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}

div[data-testid="column"]:nth-child(1) .stButton > button {
    background: #00ff99 !important;
    color: #0d0d0d !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button:hover {
    background: #00cc7a !important;
    transform: translateY(-1px);
}

div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #1a1a1a !important;
    color: #ff4444 !important;
    border: 1px solid #ff4444 !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #2a1010 !important;
}

.instructions {
    background: #111;
    border-left: 3px solid #00ff99;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    line-height: 1.8;
}

.stAlert {
    border-radius: 8px !important;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    alpha_model   = joblib.load("model.pkl")
    alpha_encoder = joblib.load("label_encoder.pkl")
    num_model     = joblib.load("model_numbers_full.pkl")
    num_encoder   = joblib.load("label_encoder_numbers_full.pkl")
    word_model    = joblib.load("model_words.pkl")
    word_encoder  = joblib.load("label_encoder_words.pkl")
    return alpha_model, alpha_encoder, num_model, num_encoder, word_model, word_encoder

try:
    alpha_model, alpha_encoder, num_model, num_encoder, word_model, word_encoder = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "prediction" not in st.session_state:
    st.session_state.prediction = "—"
if "mode" not in st.session_state:
    st.session_state.mode = "ALPHABET"


# ─────────────────────────────────────────
# VIDEO PROCESSOR
# ─────────────────────────────────────────
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands  = mp.solutions.hands
        self.hands     = self.mp_hands.Hands(max_num_hands=2)
        self.mp_draw   = mp.solutions.drawing_utils
        self.prev_pred = ""
        self.counter   = 0
        self.prediction = "—"
        self.mode      = "ALPHABET"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        current_pred = ""

        if results.multi_hand_landmarks:
            all_landmarks = []

            hands_sorted = sorted(
                results.multi_hand_landmarks,
                key=lambda h: h.landmark[0].x
            )

            for hand_landmarks in hands_sorted:
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 153), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2)
                )
                for lm in hand_landmarks.landmark:
                    all_landmarks.extend([lm.x, lm.y, lm.z])

            # ── ALPHABET ──
            if self.mode == "ALPHABET" and len(all_landmarks) >= 63:
                data = np.array(all_landmarks[:63]).reshape(21, 3)
                base_x, base_y, base_z = data[0]
                normalized = []
                for x, y_val, z in data:
                    normalized.extend([x - base_x, y_val - base_y, z - base_z])
                normalized = np.array(normalized).reshape(1, -1)
                pred = alpha_model.predict(normalized)
                current_pred = alpha_encoder.inverse_transform(pred)[0]

            # ── NUMBER ──
            elif self.mode == "NUMBER":
                if len(results.multi_hand_landmarks) == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0].classification[0].label
                    single_hand = []
                    for lm in hand_landmarks.landmark:
                        x, y, z = lm.x, lm.y, lm.z
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

                    if pred1_label == pred2_label:
                        current_pred = pred1_label
                    else:
                        current_pred = max(pred1_label, pred2_label)

            # ── WORD ──
            elif self.mode == "WORD":
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

            # Stability filter
            if current_pred == self.prev_pred and current_pred != "":
                self.counter += 1
            else:
                self.counter = 0
            self.prev_pred = current_pred

            if self.counter >= 6:
                self.prediction = current_pred

        # Overlay on frame
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), (13, 13, 13), -1)
        cv2.putText(img, f"Mode: {self.mode}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 153), 2)
        cv2.putText(img, f"Output: {self.prediction}", (15, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 153), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🤟 GestureBridge</h1>
    <p>Real-time Sign Language Recognition System</p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load model files. Make sure all `.pkl` files are in the same folder as `app.py`.\n\nError: `{model_error}`")
    st.stop()

# Mode selector
mode_choice = st.selectbox(
    "Recognition Mode",
    options=["ALPHABET", "NUMBER", "WORD"],
    index=0,
    help="Select what you want to recognise"
)

# Camera controls
col1, col2 = st.columns(2)
with col1:
    start = st.button("▶  Start Camera", use_container_width=True)
with col2:
    stop = st.button("■  Stop Camera", use_container_width=True)

if start:
    st.session_state.camera_on = True
if stop:
    st.session_state.camera_on = False

camera_on = st.session_state.get("camera_on", False)

# WebRTC streamer
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if camera_on:
    ctx = webrtc_streamer(
        key="gesturebridge",
        video_processor_factory=GestureProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.mode = mode_choice

        # Live prediction display
        st.markdown(f"""
        <div class="output-box">
            <div class="output-label">Detected Sign</div>
            <div class="output-value">{ctx.video_processor.prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="text-align:center"><span class="mode-badge">{mode_choice}</span></div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="output-box" style="border-color:#333;">
        <div class="output-label">Camera is off</div>
        <div class="output-value" style="color:#333;">—</div>
    </div>
    """, unsafe_allow_html=True)

# Instructions
st.markdown("""
<div class="instructions">
    <strong style="color:#f0f0f0;">HOW TO USE</strong><br>
    1. Select a mode — Alphabet, Number, or Word<br>
    2. Click <strong style="color:#00ff99;">Start Camera</strong><br>
    3. Show your hand sign in front of the camera<br>
    4. Hold the sign steady — result appears after a moment<br>
    5. Click <strong style="color:#ff4444;">Stop Camera</strong> when done
</div>
""", unsafe_allow_html=True)