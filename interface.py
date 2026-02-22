import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import cv2
import mediapipe as mp
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import base64
import threading
import pyttsx3
import queue

# ==============================
# VOICE ENGINE (REAL FIX)
# ==============================
engine = pyttsx3.init(driverName='sapi5')
engine.setProperty('rate', 150)

speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text:
            engine.say(text)
            engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak_async(text):
    speech_queue.put(text)

# ==============================
# LOAD MODEL
# ==============================
print("Loading ASL model...")
model = load_model("asl_sign_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
print("Model loaded ✔")

# ==============================
# MEDIAPIPE HAND LANDMARKER
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ==============================
# AI ENGINE
# ==============================
class ASLRecognizer:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.current_label = ""
        self.lock = threading.Lock()

        self.timestamp = 0
        self.last_points = None

        # SPEED FIX
        self.pred_buffer = deque(maxlen=5)
        self.HOLD_FRAMES = 5

        self.hand_present = False
        self.hold_counter = 0
        self.stable_label = ""
        self.last_added = ""

        self.sentence = []

    def start(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            threading.Thread(target=self.process, daemon=True).start()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

    def process(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, self.timestamp)
            self.timestamp += 1

            if result.hand_landmarks:
                self.hand_present = True
                self.hold_counter += 1

                hand = result.hand_landmarks[0]
                h, w, _ = frame.shape

                features = []
                pts = []

                for lm in hand:
                    features.extend([lm.x, lm.y, lm.z])
                    pts.append((int(lm.x*w), int(lm.y*h)))

                self.last_points = pts

                X = scaler.transform(np.array(features).reshape(1, -1))
                preds = model.predict(X, verbose=0)
                pred_class = np.argmax(preds)

                self.pred_buffer.append(pred_class)

                if self.hold_counter >= self.HOLD_FRAMES:
                    smooth = max(set(self.pred_buffer), key=self.pred_buffer.count)
                    self.stable_label = encoder.inverse_transform([smooth])[0]

            else:
                if self.hand_present and self.stable_label:
                    if self.stable_label != self.last_added:
                        self.sentence.append(self.stable_label)
                        self.last_added = self.stable_label

                self.hand_present = False
                self.hold_counter = 0
                self.pred_buffer.clear()
                self.stable_label = ""
                self.last_points = None

            if self.last_points:
                for s, e in connections:
                    cv2.line(frame, self.last_points[s], self.last_points[e], (255,140,0), 3)
                for p in self.last_points:
                    cv2.circle(frame, p, 6, (0,220,255), -1)

            if self.stable_label:
                cv2.rectangle(frame, (0,0), (640,70), (0,0,0), -1)
                cv2.putText(frame, f"Sign: {self.stable_label}",
                            (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (255,255,255),
                            3)

            with self.lock:
                self.current_frame = frame
                self.current_label = self.stable_label

recognizer = ASLRecognizer()

# ==============================
# DASH UI
# ==============================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

glass = {
    "background":"rgba(255,255,255,0.05)",
    "backdropFilter":"blur(12px)",
    "borderRadius":"18px",
    "padding":"20px",
    "boxShadow":"0 8px 32px rgba(0,0,0,0.3)"
}

app.layout = dbc.Container([

    html.H1("🤟 AI Sign Language Recognition",
            className="text-center mt-4 mb-4",
            style={"fontWeight":"700","color":"#00e5ff"}),

    dbc.Row([

        dbc.Col([
            html.Div([
                html.Img(id="video",
                         style={"width":"100%","borderRadius":"15px"})
            ], style=glass)
        ], width=7),

        dbc.Col([

            html.Div([
                html.H4("Detected Sign"),
                html.Div(id="sign",
                         style={"fontSize":"70px",
                                "textAlign":"center",
                                "color":"#00e5ff",
                                "fontWeight":"bold"})
            ], style=glass),

            html.Br(),

            html.Div([
                html.H4("Sentence Formation"),
                html.Div(id="sentence",
                         style={"fontSize":"28px","minHeight":"60px"})
            ], style=glass),

            html.Br(),

            dbc.Button("🔊 Speak Sentence", id="speak", color="success", className="me-2"),
            dbc.Button("🧹 Clear Sentence", id="clear", color="danger"),

            html.Br(), html.Br(),

            dbc.ButtonGroup([
                dbc.Button("▶ Start Camera", id="start"),
                dbc.Button("⏹ Stop Camera", id="stop", disabled=True)
            ])

        ], width=5)

    ]),

    dcc.Interval(id="interval", interval=40)  # faster UI refresh

], fluid=True)

# ==============================
# CAMERA CONTROL
# ==============================
@app.callback(
    Output("start","disabled"),
    Output("stop","disabled"),
    Input("start","n_clicks"),
    Input("stop","n_clicks"),
    prevent_initial_call=True
)
def control(start, stop):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    if "start" in ctx:
        recognizer.start()
        return True, False
    else:
        recognizer.stop()
        return False, True

# ==============================
# VIDEO + TEXT UPDATE
# ==============================
@app.callback(
    Output("video","src"),
    Output("sign","children"),
    Output("sentence","children"),
    Input("interval","n_intervals"),
    Input("clear","n_clicks"),
    prevent_initial_call=True
)
def update(n, clear):
    ctx = dash.callback_context.triggered[0]["prop_id"]

    if "clear" in ctx:
        recognizer.sentence = []
        recognizer.last_added = ""

    if not recognizer.is_running:
        return "", "", ""

    with recognizer.lock:
        frame = recognizer.current_frame
        label = recognizer.current_label
        sentence = " ".join(recognizer.sentence)

    if frame is None:
        return "", "", sentence

    _, buffer = cv2.imencode(".jpg", frame)
    img = base64.b64encode(buffer).decode()

    return f"data:image/jpeg;base64,{img}", label, sentence

# ==============================
# SPEAK BUTTON
# ==============================
@app.callback(
    Output("sign","style"),
    Input("speak","n_clicks"),
    prevent_initial_call=True
)
def speak_sentence(n):
    text = " ".join(recognizer.sentence)
    if text:
        speak_async(text)
    return {"fontSize":"70px","textAlign":"center","color":"#00e5ff","fontWeight":"bold"}

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    print("Open http://127.0.0.1:8050")
    app.run(debug=True)