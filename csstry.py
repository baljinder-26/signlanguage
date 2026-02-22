import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import cv2
import mediapipe as mp
import joblib
import numpy as np
import base64
import threading
import pyttsx3
import queue
import time
from tensorflow.keras.models import load_model
from collections import deque

# ==============================
# VOICE ENGINE (Multi-threaded)
# ==============================
speech_queue = queue.Queue()

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    while True:
        try:
            text = speech_queue.get(timeout=1)
            if text and text.strip():
                engine.stop()
                engine.say(str(text))
                engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Speech error: {e}")

threading.Thread(target=speech_worker, daemon=True).start()

def speak_async(text):
    if text and text.strip():
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except queue.Empty:
                break
        speech_queue.put(text)

# ==============================
# LOAD ASL MODELS
# ==============================
print("🚀 Loading ASL Engine...")
try:
    model = load_model("asl_sign_model.h5")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    print("✅ AI Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    exit(1)

# ==============================
# MEDIAPIPE INITIALIZATION
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
    landmarker = HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"❌ Hand Landmarker Error: {e}")
    exit(1)

connections = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ==============================
# AI RECOGNITION ENGINE
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
        self.pred_buffer = deque(maxlen=5)
        self.HOLD_FRAMES = 5
        self.hand_present = False
        self.hold_counter = 0
        self.stable_label = ""
        self.last_added = ""
        self.sentence = []
        self.confidence = 0.0

    def start(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            threading.Thread(target=self.process, daemon=True).start()
            return True
        return False

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        return True

    def process(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
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
                self.confidence = float(preds[0][pred_class])
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
                self.confidence = 0.0

            if self.last_points:
                for s, e in connections:
                    cv2.line(frame, self.last_points[s], self.last_points[e], (139, 92, 246), 2)
                for p in self.last_points:
                    cv2.circle(frame, p, 4, (255, 255, 255), -1)

            with self.lock:
                self.current_frame = frame
                self.current_label = self.stable_label

recognizer = ASLRecognizer()

# ==============================
# DASH UI
# ==============================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SignSpeak AI</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
            * { font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background: linear-gradient(135deg, #D4ADFC 0%, #FFD1DA 50%, #F3F6FF 100%); background-attachment: fixed; min-height: 100vh; }
            .glass-nav { background: rgba(17, 24, 39, 0.8) !important; backdrop-filter: blur(12px); border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
            .ai-card { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 24px; border: 1px solid rgba(255, 255, 255, 0.5); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04); }
            .btn-primary-ai { background: #FF6000; color: white; border: none; border-radius: 12px; font-weight: 600; padding: 10px 20px; transition: 0.3s; }
            .btn-primary-ai:hover { background: #E55600; transform: scale(1.05); }
            .confidence-fill-ai { height: 100%; background: linear-gradient(90deg, #6366f1, #a855f7); border-radius: 10px; transition: width 0.5s ease; }
            .video-box-ai { border-radius: 20px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); background: #1a1a2e; min-height: 400px; display: flex; align-items: center; justify-content: center; }
            .standby-overlay { text-align: center; color: #6366f1; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

app.layout = html.Div([
    # Navbar with NEW RIGHT LINKS
    html.Nav(className="glass-nav fixed w-full z-50 top-0 py-3", children=[
        html.Div(className="max-w-7xl mx-auto px-6 flex justify-between items-center", children=[
            html.Div(className="flex items-center space-x-3", children=[
                html.Div(className="p-2 bg-indigo-500 rounded-lg", children=[html.I(className="fas fa-hand-sparkles text-white text-xl")]),
                html.Span("SignSpeak AI", className="text-white font-extrabold text-2xl tracking-tight"),
            ]),
            # NEW RIGHT LINKS
            html.Div(className="flex items-center space-x-6", children=[
                html.A("Settings", href="#", className="text-gray-300 hover:text-white text-sm font-semibold transition-colors"),
                html.A("About", href="#", className="text-gray-300 hover:text-white text-sm font-semibold transition-colors"),
                html.Div(className="w-8 h-8 rounded-full bg-gradient-to-tr from-orange-400 to-pink-500 border border-white/20")
            ])
        ])
    ]),

    html.Div(className="pt-32 pb-12 px-6 max-w-7xl mx-auto", children=[
        html.Div(className="text-center mb-12", children=[
            html.H1("Break the Silence with AI", className="text-5xl font-black text-slate-900 mb-4"),
            html.P("Translating ASL into speech and text in real-time.", className="text-lg text-slate-600")
        ]),

        html.Div(className="grid grid-cols-1 lg:grid-cols-12 gap-8", children=[
            # Video Container
            html.Div(className="lg:col-span-8", children=[
                html.Div(className="ai-card p-4", children=[
                    html.Div(id="video-container", className="video-box-ai relative", children=[
                        # This displays when camera is off
                        html.Div(id="standby-ui", className="standby-overlay", children=[
                            html.I(className="fas fa-video-slash text-6xl mb-4 opacity-20"),
                            html.P("SYSTEM STANDBY", className="font-bold tracking-widest opacity-40")
                        ]),
                        html.Img(id="video", className="w-full hidden"), # Hidden by default
                        html.Div(id="camera-status", className="absolute top-4 left-4 bg-black/50 backdrop-blur-md text-white px-4 py-1.5 rounded-full text-xs font-bold", children="CAMERA OFF")
                    ]),
                    html.Div(className="flex gap-3 mt-6", children=[
                        dbc.Button("Start Engine", id="start", className="btn-primary-ai"),
                        dbc.Button("Stop", id="stop", disabled=True, className="btn-primary-ai bg-slate-800")
                    ])
                ])
            ]),

            # Analysis Panel
            html.Div(className="lg:col-span-4 space-y-6", children=[
                html.Div(className="ai-card p-6", children=[
                    html.H3("DETECTED SIGN", className="text-xs font-bold text-indigo-600 tracking-widest mb-4"),
                    html.Div(id="sign", className="text-7xl font-black text-slate-800 text-center py-4", children="—"),
                    html.Div(className="mt-4", children=[
                        html.Div(className="flex justify-between text-xs font-bold mb-2", children=[
                            html.Span("CONFIDENCE"), html.Span(id="confidence", children="0%")
                        ]),
                        html.Div(className="h-2 w-full bg-slate-100 rounded-full overflow-hidden", children=[
                            html.Div(id="confidence-bar", className="confidence-fill-ai", style={"width": "0%"})
                        ])
                    ])
                ]),

                html.Div(className="ai-card p-6", children=[
                    html.H3("TRANSCRIPT", className="text-xs font-bold text-indigo-600 tracking-widest mb-4"),
                    html.Div(id="sentence", className="bg-slate-50 rounded-2xl p-4 min-h-[100px] text-slate-700 italic", children="Transcription..."),
                    html.Div(className="grid grid-cols-2 gap-3 mt-4", children=[
                        dbc.Button("Speak", id="speak", className="btn-primary-ai w-full"),
                        dbc.Button("Clear", id="clear", className="btn-primary-ai bg-white text-slate-600 border w-full")
                    ])
                ])
            ])
        ])
    ]),
    dcc.Interval(id="interval", interval=50),
    html.Div(id="speech-trigger", style={"display": "none"})
])

# ==============================
# CALLBACKS
# ==============================
@app.callback(
    [Output("start", "disabled"), Output("stop", "disabled"), Output("camera-status", "children")],
    [Input("start", "n_clicks"), Input("stop", "n_clicks")],
    prevent_initial_call=True
)
def control_camera(start, stop):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    if "start" in ctx:
        recognizer.start()
        return True, False, "ENGINE ACTIVE"
    else:
        recognizer.stop()
        return False, True, "CAMERA OFF"

@app.callback(
    [Output("video", "src"), Output("sign", "children"), Output("sentence", "children"),
     Output("confidence", "children"), Output("confidence-bar", "style"),
     Output("video", "className"), Output("standby-ui", "className")],
    [Input("interval", "n_intervals"), Input("clear", "n_clicks")]
)
def update_dashboard(n, clear):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    if "clear" in ctx:
        recognizer.sentence = []
    
    if not recognizer.is_running:
        # Hide video and show Standby UI when stopped
        return "", "—", "Ready...", "0%", {"width": "0%"}, "hidden", "standby-overlay"

    with recognizer.lock:
        frame = recognizer.current_frame
        label = recognizer.current_label
        conf = recognizer.confidence
        text = " ".join(recognizer.sentence)

    if frame is None: return dash.no_update

    _, buffer = cv2.imencode(".jpg", frame)
    img = base64.b64encode(buffer).decode()
    
    # Show video and hide Standby UI when running
    return (f"data:image/jpeg;base64,{img}", 
            label if label else "—", 
            text if text else "Waiting for input...",
            f"{conf:.0%}", 
            {"width": f"{conf*100}%"},
            "w-full block", # video class
            "hidden")      # standby ui class

@app.callback(
    Output("speech-trigger", "children"),
    Input("speak", "n_clicks"),
    prevent_initial_call=True
)
def handle_speech(n):
    full_text = " ".join(recognizer.sentence)
    if full_text: speak_async(full_text)
    return ""

if __name__ == "__main__":
    app.run(debug=True)