# -----------------------------------------------------------------------------------------------------------------------
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
import time

# ==============================
# VOICE ENGINE 
# ==============================
speech_queue = queue.Queue()

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
    
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
        # Clear queue before adding new text
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except queue.Empty:
                break
        speech_queue.put(text)

# ==============================
# LOAD MODEL
# ==============================
print("🚀 Loading ASL model...")
try:
    model = load_model("asl_sign_model.h5")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ==============================
# MEDIAPIPE HAND LANDMARKER
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
    print("✅ Hand landmarker initialized")
except Exception as e:
    print(f"❌ Error loading hand landmarker: {e}")
    exit(1)

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_running = True
            threading.Thread(target=self.process, daemon=True).start()
            return True
        return False

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        return True

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

            # Draw hand landmarks with smaller, medium-sized circles
            if self.last_points:
                # Draw connections with thinner lines
                for s, e in connections:
                    cv2.line(frame, self.last_points[s], self.last_points[e], (0, 255, 255), 2)
                
                # Draw smaller, medium-sized circles for landmarks
                for p in self.last_points:
                    # Inner circle - smaller and solid
                    cv2.circle(frame, p, 4, (255, 255, 255), -1)
                    # Outer circle - thinner border
                    cv2.circle(frame, p, 6, (0, 255, 255), 1)

            if self.stable_label:
                # Add confidence bar
                bar_length = int(200 * self.confidence)
                cv2.rectangle(frame, (20, 420), (20 + bar_length, 440), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 420), (220, 440), (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {self.confidence:.2f}", (20, 415), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            with self.lock:
                self.current_frame = frame
                self.current_label = self.stable_label

recognizer = ASLRecognizer()

# ==============================
# DASH UI WITH MEDICAL COLORS
# ==============================
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
                ])
server = app.server#f5f7fb

# Custom CSS - SIRF COLORS CHANGE KIYE HAIN, LAYOUT SAME HAI
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SignSpeak AI - Real-time Sign Language Recognition</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            body {
        background: linear-gradient(135deg, #D4ADFC 0%, #FFD1DA 50%, #F3F6FF 100%);
                  /* Light medical background */
                min-height: 100vh;
            }
            
            /* Medical theme cards - white with subtle shadows */
            .medical-card {
                background: white;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.03);
                border: 1px solid #edf2f7;
                transition: all 0.3s ease;
            }
            
            .medical-card:hover {
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            }
            
            /* Status badge */
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 14px;
                background: #f0f9ff;
                border-radius: 40px;
                font-size: 14px;
                color: #0369a1;
                font-weight: 500;
            }
            
            .dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #94a3b8;
            }
            
            .dot.active {
                background: #10b981;
                animation: pulse 1.5s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            /* Button styles - medical blue */
            .btn-medical {
                padding: 12px 24px;
                border-radius: 40px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                border: none;
                transition: all 0.2s;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                justify-content: center;
            }
            
            .btn-primary {
                background: #4a90e2;  /* Medical blue */
                color: white;
                border: 1px solid #3a7bc8;
            }
            
            .btn-primary:hover:not(:disabled) {
                background: #3a7bc8;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(74,144,226,0.2);
            }
            
            .btn-secondary {
                background: white;
                color: #4a5568;
                border: 1px solid #e2e8f0;
            }
            
            .btn-secondary:hover:not(:disabled) {
                background: #f8fafd;
            }
            
            .btn-success {
                background: #10b981;  /* Medical green */
                color: white;
            }
            
            .btn-danger {
                background: #ef4444;  /* Medical red */
                color: white;
            }
            
            /* Confidence bar */
            .confidence-bar {
                width: 100%;
                height: 6px;
                background: #e2e8f0;
                border-radius: 3px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                background: #4a90e2;  /* Medical blue */
                border-radius: 3px;
                transition: width 0.3s ease;
            }
            
            /* Text colors */
            .text-medical-dark {
                color: #1a2b4c;  /* Dark blue-gray for headings */
            }
            
            .text-medical-light {
                color: #6b7a8f;  /* Light gray for secondary text */
            }
            
            /* Header styles */
            .header-title {
                font-size: 32px;
                font-weight: 700;
                color: #1a2b4c;
                margin-bottom: 8px;
            }
            
            .header-subtitle {
                font-size: 16px;
                color: #6b7a8f;
                margin-bottom: 32px;
            }
            
            /* Info cards */
            .info-card {
                background: #f8fafd;
                border-radius: 16px;
                padding: 16px;
                border: 1px solid #e2e8f0;
            }
            
            .info-label {
                font-size: 13px;
                color: #6b7a8f;
                margin-bottom: 8px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .info-value {
                font-size: 28px;
                font-weight: 700;
                color: #1a2b4c;
            }
            
            .info-value-small {
                font-size: 18px;
                font-weight: 600;
                color: #1a2b4c;
            }
            
            /* Video container */
            .video-container {
                background: white;
                border-radius: 20px;
                padding: 20px;
                border: 1px solid #edf2f7;
            }
            
            .video-box {
                background: #fafbfc;
                border-radius: 16px;
                overflow: hidden;
                aspect-ratio: 4/3;
                border: 1px solid #e2e8f0;
            }
            
            .video-box img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            /* Layout utilities */
            .btn-group {
                display: flex;
                gap: 12px;
                margin-top: 20px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                margin-top: 20px;
            }
            
            .sign-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin-top: 20px;
            }
            
            /* Footer */
            .medical-footer {
                background: white;
                border-top: 1px solid #edf2f7;
                margin-top: 40px;
                padding: 20px;
                text-align: center;
                color: #6b7a8f;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# LAYOUT BILKUL SAME HAI, SIRF CLASSES CHANGE KARDI HAIN
app.layout = html.Div([
    # Navbar
    html.Nav(className="bg-white border-b border-gray-200 fixed w-full z-50 top-0", children=[
        html.Div(className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8", children=[
            html.Div(className="flex justify-between items-center h-16", children=[
                html.Div(className="flex items-center space-x-4", children=[
                    html.I(className="fas fa-hands text-3xl", style={"color": "#4a90e2"}),
                    html.Span(className="text-medical-dark font-bold text-xl", children="SignSpeak AI"),
                ]),
                html.Div(className="flex items-center space-x-3", children=[
                    html.Span(className="text-medical-light text-sm hidden md:block", children="ASL Recognition"),
                    html.Div(className="w-2 h-2 bg-green-400 rounded-full animate-pulse")
                ])
            ])
        ])
    ]),

    # Main Content
    html.Div(className="pt-20 pb-8 px-4 max-w-7xl mx-auto", children=[
        # Header
        html.Div(className="text-center mb-8", children=[
            html.H1("Sign Language Recognition", 
                   className="header-title"),
            html.P("Seamless AI-Powered Sign Language Detection", 
                   className="header-subtitle")
        ]),

        # Main Grid
        html.Div(className="grid grid-cols-1 lg:grid-cols-3 gap-6", children=[
            # Video Feed (Left Column)
            html.Div(className="lg:col-span-2", children=[
                html.Div(className="medical-card", children=[
                    html.Div(className="relative", children=[
                        html.Img(id="video", 
                                className="w-full rounded-xl",
                                style={"aspectRatio": "4/3", "objectFit": "cover"}),
                        
                        # Camera Status Overlay
                        html.Div(id="camera-status", className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm text-medical-dark px-4 py-2 rounded-full flex items-center space-x-2 shadow-sm border border-gray-200", children=[
                            html.I(className="fas fa-circle text-green-400 text-xs"),
                            html.Span("Camera Active", className="text-sm font-medium")
                        ]),
                        
                        # Recognition Status
                        html.Div(className="absolute bottom-4 right-4 bg-white/90 backdrop-blur-sm text-medical-dark px-4 py-2 rounded-full shadow-sm border border-gray-200", children=[
                            html.I(className="fas fa-hand-paper mr-2", style={"color": "#4a90e2"}),
                            html.Span(id="hand-status", children="No Hand Detected")
                        ])
                    ])
                ])
            ]),

            # Controls Panel (Right Column)
            html.Div(className="space-y-6", children=[
                # Detected Sign Card
                html.Div(className="medical-card", children=[
                    html.Div(className="flex items-center justify-between mb-4", children=[
                        html.H3(className="text-medical-light text-sm uppercase tracking-wider", children="Current Sign"),
                        html.I(className="fas fa-arrow-right", style={"color": "#4a90e2"})
                    ]),
                    html.Div(id="sign", 
                            className="text-8xl font-bold text-medical-dark text-center my-4",
                            children="👋"),
                    
                    # Confidence Bar
                    html.Div(className="confidence-bar", children=[
                        html.Div(id="confidence-bar", className="confidence-fill", style={"width": "0%"})
                    ]),
                    html.Div(id="confidence", 
                            className="text-center text-medical-light text-sm mt-2",
                            children="Confidence: 0%")
                ]),

                # Sentence Formation Card
                html.Div(className="medical-card", children=[
                    html.Div(className="flex items-center justify-between mb-4", children=[
                        html.H3(className="text-medical-light text-sm uppercase tracking-wider", children="Sentence"),
                        html.I(className="fas fa-quote-right", style={"color": "#4a90e2"})
                    ]),
                    html.Div(id="sentence",
                            className="bg-gray-50 rounded-xl p-4 min-h-[80px] text-medical-dark text-lg border border-gray-200",
                            children="Start signing to form a sentence..."),
                    
                    # Sentence Actions
                    html.Div(className="btn-group", children=[
                        dbc.Button([
                            html.I(className="fas fa-volume-up mr-2"),
                            "Speak"
                        ], id="speak", n_clicks=0,
                        className="btn-medical btn-success flex-1"),
                        
                        dbc.Button([
                            html.I(className="fas fa-trash-alt mr-2"),
                            "Clear"
                        ], id="clear", n_clicks=0,
                        className="btn-medical btn-danger flex-1")
                    ])
                ]),

                # Camera Controls Card
                html.Div(className="medical-card", children=[
                    html.Div(className="flex items-center justify-between mb-4", children=[
                        html.H3(className="text-medical-light text-sm uppercase tracking-wider", children="Camera Controls"),
                        html.I(className="fas fa-camera", style={"color": "#4a90e2"})
                    ]),
                    
                    html.Div(className="btn-group", children=[
                        dbc.Button([
                            html.I(className="fas fa-play mr-2"),
                            "Start"
                        ], id="start", n_clicks=0,
                        className="btn-medical btn-primary flex-1"),
                        
                        dbc.Button([
                            html.I(className="fas fa-stop mr-2"),
                            "Stop"
                        ], id="stop", disabled=True, n_clicks=0,
                        className="btn-medical btn-secondary flex-1 opacity-50")
                    ]),
                    
                    # Quick Stats
                    html.Div(className="stats-grid", children=[
                        html.Div(className="info-card", children=[
                            html.Div("Words", className="info-label"),
                            html.Div(id="word-count", className="info-value-small", children="0")
                        ]),
                        html.Div(className="info-card", children=[
                            html.Div("Hand Status", className="info-label"),
                            html.Div(id="hand-status-small", className="info-value-small", children="—")
                        ]),
                    ])
                ]),

                # Tips Card
                html.Div(className="medical-card", children=[
                    html.Div(className="flex items-center space-x-2 mb-3", children=[
                        html.I(className="fas fa-lightbulb", style={"color": "#f59e0b"}),
                        html.H4(className="font-semibold text-medical-dark", children="Pro Tips")
                    ]),
                    html.Ul(className="space-y-2 text-medical-light text-sm", children=[
                        html.Li(className="flex items-start space-x-2", children=[
                            html.I(className="fas fa-check-circle text-xs mt-1", style={"color": "#10b981"}),
                            html.Span("Keep your hand clearly visible")
                        ]),
                        html.Li(className="flex items-start space-x-2", children=[
                            html.I(className="fas fa-check-circle text-xs mt-1", style={"color": "#10b981"}),
                            html.Span("Good lighting improves accuracy")
                        ]),
                        html.Li(className="flex items-start space-x-2", children=[
                            html.I(className="fas fa-check-circle text-xs mt-1", style={"color": "#10b981"}),
                            html.Span("Move hand slowly for better detection")
                        ])
                    ])
                ])
            ])
        ])
    ]),

    # Hidden Divs
    html.Div(id="speech-trigger", style={"display": "none"}, children="0"),
    dcc.Interval(id="interval", interval=40),

    # Footer
    html.Footer(className="medical-footer", children=[
        html.P(children=[
            "© SignSpeak AI - Sign Language Recognition | ",
            html.I(className="fas fa-heart", style={"color": "#ef4444"}),
            " Made for Humans"
        ])
    ])
])

# ==============================
# CALLBACKS (SAME RAHE)
# ==============================
@app.callback(
    [Output("start", "disabled"),
     Output("stop", "disabled"),
     Output("camera-status", "children")],
    [Input("start", "n_clicks"),
     Input("stop", "n_clicks")],
    prevent_initial_call=True
)
def control(start, stop):
    ctx = dash.callback_context.triggered[0]["prop_id"]
    if "start" in ctx:
        recognizer.start()
        return True, False, [
            html.I(className="fas fa-circle text-green-400 text-xs mr-2"),
            html.Span("Camera Active", className="text-sm font-medium")
        ]
    else:
        recognizer.stop()
        return False, True, [
            html.I(className="fas fa-circle text-gray-400 text-xs mr-2"),
            html.Span("Camera Off", className="text-sm font-medium")
        ]

@app.callback(
    [Output("video", "src"),
     Output("sign", "children"),
     Output("sentence", "children"),
     Output("hand-status", "children"),
     Output("hand-status-small", "children"),
     Output("confidence", "children"),
     Output("word-count", "children"),
     Output("confidence-bar", "style")],
    [Input("interval", "n_intervals"),
     Input("clear", "n_clicks")],
    prevent_initial_call=True
)
def update(n, clear):
    ctx = dash.callback_context.triggered[0]["prop_id"]

    if "clear" in ctx:
        recognizer.sentence = []
        recognizer.last_added = ""

    if not recognizer.is_running:
        return ("", "👋", "Start camera to begin signing...", 
                "No Hand Detected", "—", "Confidence: 0%", "0", {"width": "0%"})

    with recognizer.lock:
        frame = recognizer.current_frame
        label = recognizer.current_label
        sentence = "".join(recognizer.sentence)
        confidence = recognizer.confidence

    if frame is None:
        return ("", label if label else "👋", sentence or "Start signing...", 
                "No Hand Detected", "—", 
                f"Confidence: {confidence:.1%}" if confidence > 0 else "Confidence: 0%", 
                str(len(recognizer.sentence)), 
                {"width": f"{confidence*100}%"})

    hand_status = "✅ Hand Detected" if recognizer.hand_present else "❌ No Hand Detected"
    
    _, buffer = cv2.imencode(".jpg", frame)
    img = base64.b64encode(buffer).decode()
    
    sign_display = label if label else "👋"
    confidence_text = f"Confidence: {confidence:.1%}" if confidence > 0 else "Confidence: 0%"
    display_sentence = sentence if sentence else "Start signing to form a sentence..."
    
    return (f"data:image/jpeg;base64,{img}", 
            sign_display, 
            display_sentence,
            hand_status,
            "Active" if recognizer.hand_present else "Inactive",
            confidence_text,
            str(len(recognizer.sentence)),
            {"width": f"{confidence*100}%"})

@app.callback(
    Output("speech-trigger", "children"),
    Input("speak", "n_clicks"),
    prevent_initial_call=True
)
def speak_sentence(n_clicks):
    text = "".join(recognizer.sentence)
    if text and text.strip():
        speak_async(text)
    return str(n_clicks)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 SignSpeak Medical Edition")
    print("📱 http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(debug=True)
# -----------------------------------------------------------------------------------------------------------------------