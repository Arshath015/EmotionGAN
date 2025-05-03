import numpy as np
import requests
import time
import google.generativeai as genai
from collections import Counter
from scipy.stats import entropy
import cv2
import base64
from flask import Flask, render_template_string, request, jsonify, session
import threading
from deepface import DeepFace
import urllib.request
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

genai.configure(api_key="AIzaSyCIltxS9C-feeMo63TUy6CPaCZtCEn4aPo")
model = genai.GenerativeModel("gemini-1.5-flash")

# GIPHY API Key
GIPHY_API_KEY = "LdXitdkaX9wl5dGEZIxolYEcWkD6j1qH"  # Replace with your actual key

# Global Variables
face_emotions = []
face_confidence_scores = []
face_capture_active = False  # Start as False
face_data_ready = False
capture_thread = None

def download_giphy_gif(emotion):
    """Download a GIF from GIPHY based on the detected emotion"""
    try:
        # Make the API request
        response = requests.get(
            f"https://api.giphy.com/v1/gifs/search?api_key={GIPHY_API_KEY}&q={emotion}&limit=1"
        )
        
        if response.status_code == 200:
            gif_url = response.json()["data"][0]["images"]["original"]["url"]
            static_folder = os.path.join(app.root_path, 'static/gifs')
            if not os.path.exists(static_folder):
                os.makedirs(static_folder)
            filename = os.path.join(static_folder, f"{emotion}.gif")
            
            # Download the GIF
            urllib.request.urlretrieve(gif_url, filename)
            return f"/static/gifs/{emotion}.gif"
    except Exception as e:
        print(f"Error downloading GIF: {e}")
    return None

# Function to capture face emotions
def capture_face_emotion():
    global face_emotions, face_confidence_scores, face_capture_active, face_data_ready

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(5):  
        if cap.isOpened():
            break
        print("Retrying to open the camera...")
        time.sleep(1)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera after multiple attempts.")
        face_capture_active = False
        return
    
    confidence_scores = []
    emotion_counts = Counter()
    total_detections = 0

    while face_capture_active:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame.")
            time.sleep(1)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            detected_emotions = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
            print("Detected Emotions:", detected_emotions)

            if detected_emotions and isinstance(detected_emotions, list):
                emotions = detected_emotions[0].get("emotion", {})
                if emotions:
                    max_emotion = max(emotions, key=emotions.get)
                    emotion_counts[max_emotion] += 1
                    confidence_scores.append(emotions[max_emotion])
                    total_detections += 1
                    face_data_ready = True
        except Exception as e:
            print("DeepFace error:", e)

        time.sleep(0.5)

    cap.release()
    face_capture_active = True

    if total_detections > 0:
        face_emotions.extend([emotion for emotion, count in emotion_counts.items() for _ in range(count)])
        face_confidence_scores.extend(confidence_scores)
        
# Function to analyze text emotion
def analyze_text_emotion(text):
    prompt = (f"Analyze the emotion in this sentence: '{text}' "
              "Provide only the dominant emotion (e.g., happy, sad, anger, fear, surprise, neutral) "
              "and a confidence score from 0-100. Format response as: 'Emotion: <emotion>, Confidence: <score>'")
    
    response = model.generate_content(prompt)
    result = response.text
    try:
        emotion, confidence = result.split(', ')
        emotion = emotion.split(': ')[1]
        confidence = float(confidence.split(': ')[1])
        return emotion, confidence
    except:
        return "neutral", 50.0

# Function to compute AEWDO metrics
def compute_metrics(emotion_list, confidence_scores):
    if not emotion_list:
        return "neutral", 50.0, 0.0, 0.0, 50.0

    emotion_counts = Counter(emotion_list)
    total_detections = sum(emotion_counts.values())
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 50.0
    most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]
    consistency_rate = most_common_count / total_detections if total_detections > 0 else 0.0
    emotion_probabilities = np.array(list(emotion_counts.values()))
    avg_entropy = entropy(emotion_probabilities) if emotion_probabilities.size > 1 else 0.0
    trustworthiness_score = (avg_confidence / 100 * 50) + (consistency_rate * 30) - (avg_entropy * 20)
    trustworthiness_score = max(0, min(trustworthiness_score, 100))
    return most_common_emotion, avg_confidence, consistency_rate, avg_entropy, trustworthiness_score

@app.route('/')
def index():
    # Clear previous session data
    session.clear()
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Fusion Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #6c63ff;
            --secondary-color: #4d44db;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            --success-color: #28a745;
            --info-color: #17a2b8;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
        }
        
        body {
            background-color: #f5f7ff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            border: none;
            transition: transform 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
            padding: 10px 25px;
            font-weight: 500;
        }
        
        .btn-primary:hover {
            background-color: var(--secondary-color);
            border-color: var(--secondary-color);
        }
        
        .emotion-display {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .happy { color: var(--success-color); }
        .sad { color: var(--info-color); }
        .anger { color: var(--danger-color); }
        .fear { color: var(--warning-color); }
        .surprise { color: #ff6b6b; }
        .neutral { color: var(--dark-color); }
        
        .gif-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .gif-container img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            max-height: 200px;
        }
        
        .gif-emotion-text {
            font-size: 1.2rem;
            margin-top: 10px;
            font-weight: bold;
        }
        
        .metric-card {
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .confidence-container {
            position: relative;
            height: 20px;
            margin-bottom: 30px;
        }
        
        .confidence-meter {
            height: 10px;
            background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
            border-radius: 5px;
            margin-top: 10px;
        }
        
        .confidence-indicator {
            height: 20px;
            width: 20px;
            background-color: white;
            border: 3px solid var(--primary-color);
            border-radius: 50%;
            position: absolute;
            top: -5px;
            transform: translateX(-50%);
        }
        
        .tab-content {
            padding: 20px 0;
        }
        
        .nav-tabs .nav-link {
            color: var(--dark-color);
            font-weight: 500;
            border: none;
            padding: 10px 20px;
        }
        
        .nav-tabs .nav-link.active {
            color: var(--primary-color);
            font-weight: 600;
            border-bottom: 3px solid var(--primary-color);
            background-color: transparent;
        }
        
        .result-section {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card p-4 mb-4">
                    <h1 class="text-center mb-4"><i class="fas fa-brain me-2"></i> Emotion Fusion Analyzer</h1>
                    <p class="text-center text-muted mb-4">Combining facial expression analysis and text sentiment for accurate emotion detection</p>
                    
                    <form id="analysisForm">
                        <div class="mb-3">
                            <label for="textInput" class="form-label">Enter your text for analysis:</label>
                            <textarea class="form-control" id="textInput" name="text_input" rows="3" placeholder="How are you feeling today?"></textarea>
                        </div>
                        <div class="d-grid">
                            <button type="submit" class="btn btn-primary btn-lg" id="analyzeBtn">
                                <i class="fas fa-chart-line me-2"></i> Analyze Emotion
                            </button>
                        </div>
                    </form>
                </div>
                
                <div class="result-section" id="resultSection">
                    <div class="card p-4 mb-4">
                        <h2 class="text-center mb-4">Analysis Results</h2>
                        
                        <div class="text-center mb-5">
                            <h4 class="mb-3">Final Emotion Detection</h4>
                            <div class="emotion-display" id="finalEmotion">neutral</div>
                            
                            <div class="gif-container" id="gifContainer">
                                <!-- GIF will be inserted here -->
                            </div>
                            
                            <div class="row mt-4">
                                <div class="col-md-6">
                                    <div class="metric-card">
                                        <div class="metric-value" id="trustScore">92%</div>
                                        <div class="metric-label">Trust Score</div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="metric-card">
                                        <div class="metric-value" id="confidenceScore">65%</div>
                                        <div class="metric-label">Confidence Level</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <ul class="nav nav-tabs" id="analysisTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="face-tab" data-bs-toggle="tab" data-bs-target="#face" type="button" role="tab">Facial Analysis</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="text-tab" data-bs-toggle="tab" data-bs-target="#text" type="button" role="tab">Text Analysis</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="fusion-tab" data-bs-toggle="tab" data-bs-target="#fusion" type="button" role="tab">Fusion Details</button>
                            </li>
                        </ul>
                        
                        <div class="tab-content" id="analysisTabsContent">
                            <div class="tab-pane fade show active" id="face" role="tabpanel">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="faceEmotion">sad</div>
                                            <div class="metric-label">Dominant Emotion</div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="faceConfidence">87%</div>
                                            <div class="metric-label">Confidence</div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="faceTrust">42%</div>
                                            <div class="metric-label">Trust Score</div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="faceEntropy">0.94</div>
                                            <div class="metric-label">Emotion Entropy</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="tab-pane fade" id="text" role="tabpanel">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="textEmotion">anger</div>
                                            <div class="metric-label">Detected Emotion</div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="textConfidence">65%</div>
                                            <div class="metric-label">Confidence</div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="textTrust">75%</div>
                                            <div class="metric-label">Trust Score</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="tab-pane fade" id="fusion" role="tabpanel">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="faceWeight">58%</div>
                                            <div class="metric-label">Face Weight</div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="metric-card">
                                            <div class="metric-value" id="textWeight">42%</div>
                                            <div class="metric-label">Text Weight</div>
                                        </div>
                                    </div>
                                    <div class="col-12">
                                        <div class="card p-3 mt-3">
                                            <h5 class="mb-3">Fusion Algorithm</h5>
                                            <p>The final emotion is determined by combining facial expression analysis and text sentiment, weighted by their respective trust scores.</p>
                                            <p class="mb-2"><strong>Formula:</strong></p>
                                            <p class="text-muted">Final Emotion = (W_face × Face_Emotion) + (W_text × Text_Emotion)</p>
                                            <p class="text-muted">Where weights are based on trustworthiness scores from each modality.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    
    <script>
        $(document).ready(function() {
    // Variable to track if we've already started face capture
    let faceCaptureStarted = false;
    
    // Start face capture when user types
    $('#textInput').on('input', function() {
        if ($(this).val().trim().length > 0 && !faceCaptureStarted) {
            faceCaptureStarted = true;
            $.ajax({
                url: '/start_face_capture',
                method: 'POST',
                success: function(response) {
                    console.log(response.status);
                },
                error: function() {
                    console.log('Error starting face capture');
                }
            });
        }
    });
            
            $('#analysisForm').submit(function(e) {
                e.preventDefault();
                const formData = $(this).serialize();
                
                $('#analyzeBtn').html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...');
                $('#analyzeBtn').prop('disabled', true);
                
                $.ajax({
                    url: '/analyze',
                    method: 'POST',
                    data: formData,
                    success: function(response) {
                        updateUI(response);
                        $('#resultSection').fadeIn();
                        
                        // Get and display GIF for the detected emotion
                        $.get('/get_gif?emotion=' + response.final_emotion, function(gifPath) {
                            if(gifPath) {
                                const gifHtml = `
                                    <img src="${gifPath}" alt="${response.final_emotion} GIF">
                                    <div class="gif-emotion-text ${response.final_emotion.toLowerCase()}">
                                        ${response.final_emotion} GIF
                                    </div>
                                `;
                                $('#gifContainer').html(gifHtml);
                            }
                        });
                    },
                    error: function() {
                        alert('Error occurred during analysis');
                    },
                    complete: function() {
                        $('#analyzeBtn').html('<i class="fas fa-chart-line me-2"></i> Analyze Emotion');
                        $('#analyzeBtn').prop('disabled', false);
                    }
                });
            });
            
            function updateUI(data) {
                // Update final emotion display
                $('#finalEmotion').text(data.final_emotion)
                    .removeClass('happy sad anger fear surprise neutral')
                    .addClass(data.final_emotion.toLowerCase());
                
                // Update confidence indicator
                const confidencePercent = data.fused_confidence_score;
                $('#confidenceIndicator').css('left', confidencePercent + '%');
                
                // Update main metrics
                $('#trustScore').text(Math.round(data.fused_trust_score) + '%');
                $('#confidenceScore').text(Math.round(data.fused_confidence_score) + '%');
                
                // Update face analysis metrics
                $('#faceEmotion').text(data.face_emotion_result)
                    .removeClass('happy sad anger fear surprise neutral')
                    .addClass(data.face_emotion_result.toLowerCase());
                $('#faceConfidence').text(Math.round(data.face_avg_conf) + '%');
                $('#faceTrust').text(Math.round(data.face_trust) + '%');
                $('#faceEntropy').text(data.face_entropy.toFixed(2));
                
                // Update text analysis metrics
                $('#textEmotion').text(data.text_emotion_result)
                    .removeClass('happy sad anger fear surprise neutral')
                    .addClass(data.text_emotion_result.toLowerCase());
                $('#textConfidence').text(Math.round(data.text_confidence) + '%');
                $('#textTrust').text(Math.round(data.text_trust) + '%');
                
                // Update fusion weights
                const totalTrust = data.face_trust + data.text_trust;
                const faceWeight = Math.round((data.face_trust / totalTrust) * 100);
                const textWeight = Math.round((data.text_trust / totalTrust) * 100);
                $('#faceWeight').text(faceWeight + '%');
                $('#textWeight').text(textWeight + '%');
            }
        });
    </script>
</body>
</html>
    ''')

@app.route('/start_face_capture', methods=['POST'])
def start_face_capture():
    global face_capture_active, capture_thread, face_emotions, face_confidence_scores
    
    if not face_capture_active:
        # Reset previous data
        face_emotions = []
        face_confidence_scores = []
        face_capture_active = True
        
        if capture_thread is None or not capture_thread.is_alive():
            capture_thread = threading.Thread(target=capture_face_emotion)
            capture_thread.start()
            print("Face capture started")
            return jsonify({"status": "Face capture started"})
    
    return jsonify({"status": "Face capture already running"})

@app.route('/analyze', methods=['POST'])
def analyze():
    global face_capture_active, face_emotions, face_confidence_scores, face_data_ready

    text_input = request.form['text_input']
    face_capture_active = False  # Stop capturing when button is pressed

    if not face_emotions:
        face_emotions = ["neutral"]
        face_confidence_scores = [50.0]

    text_emotion_result, text_confidence = analyze_text_emotion(text_input)
    text_trust = compute_metrics([text_emotion_result], [text_confidence])[-1]
    face_emotion_result, face_avg_conf, _, face_entropy, face_trust = compute_metrics(face_emotions, face_confidence_scores)

    W_face = face_trust / (face_trust + text_trust) if (face_trust + text_trust) != 0 else 0.5
    W_text = text_trust / (face_trust + text_trust) if (face_trust + text_trust) != 0 else 0.5
    fused_emotion_score = (W_face * 1) + (W_text * 1)
    fused_trust_score = (W_face * face_trust) + (W_text * text_trust)
    fused_confidence_score = (W_face * face_avg_conf) + (W_text * text_confidence)
    final_emotion = face_emotion_result if W_face > W_text else text_emotion_result

    # Save final emotion in session
    session['final_emotion'] = final_emotion

    return jsonify({
        'final_emotion': final_emotion,
        'fused_emotion_score': fused_emotion_score,
        'fused_trust_score': fused_trust_score,
        'fused_confidence_score': fused_confidence_score,
        'face_emotion_result': face_emotion_result,
        'face_avg_conf': face_avg_conf,
        'face_trust': face_trust,
        'face_entropy': face_entropy,
        'text_emotion_result': text_emotion_result,
        'text_confidence': text_confidence,
        'text_trust': text_trust
    })

@app.route('/get_gif')
def get_gif():
    emotion = request.args.get('emotion', 'neutral')
    gif_path = download_giphy_gif(emotion)
    return jsonify(gif_path)

if __name__ == '__main__':
    # Create static/gifs directory if it doesn't exist
    static_gifs_folder = os.path.join(app.root_path, 'static/gifs')
    if not os.path.exists(static_gifs_folder):
        os.makedirs(static_gifs_folder)
    
    app.run(debug=True)# update this code
