from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    if video:
        # Save uploaded file (just to simulate)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)

        # === FAKE RESPONSE ===
        processed_video_filename = 'demo_processed_video.mp4'  # This must be in static/processed/

        # Fake comments to return
        comments = [
            {"timecode": 5, "text": "Player 7 made a great pass!"},
            {"timecode": 10, "text": "Shot on target by Player 10."},
            {"timecode": 15, "text": "Goal scored! Amazing finish."},
            {"timecode": 25, "text": "Yellow card for Player 3."}
        ]

        return jsonify({
            "success": True,
            "processed_video": processed_video_filename,
            "comments": comments
        })

    return jsonify({"success": False})

# Route to serve files if needed (usually Flask does this automatically from static/)
@app.route('/processed/<filename>')
def processed_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
