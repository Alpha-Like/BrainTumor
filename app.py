from flask import Flask, request, send_file, render_template, request, after_this_request
from segmentation_classification.inference import process_image
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return {"error": "No file part"}, 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return {"error": "No selected file"}, 400
    
    if not file.filename.endswith('.png'):
        return {'error': f'Convert {file.filename.split('.')[-1]} to png format.'}
    
    os.makedirs('temp', exist_ok=True)
    file.save(f"temp/{file.filename}")
    im = process_image(f"temp/{file.filename}")
    os.remove(f"temp/{file.filename}")
    return {'route': f'/result?path={im}'}

@app.route('/result')
def result():
    path = request.args['path']

    @after_this_request
    def cleanup(resp):
        os.remove(path)
        return resp

    return send_file(path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
