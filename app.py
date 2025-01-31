from flask import Flask, request, render_template, send_file, Response, make_response
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
DPALANIS_FOLDER = 'static/'

class Detection:
    def __init__(self):
        #download weights from here:https://github.com/ultralytics/ultralytics and change the path
        self.model = YOLO("best.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)

        return results

    def predict_and_detect(self, file_path, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(file_path, classes, conf=conf)
       
        save_path = Path(DPALANIS_FOLDER) / 'annotated_image.jpg' 
        results[0].save(save_path)
        # Accessing the class names
        predicted_classes = []
        print("**************");

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Get the class index
                class_name = self.model.names[class_id]  # Get the class name
                predicted_classes.append(class_name)
        print(predicted_classes)

        im = results[0].plot()
       
        return im, predicted_classes

    def detect_from_image(self, file_path):
        result_img, predicted_classes = self.predict_and_detect(file_path, classes=[], conf=0.2)
        return result_img, predicted_classes


detection = Detection()

@app.route('/static/<path:filename>') 
def serve_static_file(filename): 
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        #img = Image.open(file_path).convert("RGB")
        #img = np.array(img)
        #img = cv2.resize(img, (640, 640))
        img, predicted_classes = detection.detect_from_image(file_path)
        output = Image.fromarray(img)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)
        car_name = 'Dhandapani'
        os.remove(file_path)
        #return send_file(buf, mimetype='image/png')
        #return send_file(buf, mimetype='image/png', headers={'X-Car-Name': car_name})
        
        response = make_response(send_file(buf, mimetype='image/png')) 
        response.headers['X-Car-Name'] = predicted_classes[0]
        response.headers['Access-Control-Expose-Headers'] = 'X-Car-Name'
        #response = make_response(send_file(buf, mimetype='image/png')) 
        
        return response


@app.route('/video')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (512, 512))
        if frame is None:
            break
        frame = detection.detect_from_image(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    #http://localhost:8000/video for video source
    #http://localhost:8000 for image source
