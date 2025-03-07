from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
import imutils
import image_dehazer

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputImages'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')



def remove_haze_from_image(image_path, original_filename):
    HazeImg = cv2.imread(image_path)
    HazeCorrectedImg, _ = image_dehazer.remove_haze(HazeImg, showHazeTransmissionMap=False)

    output_filename = "Gen_" + original_filename  # New filename for the processed image
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    cv2.imwrite(output_path, HazeCorrectedImg)

    return output_filename  # Return the new filename

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    processed_img = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            processed_img = remove_haze_from_image(file_path, filename)

            image = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], processed_img))
            image = imutils.resize(image, width=400)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(CLASSES):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'processed_detection.jpg'), image)

            return render_template('result.html', processed_img=processed_img)

    return render_template('index.html')
@app.route('/object')
def object():
    return render_template('object.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
