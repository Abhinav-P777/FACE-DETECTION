

from flask import Flask, request, render_template, Response
import os
import cv2
import tensorflow as tf
import numpy as np
from io import BytesIO
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the pre-trained model
model_path = 'ssd_mobilenet_v2_coco_2018_03_29/saved_model'
model = tf.saved_model.load(model_path)

# List of COCO labels (for object detection)
coco_labels = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "N/A", "backpack", "umbrella", "N/A", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Helper function to check if the file is a valid image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to resize the image to a manageable size (limit to 800x800 max)
def resize_image(image_path, max_size=(800, 800)):
    image = Image.open(image_path)
    image.thumbnail(max_size)
    image.save(image_path)
    return image_path

# Helper function to perform object detection and return an image as a response
def object_detection(image_path):
    # Resize image to avoid memory overload
    image_path = resize_image(image_path)

    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    model_fn = model.signatures['serving_default']
    output = model_fn(input_tensor)

    boxes = output['detection_boxes']
    classes = output['detection_classes']
    scores = output['detection_scores']

    for i in range(len(boxes[0])):
        if scores[0][i] > 0.5:  # Confidence score threshold (e.g., 0.5)
            box = boxes[0][i].numpy()
            class_id = int(classes[0][i].numpy())  # Class ID
            score = scores[0][i].numpy()

            # Map the class ID to the class name
            class_name = coco_labels[class_id]

            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * image.shape[1]), int(ymin * image.shape[0]))
            end_point = (int(xmax * image.shape[1]), int(ymax * image.shape[0]))
            color = (0, 255, 0)  # Green bounding box
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

            label = f"{class_name}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            image = cv2.putText(image, label, start_point, font, 0.5, color, 1, cv2.LINE_AA)

    # Convert image to BytesIO object for response
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = BytesIO(img_encoded.tobytes())
    return img_bytes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform object detection and get processed image as response
        try:
            processed_image = object_detection(file_path)
            return Response(processed_image.getvalue(), content_type='image/jpeg')
        except Exception as e:
            return f"Error processing image: {str(e)}", 500
    else:
        return 'Invalid file type. Please upload a valid image.', 400

if __name__ == '__main__':
    app.run(debug=True)