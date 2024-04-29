from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from PIL import Image
import tempfile
from io import BytesIO

app = Flask(__name__)

df = pd.read_csv("Fruitsandveg.csv", index_col=0)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_product_info(index):
    product_info = df.iloc[index].to_dict()
    return product_info
def model_prediction(image_bytes):
    model = YOLO('./runs/classify/train/weights/last.pt')  # load a custom model

    # Create a temporary file from the image bytes
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name

    # Use the temporary file path with the YOLO model
    results = model(temp_file_path)
    os.remove(temp_file_path)  # Remove the temporary file

    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    return np.argmax(probs)
cart_items = []
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.json
    image_data = data.get('image')
    if not image_data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
    except (IndexError, ValueError):
        return jsonify({'error': 'Invalid image data'}), 400

    index = model_prediction(image_bytes)

    if 0 <= index < len(df):
        product_info = get_product_info(index)
        stock_availability = product_info['Stock Availability']
        if stock_availability > 0:
            availability = 'In Stock'
            product_info['Stock Availability'] -= 1
            cart_items.append({
                'Product Name': product_info['name'],
                'Price': product_info['Price'],
                'Product ID': product_info['Product ID']
            })
            return jsonify({
                'Product Name': product_info['name'],
                'Price': product_info['Price'],
                'Product ID': product_info['Product ID']
            })
        else:
            return jsonify({'error': 'Product out of stock'}), 200
    else:
        return jsonify({'error': 'Product not found'}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_cart_items')
def get_cart_items():
    print('Current cart_items:', cart_items)  
    return jsonify({'cart_items': cart_items})

@app.route('/payment')
def payment():
    total_amount = sum(float(item['Price']) for item in cart_items)
    return render_template('payment.html', cart_items=cart_items, total_amount=total_amount)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)