from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from io import BytesIO

app = Flask(__name__)

# Other code...
df = pd.read_csv("Fruitsandveg.csv", index_col=0)

# Define the path to store uploaded images
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to get product information based on index (row number in CSV)
def get_product_info(index):
    product_info = df.iloc[index].to_dict()
    return product_info

# Function to preprocess the image and get the index
def model_prediction(image_bytes):
    model = tf.keras.models.load_model("trained_model.h5")
    # image = tf.keras.preprocessing.image.load_img(image_bytes, target_size=(64, 64))
    image = tf.keras.preprocessing.image.load_img(BytesIO(image_bytes), target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)
cart_items = []
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.json
    image_data = data.get('image')
    image_bytes = base64.b64decode(image_data.split(',')[1])

    # Preprocess the image and get the index
    index = model_prediction(image_bytes)

    if 0 <= index < len(df):
        product_info = get_product_info(index)
        # Check if the product is in stock
        stock_availability = product_info['Stock Availability']
        if stock_availability > 0:
            availability = 'In Stock'
            product_info['Stock Availability'] -= 1  # Decrement stock count
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
            return jsonify({'error': 'Product out of stock'})
    else:
        return jsonify({'error': 'Product not found'})

# Other code...
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Route to get the items in the cart
@app.route('/get_cart_items')
def get_cart_items():
    print('Current cart_items:', cart_items)  # Debug line
    return jsonify({'cart_items': cart_items})
#   return send_file(io.BytesIO(img), mimetype='image/png')

@app.route('/payment')
def payment():
    total_amount = sum(float(item['Price'].replace('$', '').strip()) for item in cart_items)

    # Pass the cart items and total amount to the payment template
    return render_template('payment.html', cart_items=cart_items, total_amount=total_amount)


if __name__ == '__main__':
    app.run(debug=True)