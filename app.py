from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Load the CSV file into a Pandas DataFrame
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
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

cart_items = []

# Route to serve the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to process the uploaded image
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file found'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the uploaded file to the upload folder
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Perform model prediction using the file path
        index = model_prediction(filename)

        if 0 <= index < len(df):
            product_info = get_product_info(index)
            stock_availability = product_info['Stock Availability']
            if stock_availability > 0:
                availability = 'In Stock'
                product_info['Stock Availability'] -= 1  # Decrement stock count
                product_info['Image URL'] = f"/uploads/{file.filename}"
                cart_items.append({
                    'Product Name': product_info['name'],
                    'Product ID': product_info['Product ID'],
                    'Category': product_info['Category'],
                    'Price': product_info['Price'],
                    'Unit of Measure': product_info['Unit of Measure'],
                    'Description': product_info['Description'],
                    'Image URL': product_info['Image URL'],
                    'Stock Availability': availability,
                    'Discount': product_info['Discount'],
                    'Origin': product_info['Origin']
                })
                return jsonify(product_info)
            else:
                return jsonify({'error': 'Product out of stock'})
        else:
            return jsonify({'error': 'Invalid product index'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to get the items in the cart
@app.route('/get_cart_items')
def get_cart_items():
    print('Current cart_items:', cart_items)  # Debug line
    return jsonify({'cart_items': cart_items})

# Route for payment
@app.route('/payment')
def payment():
    total_amount = sum(float(item['Price'].replace('$', '').strip()) for item in cart_items)
    return render_template('payment.html', cart_items=cart_items, total_amount=total_amount)

if __name__ == 'main':
    app.run(debug=True)