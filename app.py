from flask import Flask, render_template,request, jsonify, send_from_directory
import pandas as pd
import pickle
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']
    print("uploaded file: ")
    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'})

    # if file:
    #     filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(filename)
    if os.path.exists("c:/Users/Sameer/Downloads/captured_image (5).jpg"):
        filename =  "c:/Users/Sameer/Downloads/captured_image (5).jpg"
    # Check if img2 is present in the folder
    elif os.path.exists("c:/Users/Sameer/Downloads/captured_image (4).jpg"):
        filename =  "c:/Users/Sameer/Downloads/captured_image (4).jpg"
    elif os.path.exists("c:/Users/Sameer/Downloads/captured_image (3).jpg"):
        filename =  "c:/Users/Sameer/Downloads/captured_image (3).jpg"
    elif os.path.exists("c:/Users/Sameer/Downloads/captured_image (2).jpg"):
        filename =  "c:/Users/Sameer/Downloads/captured_image (2).jpg"
    elif os.path.exists("c:/Users/Sameer/Downloads/captured_image (1).jpg"):
        filename =  "c:/Users/Sameer/Downloads/captured_image (1).jpg"
    elif os.path.exists("c:/Users/Sameer/Downloads/captured_image.jpg"):
        filename =  "c:/Users/Sameer/Downloads/captured_image.jpg"
    


    # filename = "c:/Users/Sameer/Downloads/captured_image.jpg"
    index = model_prediction(filename)
    
    if 0 <= index < len(df):
        product_info = get_product_info(index)
        print("fetching product info: ")
    # Check if the product is in stock
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
        return jsonify({
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
    else:
        return jsonify({'error': 'Product out of stock'})
    return jsonify({
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
    # else:
    #     return jsonify({'error': 'Product not found'})

# Serve uploaded images
# def delete_image(file_path):
#     try:
#         os.remove(file_path)
#         print(f"File '{file_path}' deleted successfully.")
#     except OSError as e:
#         print(f"Error deleting file '{file_path}': {e}")

# # Example usage
# file_to_delete = "c:/Users/Sameer/Downloads/captured_image.jpg"
# delete_image(file_to_delete)

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