# ai
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

# Load Pre-trained CheXNet Model (DenseNet121 trained on Chest X-ray data)
model = DenseNet121(weights="imagenet")

# Function to check if file exists
def check_file_exists(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Error: File '{img_path}' not found. Check the path.")

# Function to preprocess medical image
def preprocess_medical_image(img_path):
    check_file_exists(img_path)
    img = image.load_img(img_path, target_size=(224, 224))  # Ensure correct resizing
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict disease
def predict_disease(img_path):
    img_array = preprocess_medical_image(img_path)
    preds = model.predict(img_array)
    decoded_preds = tf.keras.applications.densenet.decode_predictions(preds, top=3)[0]
    return decoded_preds

# Grad-CAM Implementation for Explainability
def grad_cam(img_path, model, layer_name='conv5_block16_concat'):
    img_array = preprocess_medical_image(img_path)
    
    # Ensure the model contains the specified layer
    if layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Error: Layer '{layer_name}' not found in model. Check the model architecture.")
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # Ensure no negative values
    heatmap = heatmap / np.max(heatmap)  # Normalize
    heatmap = heatmap.reshape((7, 7))
    
    return heatmap

# Function to overlay Grad-CAM heatmap on the original image
def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Error: Failed to load image. Check the file format and path.")
    
    img = cv2.resize(img, (224, 224))  # Resize to match Grad-CAM output
    heatmap = cv2.resize(heatmap, (224, 224))  # Resize heatmap

    # Normalize and apply color mapping
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend heatmap with the original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

# Example Usage
img_path = 'medical_xray.jpg'  # Replace with the actual path of the medical image

try:
    # Step 1: Get AI Diagnosis
    diagnosis_results = predict_disease(img_path)
    print("Diagnosis Predictions:", diagnosis_results)

    # Step 2: Generate Grad-CAM Heatmap
    heatmap = grad_cam(img_path, model)

    # Step 3: Overlay heatmap on original image
    superimposed_img = overlay_heatmap(img_path, heatmap)

    # Display Results
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("AI Diagnosis with Heatmap")
    plt.axis("off")
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
