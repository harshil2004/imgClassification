import PIL
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

class_names: dict[int, str] = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

model = load_model('trained_model.keras')


def classify(model, image):
    """
    Classifies an image using the provided model.

    Args:
        model (tensorflow.keras.Model): The pre-trained model for classification.
        image (np.ndarray or PIL.Image): The image to be classified.

    Returns:
        tuple: A tuple containing the probability of the predicted class and the predicted class label.
    """

    if isinstance(image, PIL.Image.Image):
        img = image.convert("L")
        img = img.resize((48, 48))
        data = np.asarray(img)
    else:
        # Assuming image is a NumPy array (BGR format from Streamlit)
        data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, (48, 48))

    # Normalize pixel values (optional)
    data = data.astype("float32") / 255.0  # Normalize for some models

    probs = model.predict(np.expand_dims(data, axis=0))  # Add batch dimension

    prob = probs.max()
    pred = class_names[np.argmax(probs)]

    return prob, pred


def main():
    """Image Classification App with Camera Input"""
    st.title("Image Classification with Camera")

    # Get camera input from user
    image = st.camera_input("Capture an image")

    # Run classification if image is captured
    if image is not None:
        try:
            prob, predicted_class = classify(model, image)

            st.write(f"**Classification Results:**")
            st.write(f"- Predicted Class: {predicted_class}")
            st.write(f"- Confidence Probability: {prob:.2f}")  # Display probability with 2 decimal places

        except Exception as e:
            st.error(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
