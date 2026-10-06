from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
CORS(app)


# ==========================================
# LOAD AI MODEL
# ==========================================

print("Loading AI model...")

model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

print("AI model loaded successfully!")


# ==========================================
# FOOD INFORMATION
# ==========================================

food_info = {

    "pizza": {
        "name": "Pizza",
        "nutrition": "Approximately 266 calories per 100g"
    },

    "cheeseburger": {
        "name": "Cheeseburger",
        "nutrition": "Approximately 295 calories per 100g"
    },

    "hotdog": {
        "name": "Hot Dog",
        "nutrition": "Approximately 290 calories per 100g"
    },

    "banana": {
        "name": "Banana",
        "nutrition": "Approximately 89 calories per 100g"
    },

    "orange": {
        "name": "Orange",
        "nutrition": "Approximately 47 calories per 100g"
    },

    "apple": {
        "name": "Apple",
        "nutrition": "Approximately 52 calories per 100g"
    },

    "pineapple": {
        "name": "Pineapple",
        "nutrition": "Approximately 50 calories per 100g"
    },

    "strawberry": {
        "name": "Strawberry",
        "nutrition": "Approximately 32 calories per 100g"
    },

    "broccoli": {
        "name": "Broccoli",
        "nutrition": "Approximately 34 calories per 100g"
    },

    "cauliflower": {
        "name": "Cauliflower",
        "nutrition": "Approximately 25 calories per 100g"
    },

    "carrot": {
        "name": "Carrot",
        "nutrition": "Approximately 41 calories per 100g"
    },

    "cucumber": {
        "name": "Cucumber",
        "nutrition": "Approximately 15 calories per 100g"
    },

    "corn": {
        "name": "Corn",
        "nutrition": "Approximately 86 calories per 100g"
    },

    "egg": {
        "name": "Egg",
        "nutrition": "Approximately 155 calories per 100g"
    },

    "mushroom": {
        "name": "Mushroom",
        "nutrition": "Approximately 22 calories per 100g"
    },

    "rice": {
        "name": "Rice",
        "nutrition": "Approximately 130 calories per 100g"
    }

}


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "message": "FoodWise AI server is running!",
        "status": "success"
    })


# ==========================================
# PREDICT FOOD
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check whether image was received

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400


        file = request.files["image"]


        if file.filename == "":

            return jsonify({
                "error": "No image selected"
            }), 400


        # ==================================
        # OPEN IMAGE
        # ==================================

        image = Image.open(file).convert("RGB")


        # ==================================
        # RESIZE IMAGE
        # ==================================

        image = image.resize((224, 224))


        # ==================================
        # CONVERT TO ARRAY
        # ==================================

        image_array = np.array(image)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ==================================
        # PREPROCESS IMAGE
        # ==================================

        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )


        # ==================================
        # AI PREDICTION
        # ==================================

        predictions = model.predict(
            image_array,
            verbose=0
        )


        decoded_predictions = (
            tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions,
                top=5
            )[0]
        )


        # Get highest prediction

        prediction = decoded_predictions[0]

        class_id = prediction[0]

        label = prediction[1]

        confidence = float(prediction[2]) * 100


        # ==================================
        # CHECK FOOD DATABASE
        # ==================================

        detected_food = None

        for key in food_info:

            if key.lower() in label.lower():

                detected_food = key
                break


        # ==================================
        # FOOD FOUND
        # ==================================

        if detected_food:

            food_name = food_info[detected_food]["name"]

            nutrition = food_info[detected_food]["nutrition"]


        else:

            food_name = label.replace("_", " ").title()

            nutrition = "Nutrition information is not available for this item yet."


        # ==================================
        # SEND RESULT
        # ==================================

        return jsonify({

            "success": True,

            "food": food_name,

            "prediction": label,

            "confidence": round(confidence, 2),

            "nutrition": nutrition

        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print("")
    print("====================================")
    print("      FOODWISE AI SERVER")
    print("====================================")
    print("")
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print("")
    print("Waiting for food images...")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )