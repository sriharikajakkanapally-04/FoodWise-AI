from transformers import pipeline

print("Loading AI model...")

classifier = pipeline(
    "image-classification",
    model="nateraw/food"
)

print("AI model loaded successfully!")