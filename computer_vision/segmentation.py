import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from IPython.display import FileLink

feature_extractor = load_model('models/resnest50.keras')
model = FileLink('models/segmentation_model.keras')

apiUrl = 'http://54.158.224.198:8000'
user_img_path_segment = 'images/upload/segmentation/user_image.jpg'
segmented_img_path = 'images/results/segmentation/'


# ===| Helper functions |===
# Image preprocessing
def preprocess_image(image_path, image_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Extract features
def extract_features(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    query_image_embeddings = model.predict(preprocessed_image)
    return query_image_embeddings


def segment_image():
    query_image_embeddings = extract_features(user_img_path_segment, feature_extractor)
    print(f'{query_image_embeddings.shape=}')
    # output = model(query_image_embeddings)['out']
    return JSONResponse(content={"file": user_img_path_segment})

