import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import shutil
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from tensorflow.keras.models import load_model


apiUrl = 'http://0.0.0.0:8000'

user_img_path_recommend = 'images/upload/recommendation/user_image.jpg'
recommended_img_path = 'images/results/recommendation/'

# Load pre-trained model
feature_extractor = load_model('models/resnest50.keras')

classes = {'0': 'Trousers', '1': 'Dress', '2': 'Sweater', '3': 'T-shirt', '4': 'Top', '5': 'Blouse'}

# Load database embeddings, labels, and paths
database_embeddings = np.load('data/embeddings/embeddings.npy')
database_labels = np.load('data/embeddings/labels.npy')
paths = np.load('data/embeddings/paths.npy')
ids = [path[-13: -4] for path in paths]

# Create a DataFrame for easy filtering
data = pd.DataFrame({
    'id': ids,
    'embedding': list(database_embeddings),
    'label': list(database_labels),
    'path': list(paths)
})

# Valid combinations for recommendations
valid_combinations = {
    '0': ['2', '3', '4', '5'],
    '1': ['2'],
    '2': ['0', '1', '5'],
    '3': ['0'],
    '4': ['0'],
    '5': ['0', '2'],
}

classes = {'0': 'Trousers', '1': 'Dress', '2': 'Sweater', '3': 'T-shirt', '4': 'Top', '5': 'Blouse'}

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
    query_image_class = np.argmax(query_image_embeddings, axis=1)[0]
    return query_image_embeddings, query_image_class

# Get valid items
def get_valid_items(input_class, data):
    valid_classes = valid_combinations.get(input_class, [])
    valid_items = data[data['label'].isin(valid_classes)]
    if valid_items.empty:
        return np.array([]), np.array([]), []
    valid_items_embeddings = np.stack(valid_items['embedding'])
    valid_items_labels = np.stack(valid_items['label'])
    valid_items_paths = valid_items['path'].tolist()
    return valid_items_embeddings, valid_items_labels.tolist(), valid_items_paths

import shutil

def knn_recommend(query_embedding, valid_items_embeddings, valid_items_labels, valid_items_paths, k = 5):
    knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn_model.fit(valid_items_embeddings)
    distances, indices = knn_model.kneighbors(query_embedding)

    recommended_paths = [valid_items_paths[idx] for idx in indices[0]]
    
    paths = []
    # Save recommended images to the designated folder
    for i, file_path in enumerate(recommended_paths):
        dest_path = f'{recommended_img_path}recommended_{i}.jpg'
        img_path = f'{file_path[3:]}'
        paths.append(dest_path)
        print(f'{img_path=}')
        shutil.copy(img_path, dest_path)


    recommended_labesl_name = []
    ind = indices[0].tolist()

    for i in ind:
        num = valid_items_labels[i]
        recommended_labesl_name.append(classes[num])

    print(f'{distances=}')
    
    # Create recommendations
    recommendations = [
        {
            'similarity': str(distances[0][i]),  # Convert similarity to a string
            'path': f'{apiUrl}/{paths[i]}', # Use the original path of the image
            'label':  recommended_labesl_name[i] # Ensure labels are lists
        }
        for i in range(k)
    ]

    return recommendations


def recommend_outfit(file: UploadFile = File(...)):
    # Extract features
    query_image_embeddings, query_image_class = extract_features(user_img_path_recommend, feature_extractor)

    query_image_class = str(query_image_class)
    print(f'Query image class: {query_image_class}')

    # Get valid items
    valid_items_embeddings, valid_items_labels, valid_items_paths = get_valid_items(query_image_class, data)
    if valid_items_embeddings.size == 0:
        return JSONResponse(content={
        "message": 'No valid recommendations found for the query class'
    })
                               

    # Find recommendations
    recommendations = knn_recommend(query_image_embeddings, valid_items_embeddings, valid_items_labels, valid_items_paths)

    print(f'{user_img_path_recommend=}')
    print(f'{query_image_class=}')
    print(f'{recommendations=}')
    
    # Return recommendations in JSON format
    return JSONResponse(content={
        "user_image_class": f'{classes[query_image_class]}',
        "recommendations": recommendations
    })