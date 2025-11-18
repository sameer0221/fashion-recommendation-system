import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load precomputed data
# -----------------------------
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
meta_map = pickle.load(open("meta_map.pkl", "rb"))

# Metadata for hybrid similarity
meta_vectors = np.load('metadata_features.npy')
meta_index_map = pickle.load(open('meta_index_map.pkl', 'rb'))

# -----------------------------
# Load model
# -----------------------------
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommender System')

# -----------------------------
# Helper functions
# -----------------------------
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# -----------------------------
# Hybrid Recommendation
# -----------------------------
def hybrid_recommend(features, feature_list, uploaded_fname, top_k=5, alpha=0.7, beta=0.3):
    # Image similarity
    img_sim = cosine_similarity([features], feature_list)[0]

    # Metadata similarity
    meta_idx = meta_index_map.get(uploaded_fname, None)
    if meta_idx is not None:
        meta_sim = cosine_similarity([meta_vectors[meta_idx]], meta_vectors)[0]
    else:
        meta_sim = np.zeros(len(feature_list))

    # Weighted similarity
    combined_sim = alpha * img_sim + beta * meta_sim

    # Get top indices (skip query itself)
    top_indices = combined_sim.argsort()[::-1][1:top_k+1]
    return top_indices

# -----------------------------
# Streamlit UI
# -----------------------------
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image")

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Hybrid recommendation
        indices = hybrid_recommend(features, feature_list, uploaded_file.name)

        st.subheader("Recommended Similar Items:")

        # Display recommendations with metadata
        cols = st.columns(5)
        for i, col in enumerate(cols):
            idx = indices[i]
            img_path = filenames[idx]
            col.image(img_path, width=150)

            fname = os.path.basename(img_path)
            if fname in meta_map:
                meta = meta_map[fname]
                meta_text = f"{meta.get('articleType','')} | {meta.get('baseColour','')}"
                col.caption(meta_text)
    else:
        st.header("Some error occurred in file upload")
