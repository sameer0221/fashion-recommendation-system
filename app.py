import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ---------------------------
# Load ResNet50 model
# ---------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ---------------------------
# Feature extraction function
# ---------------------------
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# ---------------------------
# Load all image filenames
# ---------------------------
# filenames = [os.path.join('images', f) for f in os.listdir('images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
filenames = []
for i, file in enumerate(os.listdir('images')):
    if i >= 1000:
        break
    filenames.append(os.path.join('images', file))


# ---------------------------
# Extract features in batches
# ---------------------------
feature_list = []
batch_size = 32
for i in tqdm(range(0, len(filenames), batch_size), desc="Processing batches"):
    batch_files = filenames[i:i+batch_size]
    batch_features = [extract_features(f, model) for f in batch_files]
    feature_list.extend(batch_features)

# Save embeddings
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
print(f"✅ Saved embeddings.pkl and filenames.pkl ({len(filenames)} images)")

# ---------------------------
# Load Metadata
# ---------------------------
meta_df = pd.read_csv("data/styles.csv", on_bad_lines='skip')
if 'id' in meta_df.columns:
    meta_df['id'] = meta_df['id'].astype(str) + ".jpg"

# ---------------------------
# Prepare Hybrid Metadata
# ---------------------------
meta_features_cols = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
meta_features = meta_df[meta_features_cols].fillna('Unknown')

encoder = OneHotEncoder(sparse_output=False)
meta_vectors = encoder.fit_transform(meta_features)

np.save('metadata_features.npy', meta_vectors)

# Map filename -> index in meta_vectors
meta_index_map = {row['id']: idx for idx, row in meta_df.iterrows() if row['id'] in [os.path.basename(f) for f in filenames]}
pickle.dump(meta_index_map, open('meta_index_map.pkl', 'wb'))
print("✅ Saved metadata_features.npy and meta_index_map.pkl")

# Map filename -> metadata dict
meta_map = {}
for _, row in meta_df.iterrows():
    fname = row['id']
    if fname in [os.path.basename(f) for f in filenames]:
        meta_map[fname] = row.to_dict()
pickle.dump(meta_map, open("meta_map.pkl", "wb"))
print(f"✅ Saved meta_map.pkl with {len(meta_map)} entries")
