🛍️ Fashion Recommendation System (Image-Based + Trend Analysis)

A Deep Learning–based Fashion Product Recommendation System using ResNet50 and Similarity Search

📌 Project Overview

This project implements an image-based fashion recommendation system that suggests visually similar fashion products using deep visual embeddings extracted from a pre-trained ResNet50 model.

Along with recommendations, the project also includes fashion trend analysis using the dataset’s metadata (categories, colors, season).

The project includes:

🌟 Image-based recommendation engine

🤖 ResNet50-based feature extraction

🧠 Hybrid similarity evaluation (Image-only vs Hybrid)

📊 Trend analysis (Top colors, category distribution)

🎨 Streamlit UI for easy interaction

📁 Full codebase: feature extraction, evaluation, trends, Streamlit

🔥 Key Features
✔ 1. Image-Based Recommendations

Upload a product image → system returns visually similar items using:

ResNet50 embeddings

Euclidean / Cosine similarity

✔ 2. Trend Analysis

Using styles.csv, we generate:

Category distribution (Fig. 5.1)

Top 10 colors (Fig. 5.2)

✔ 3. Evaluation (Precision@5)

Comparison of:

Image-only model

Hybrid model (image + metadata)

(Used only offline, UI uses image-only.)

✔ 4. Modular Project Structure

Separate modules for:

Feature extraction

Metadata encoding

Recommendation logic

Trend analysis

Evaluation

📂 Project Structure
fashion-recommendation-system/
│
│── app/
│   └── main.py                       # Streamlit UI
│
│── models/
│   ├── embeddings.pkl                # Visual embeddings (ResNet50)
│   ├── metadata_features.npy         # Encoded metadata vectors
│   ├── meta_map.pkl
│   ├── meta_index_map.pkl
│
│── trend_analysis/
│   ├── trend_analysis.py
│   ├── top_colors.png
│   ├── category_distribution.png
│
│── evaluation/
│   ├── evaluate.py
│   ├── evaluation_results.txt
│
│── data/
│   ├── styles.csv  (optional – link recommended)
│
│── README.md
│── requirements.txt
│── .gitignore

🧠 How It Works (Architecture)
1. Input

User uploads a fashion product image

System preprocesses it (224×224 size, normalization)

2. Feature Extraction

ResNet50 (ImageNet weights)

Global Max Pooling → creates 2048-dim visual embedding

3. Similarity Computation

Cosine similarity between embeddings

Top-5 nearest items returned

4. Trend Insights

Generated from metadata, not from the model:

Color popularity

Category distribution

🚀 How to Run Locally
1. Clone the Repository
git clone https://github.com/your-username/fashion-recommendation-system.git
cd fashion-recommendation-system

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run Streamlit App
streamlit run app/main.py

📊 Trend Analysis Outputs
Fig 5.1 – Category Distribution

(Pie chart showing Apparel, Footwear, Accessories dominance)

Fig 5.2 – Top 10 Colors

(Bar chart showing Blue, Black, White as top colors)

🧪 Evaluation Results
Model Type	Avg Precision@5
Image-Only Model	1.00
Hybrid Model	1.00 (offline only)

✔ Hybrid model uses metadata + image
✔ Image-only is used in the deployed UI

🛠️ Technologies Used
Deep Learning

TensorFlow / Keras

ResNet50 (feature extractor)

Similarity & ML

scikit-learn

cosine similarity

Nearest Neighbors

Data Processing

NumPy

Pandas

Pickle

Visualization

Matplotlib

Seaborn

Frontend

Streamlit

📘 Dataset Source

Fashion Product Images (Small)
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

🎯 Use Cases

Apparel e-commerce

Visual search

Trend analysis for brands

Product recommendation systems

Fashion analytics

📚 Screenshots

(Add these manually after pushing:)

Streamlit Homepage

Input image + recommendations

Trend analysis charts

Architecture diagram

📝 Future Improvements

Add real-time trend data

Include user ratings/feedback

Deploy hybrid model in UI

Integrate ANN search (FAISS)

Add brand, fabric, text description embeddings

👤 Author

Sameer Lonare
M.Tech IT • Delhi Technological University (DTU)
