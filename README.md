# Hybrid CNN based Movie Recommendation System

This repository contains a **hybrid movie recommendation system** built using the TMDB 5000 Movies dataset. The system combines **collaborative filtering** (synthetic user ratings) with **content-based features** (textual and visual embeddings) using a deep neural network.

---

## 🚀 Features

1. **Dataset & Preprocessing**
   - Uses the TMDB 5000 Movies dataset (movies and credits CSVs from Kaggle).
   - Merges movies and credits on the title.
   - Custom `MovieDataProcessor` cleans and prepares data:
     - Parses JSON-like columns (`genres`, `keywords`, `cast`, `crew`) to text.
     - Extracts top-3 cast members and director.
     - Fills missing values and creates a `combined_features` string per movie (genres, keywords, cast, director, overview).
     - Assigns a unique integer index `movie_idx` for each movie.
   - Final processed DataFrame: ~4809 movies.

2. **Synthetic Ratings (Collaborative Filtering)**
   - Generates synthetic ratings for 500 users:
     - Each user rates 5–30 randomly selected movies.
     - Ratings based on TMDB `vote_average` and `popularity`, clipped to 1–5.
   - Produces ~8844 total ratings.
   - DataFrame columns: `user_id`, `movie_idx`, `movie_id`, `rating`.
   - Data split: 80% train, 20% validation/test.

3. **Visual Feature Extraction**
   - Creates artificial poster images (224×224 RGB) based on movie genres.
   - Extracts visual embeddings using:
     - **VGG16** (256-d features via Dense layers).
     - **ResNet50** alternative (512-d features with global average pooling).
   - Cosine similarity can be used for purely content-based recommendations.

4. **Hybrid Neural Recommendation Model**
   - Custom TensorFlow Keras model: `HybridNeuralRecommenderTF`.
   - Combines:
     - **User embeddings** (collaborative filtering).
     - **Movie embeddings** (collaborative filtering).
     - **Content embeddings** (text features or synthetic vectors).
     - **Visual embeddings** (VGG16/ResNet poster features).
   - Concatenated embeddings fed through deep Dense layers (512 → 256 → 128 → 1) with BatchNorm, ReLU, and Dropout.
   - Regression output predicts a real-valued rating (1–5).

5. **Training & Evaluation**
   - Loss: Mean Squared Error (MSE).
   - Metric: Mean Absolute Error (MAE).
   - Optimizer: Adam (lr=1e-3).
   - Training monitored via loss and MAE on train/validation splits.
   - Evaluation on held-out test set reports MSE and MAE.

6. **Generating Recommendations**
   - Function `generate_tf_recommendations()` predicts ratings for all movies for a given user.
   - Top-K recommendations are printed with predicted rating vs actual `vote_average`.
   - Visual content-based recommendations using cosine similarity of CNN embeddings are also demonstrated.

---

## 🔧 Dependencies

- Python 3.x  
- TensorFlow / Keras  
- NumPy, Pandas, scikit-learn  
- Matplotlib, PIL (image creation)  
- Kaggle API (optional for dataset download)  

---

## ⚙️ Pipeline Overview

1. **Data Loading** → Load movies & credits CSVs.  
2. **Preprocessing** → Merge tables, parse JSON, extract features, build combined text.  
3. **Synthetic Ratings** → Generate user–movie ratings for collaborative filtering.  
4. **Content & Visual Features** → Create posters, extract CNN embeddings.  
5. **TF Dataset Preparation** → Map IDs to indices, align ratings with content & visual vectors.  
6. **Model Construction** → Build and compile `HybridNeuralRecommenderTF`.  
7. **Training** → Fit on training data, monitor loss/MAE.  
8. **Evaluation & Recommendations** → Evaluate test set, generate top-K recommendations.

---

## 📊 Notes

- The notebook illustrates **hybrid recommendation** combining collaborative and content-based signals.
- Uses **cosine similarity** for visual embeddings for content-based retrieval.
- Focuses on **regression metrics (MSE, MAE)**; no explicit ranking metrics like Precision@K or MAP are computed.
- Synthetic ratings demonstrate collaborative filtering when real user ratings are unavailable.

---

## 📚 References

- [Keras: Neural Collaborative Filtering & Hybrid Recommenders](https://keras.io/examples/structured_data/collaborative_filtering/)
- [Evidently AI: Regression Metrics for Recommendation](https://evidentlyai.com/)
- Pre-trained CNNs: **VGG16, ResNet50** (Keras Applications)

---

## 🔗 Dataset

- [TMDB 5000 Movies Dataset (Kaggle)](https://www.kaggle.com/tmdb/tmdb-movie-metadata)

---

## 💡 Future Improvements

- Replace synthetic content vectors with actual TF-IDF or word embeddings from movie text.
- Incorporate real user ratings if available.
- Implement ranking-based metrics (Precision@K, NDCG) for evaluation.
- Extend visual content extraction with actual movie posters.
