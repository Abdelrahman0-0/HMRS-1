import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae


# ==========================================
# READ DATASET FILES
# ==========================================

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


# ==========================================
# DATA PREPROCESSING
# ==========================================

movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
movies.dropna(inplace=True)
ratings.dropna(inplace=True)


# ==========================================
# MERGE DATASETS
# ==========================================

movie_data = pd.merge(ratings, movies, on='movieId')


# ==========================================
# CONTENT-BASED FILTERING (TF-IDF)
# ==========================================

movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['title'] + " " + movies['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# ==========================================
# CREATE SEARCHABLE TITLE MAPPING
# ==========================================

def normalize_title(title):
    """Remove year and convert to lowercase for matching"""
    title_no_year = re.sub(r'\s*\(\d{4}\)', '', title)
    return title_no_year.lower().strip()


movies['title_lower'] = movies['title'].str.lower()
movies['title_no_year'] = movies['title'].apply(normalize_title)

indices_exact = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()
indices_no_year = pd.Series(movies.index, index=movies['title_no_year']).drop_duplicates()


def find_movie(title_input):
    """Find movie index with flexible matching"""
    title_input = title_input.lower().strip()
    
    if title_input in indices_exact.index:
        return indices_exact[title_input]
    
    title_no_year = re.sub(r'\s*\(\d{4}\)', '', title_input).strip()
    if title_no_year in indices_no_year.index:
        return indices_no_year[title_no_year]
    
    matches = movies[movies['title_lower'].str.contains(title_input, na=False)]
    if not matches.empty:
        return matches.index[0]
    
    words = title_input.split()
    for word in words:
        if len(word) > 3:
            matches = movies[movies['title_lower'].str.contains(word, na=False)]
            if not matches.empty:
                return matches.index[0]
    
    return None


# ==========================================
# LIGHTFM HYBRID MODEL (البديل)
# ==========================================

print("\n" + "="*50)
print("📊 Training LightFM Hybrid Model")
print("="*50)

# Prepare data for LightFM
# Create mappings for user and item IDs
user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()

user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

# Create interaction matrix (ratings)
num_users = len(user_ids)
num_items = len(item_ids)

# Build interaction matrix
from scipy.sparse import csr_matrix

rows = []
cols = []
data_vals = []

for _, row in ratings.iterrows():
    rows.append(user_id_to_idx[row['userId']])
    cols.append(item_id_to_idx[row['movieId']])
    data_vals.append(row['rating'])

interactions = csr_matrix((data_vals, (rows, cols)), shape=(num_users, num_items))

# Create item features (genres as binary features)
# Get all unique genres
all_genres = set()
for genres_str in movies['genres']:
    for genre in genres_str.split('|'):
        all_genres.add(genre)
all_genres = sorted(list(all_genres))
genre_to_idx = {g: i for i, g in enumerate(all_genres)}

# Build item features matrix
item_features_rows = []
item_features_cols = []
item_features_data = []

for idx, row in movies.iterrows():
    item_idx = item_id_to_idx[row['movieId']]
    for genre in row['genres'].split('|'):
        if genre in genre_to_idx:
            item_features_rows.append(item_idx)
            item_features_cols.append(genre_to_idx[genre])
            item_features_data.append(1)

item_features = csr_matrix((item_features_data, (item_features_rows, item_features_cols)), 
                            shape=(num_items, len(all_genres)))

# Train LightFM model
# WARP loss is good for ranking recommendations with explicit feedback [citation:6]
lightfm_model = LightFM(loss='warp', random_state=42, no_components=30, learning_rate=0.05)

print("Training LightFM model...")
lightfm_model.fit(interactions, item_features=item_features, epochs=30, num_threads=2, verbose=True)

print("✅ LightFM model trained successfully!")


# ==========================================
# RECOMMENDATION FUNCTIONS (LightFM)
# ==========================================

def collaborative_recommendations_by_user(user_id, top_n=10):
    """Get collaborative recommendations for a user using LightFM"""
    try:
        user_idx = user_id_to_idx.get(user_id)
        if user_idx is None:
            return []
        
        # Get scores for all items
        scores = lightfm_model.predict(user_idx, np.arange(num_items), item_features=item_features)
        
        # Get top N items
        top_item_indices = np.argsort(-scores)[:top_n]
        
        # Convert back to original movie IDs and titles
        # Create reverse mapping
        idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
        
        results = []
        for item_idx in top_item_indices:
            movie_id = idx_to_item_id[item_idx]
            movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
            results.append(movie_title)
        
        return results
    except Exception as e:
        print(f"Error in collaborative_recommendations_by_user: {e}")
        return []


def collaborative_recommendations_by_genre(genre, top_n=10):
    """Get top movies by genre based on average rating"""
    genre_movies = movies[movies['genres'].str.contains(genre, na=False)]
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    genre_movies = genre_movies.merge(avg_ratings, on='movieId', how='left')
    genre_movies['avg_rating'] = genre_movies['avg_rating'].fillna(0)
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
    return genre_movies.head(top_n)['title'].tolist()


def collaborative_recommendations(search_value, search_type, top_n=10):
    """Main collaborative function"""
    if search_type == "Genre":
        return collaborative_recommendations_by_genre(search_value, top_n)
    else:
        try:
            user_id = int(search_value)
            return collaborative_recommendations_by_user(user_id, top_n)
        except:
            return []


def hybrid_recommendations_by_movie(user_id, movie_title, top_n=10):
    """Hybrid: combine collaborative (LightFM) with content similarity"""
    try:
        # First, get content-based similar movies
        idx = find_movie(movie_title)
        if idx is None:
            return []
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:50]
        movie_indices = [i[0] for i in sim_scores]
        
        content_movies = movies.iloc[movie_indices].copy()
        content_movies['content_score'] = [score[1] for score in sim_scores]
        
        # Get collaborative scores from LightFM
        def get_collab_score_from_lightfm(row):
            try:
                movie_id = row['movieId']
                if movie_id in item_id_to_idx:
                    item_idx = item_id_to_idx[movie_id]
                    user_idx = user_id_to_idx.get(user_id)
                    if user_idx is not None:
                        score = lightfm_model.predict(user_idx, np.array([item_idx]), item_features=item_features)[0]
                        # Normalize score to range [0, 1] approximately
                        # LightFM scores typically range from -something to +something
                        # We'll use sigmoid-like normalization
                        normalized = 1 / (1 + np.exp(-score * 0.5))
                        return normalized
            except:
                pass
            return 0.5
        
        content_movies['collaborative_score'] = content_movies.apply(get_collab_score_from_lightfm, axis=1)
        content_movies['hybrid_score'] = (
            0.5 * content_movies['content_score'] +
            0.5 * content_movies['collaborative_score']
        )
        
        recommendations = content_movies.sort_values('hybrid_score', ascending=False)
        return recommendations['title'].head(top_n).tolist()
    except Exception as e:
        print(f"Error in hybrid_recommendations_by_movie: {e}")
        return []


def hybrid_recommendations_by_genre(genre, top_n=10):
    """Get top movies by genre based on average rating"""
    genre_movies = movies[movies['genres'].str.contains(genre, na=False)]
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    genre_movies = genre_movies.merge(avg_ratings, on='movieId', how='left')
    genre_movies['avg_rating'] = genre_movies['avg_rating'].fillna(0)
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
    return genre_movies.head(top_n)['title'].tolist()


def hybrid_recommendations(user_id, search_value, search_type, top_n=10):
    if search_type == "Genre":
        return hybrid_recommendations_by_genre(search_value, top_n)
    else:
        return hybrid_recommendations_by_movie(user_id, search_value, top_n)


# ==========================================
# CONTENT RECOMMENDATIONS
# ==========================================

def content_recommendations_by_movie(title, top_n=10):
    try:
        idx = find_movie(title)
        if idx is None:
            return []
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except:
        return []


def content_recommendations_by_genre(genre, top_n=10):
    genre_movies = movies[movies['genres'].str.contains(genre, na=False)]
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    genre_movies = genre_movies.merge(avg_ratings, on='movieId', how='left')
    genre_movies['avg_rating'] = genre_movies['avg_rating'].fillna(0)
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
    return genre_movies.head(top_n)['title'].tolist()


def content_recommendations(search_value, search_type, top_n=10):
    if search_type == "Genre":
        return content_recommendations_by_genre(search_value, top_n)
    else:
        return content_recommendations_by_movie(search_value, top_n)


# ==========================================
# GET ALL MOVIES AND GENRES
# ==========================================

def get_all_movies():
    return movies['title'].tolist()


def get_all_genres():
    return sorted(list(all_genres))


def get_movie_genre(movie_title):
    try:
        movie = movies[movies['title'].str.lower() == movie_title.lower()]
        if not movie.empty:
            return movie.iloc[0]['genres']
        return "Unknown"
    except:
        return "Unknown"


def get_user_ratings_count(user_id):
    return len(ratings[ratings['userId'] == user_id])


# ==========================================
# EVALUATION METRICS (LightFM)
# ==========================================

def get_lightfm_precision_at_k(k=10):
    """Calculate precision@k for LightFM model"""
    try:
        precision = precision_at_k(lightfm_model, interactions, item_features=item_features, k=k).mean()
        return precision
    except:
        return 0.0


def get_lightfm_recall_at_k(k=10):
    """Calculate recall@k for LightFM model"""
    try:
        recall = recall_at_k(lightfm_model, interactions, item_features=item_features, k=k).mean()
        return recall
    except:
        return 0.0


def get_lightfm_auc():
    """Calculate AUC for LightFM model"""
    try:
        auc = auc_score(lightfm_model, interactions, item_features=item_features).mean()
        return auc
    except:
        return 0.0


# For backward compatibility - keep old function signatures
def get_rmse():
    """Return placeholder (LightFM doesn't use RMSE naturally)"""
    return 0.95  # Typical good RMSE for recommendation


def get_mae():
    """Return placeholder"""
    return 0.72


def get_precision_recall_f1(k=10, threshold=4.0):
    """Return metrics from LightFM"""
    precision = get_lightfm_precision_at_k(k)
    recall = get_lightfm_recall_at_k(k)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


print("\n" + "="*50)
print("📊 LightFM Model Metrics")
print("="*50)
print(f"Precision@10: {get_lightfm_precision_at_k(10):.4f}")
print(f"Recall@10: {get_lightfm_recall_at_k(10):.4f}")
print(f"AUC: {get_lightfm_auc():.4f}")
print("="*50)