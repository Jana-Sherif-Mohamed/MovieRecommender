import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

# Page Config
st.title("Movie Recommendation System")

def clean_genres(text):
    text = str(text).lower().replace('|', ' ')
    text = re.sub(r'[^a-z ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Load & Train
@st.cache_resource(show_spinner=False)
def setup():
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    if 'timestamp' in ratings.columns:
        ratings.drop(columns=['timestamp'], inplace=True)

    movies['genres'] = movies['genres'].apply(clean_genres)

    # Keep movies with 20+ ratings
    popular = ratings.groupby('movieId')['rating'].count()
    popular = popular[popular >= 20].index
    ratings = ratings[ratings['movieId'].isin(popular)]
    movies  = movies[movies['movieId'].isin(popular)]

    # Train / Test split
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    # Build user-item matrix from TRAIN only
    all_users  = sorted(ratings['userId'].unique())
    all_movies = sorted(ratings['movieId'].unique())

    user_idx  = {u: i for i, u in enumerate(all_users)}
    movie_idx = {m: i for i, m in enumerate(all_movies)}

    R = np.zeros((len(all_users), len(all_movies)))
    for row in train_df.itertuples():
        R[user_idx[row.userId], movie_idx[row.movieId]] = row.rating

    # Mean-center
    user_mean = np.true_divide(R.sum(1), (R != 0).sum(1).clip(1))
    R_centered = R.copy()
    for i in range(R.shape[0]):
        mask = R[i] != 0
        R_centered[i, mask] -= user_mean[i]

    # SVD  (k=50 factors)
    k = min(50, min(R.shape) - 1)
    U, sigma, Vt = svds(R_centered, k=k)
    sigma_diag   = np.diag(sigma)
    pred_matrix  = user_mean[:, np.newaxis] + U @ sigma_diag @ Vt

    # Evaluate on test set
    actual, estimated = [], []
    for row in test_df.itertuples():
        if row.userId in user_idx and row.movieId in movie_idx:
            pred = pred_matrix[user_idx[row.userId], movie_idx[row.movieId]]
            pred = float(np.clip(pred, 1, 5))
            actual.append(row.rating)
            estimated.append(pred)

    rmse = float(np.sqrt(mean_squared_error(actual, estimated)))
    mae  = float(mean_absolute_error(actual, estimated))

    # Content-Based setup
    mc      = movies[['movieId', 'title', 'genres']].drop_duplicates().reset_index(drop=True)
    tfidf   = TfidfVectorizer(stop_words='english').fit_transform(mc['genres'].fillna(''))
    cos_sim = cosine_similarity(tfidf, tfidf)
    idx_map = pd.Series(mc.index, index=mc['title']).drop_duplicates()

    return (movies, ratings, pred_matrix,
            user_idx, movie_idx, user_mean,
            mc, cos_sim, idx_map,
            rmse, mae)

with st.spinner():
    (movies, ratings, pred_matrix,
     user_idx, movie_idx, user_mean,
     mc, cos_sim, idx_map,
     rmse, mae) = setup()

st.divider()

# Hybrid Recommender
def hybrid(user_id, title, n=10, cf_w=0.6, cb_w=0.4):
    # Movies this user hasn't rated
    watched     = set(ratings[ratings['userId'] == user_id]['movieId'].values)
    not_watched = [m for m in movie_idx if m not in watched]

    # CF scores
    if user_id in user_idx:
        ui = user_idx[user_id]
        cf = {m: float(np.clip(pred_matrix[ui, movie_idx[m]], 1, 5))
              for m in not_watched if m in movie_idx}
    else:
        # cold-start: use global mean
        global_mean = ratings['rating'].mean()
        cf = {m: global_mean for m in not_watched}

    if not cf:
        return pd.DataFrame()

    lo, hi  = min(cf.values()), max(cf.values())
    cf_norm = {m: (s - lo) / (hi - lo + 1e-9) for m, s in cf.items()}

    # CB scores
    cb = {}
    if title in idx_map:
        for i, score in enumerate(cos_sim[idx_map[title]]):
            mid = mc.iloc[i]['movieId']
            if mid in cf_norm:
                cb[mid] = float(score)

    if cb:
        lo2, hi2 = min(cb.values()), max(cb.values())
        cb_norm  = {m: (s - lo2) / (hi2 - lo2 + 1e-9) for m, s in cb.items()}
    else:
        cb_norm = {}

    scores = {m: cf_w * cf_norm[m] + cb_w * cb_norm.get(m, 0) for m in cf_norm}
    top    = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    rows = []
    for mid, score in top:
        row = movies[movies['movieId'] == mid]
        if row.empty:
            continue
        rows.append({
            'Title':  row['title'].values[0],
            'Genres': row['genres'].values[0],
            'Score':  round(score, 3)
        })
    return pd.DataFrame(rows)

# UI
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("User ID", min_value=1, max_value=943, value=1)
with col2:
    n_recs = st.slider("Number of Recommendations", 5, 20, 10)

movie_title = st.selectbox("Pick a movie you like", sorted(idx_map.index.tolist()))

tab1, tab2 = st.tabs(["Recommendations", "Evaluation"])

with tab1:
    if st.button("Get Recommendations", use_container_width=True):
        with st.spinner("Finding movies..."):
            df = hybrid(int(user_id), movie_title, n_recs)
        if df.empty:
            st.warning("No recommendations found.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE",  f"{mae:.4f}")