import streamlit as st
import pandas as pd
import numpy as np
import re
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#  Page Config
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Movie Recommendation System")
st.caption("Hybrid: Collaborative Filtering + Content-Based Filtering")
st.divider()

# Helper Functions
def clean_genres(text):
    text = str(text).lower().replace('|', ' ')
    text = re.sub(r'[^a-z ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Load Data & Train Model
@st.cache_resource(show_spinner=False)
def setup():
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    ratings.drop(columns=['timestamp'], inplace=True)

    movies['genres'] = movies['genres'].apply(clean_genres)

    # Keep movies with 20+ ratings
    popular = ratings.groupby('movieId')['rating'].count()
    popular = popular[popular >= 20].index
    ratings = ratings[ratings['movieId'].isin(popular)]
    movies  = movies[movies['movieId'].isin(popular)]

    # Train SVD
    reader  = Reader(rating_scale=(1, 5))
    data    = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    train, test = surprise_split(data, test_size=0.2, random_state=42)
    svd     = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(train)
    preds   = svd.test(test)

    # Content-Based
    mc      = movies[['movieId', 'title', 'genres']].drop_duplicates().reset_index(drop=True)
    tfidf   = TfidfVectorizer(stop_words='english').fit_transform(mc['genres'].fillna(''))
    cos_sim = cosine_similarity(tfidf, tfidf)
    idx_map = pd.Series(mc.index, index=mc['title']).drop_duplicates()

    return movies, ratings, svd, mc, cos_sim, idx_map, preds

with st.spinner("Loading model..."):
    movies, ratings, svd_model, mc, cos_sim, idx_map, preds = setup()
st.success("Ready!")
st.divider()

# Hybrid Recommender
def hybrid(user_id, title, n=10, cf_w=0.6, cb_w=0.4):
    all_movies  = ratings['movieId'].unique()
    watched     = ratings[ratings['userId'] == user_id]['movieId'].values
    not_watched = [m for m in all_movies if m not in watched]

    # CF scores
    cf = {m: svd_model.predict(user_id, m).est for m in not_watched}
    lo, hi = min(cf.values()), max(cf.values())
    cf_norm = {m: (s - lo) / (hi - lo + 1e-9) for m, s in cf.items()}

    # CB scores
    cb = {}
    if title in idx_map:
        for i, score in enumerate(cos_sim[idx_map[title]]):
            mid = mc.iloc[i]['movieId']
            if mid in cf_norm:
                cb[mid] = score

    lo2, hi2 = (min(cb.values()), max(cb.values())) if cb else (0, 1)
    cb_norm = {m: (s - lo2) / (hi2 - lo2 + 1e-9) for m, s in cb.items()}

    # Combine
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

# Inputs
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("User ID", min_value=1, max_value=943, value=1)
with col2:
    n_recs  = st.slider("Number of Recommendations", 5, 20, 10)

movie_title = st.selectbox("Pick a movie you like", sorted(idx_map.index.tolist()))

# Tabs
tab1, tab2 = st.tabs(["Recommendations", "Evaluation"])

with tab1:
    if st.button("Get Recommendations", use_container_width=True):
        with st.spinner("Finding movies."):
            df = hybrid(user_id, movie_title, n_recs)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    actual    = [p.r_ui for p in preds]
    estimated = [p.est  for p in preds]
    rmse = np.sqrt(mean_squared_error(actual, estimated))
    mae  = mean_absolute_error(actual, estimated)

    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE",  f"{mae:.4f}")