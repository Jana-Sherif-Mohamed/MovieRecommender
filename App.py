import streamlit as st
import pandas as pd
import numpy as np
import re
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

# clean genres text
def clean_genres(text):
    text = str(text).lower().replace('|', ' ')
    text = re.sub(r'[^a-z ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# extract year from title
def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    if match:
        return int(match.group(1))
    return np.nan

# clean title by removing year
def clean_title(title):
    title = re.sub(r'\(\d{4}\)', '', str(title))
    return title.strip().lower()

# load data and train model
@st.cache_data
def load_and_train():
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    ratings.drop(columns=['timestamp'], inplace=True)

    movies['genres']      = movies['genres'].apply(clean_genres)
    movies['year']        = movies['title'].apply(extract_year)
    movies['clean_title'] = movies['title'].apply(clean_title)

    # filter popular movies
    rating_count   = ratings.groupby('movieId')['rating'].count()
    popular_movies = rating_count[rating_count >= 20].index
    ratings = ratings[ratings['movieId'].isin(popular_movies)]
    movies  = movies[movies['movieId'].isin(popular_movies)]

    # train SVD model
    reader        = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)
    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    # content-based setup
    movies_content = movies[['movieId', 'title', 'genres']].drop_duplicates().reset_index(drop=True)
    movies_content['genres'] = movies_content['genres'].fillna('')
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(movies_content['genres'])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices      = pd.Series(movies_content.index, index=movies_content['title']).drop_duplicates()

    return movies, ratings, model, movies_content, cosine_sim, indices, predictions

with st.spinner("Loading..."):
    movies, ratings, model, movies_content, cosine_sim, indices, predictions = load_and_train()

st.success("Ready!")

# hybrid recommendation function
def recommend_hybrid(user_id, title, n, cf_w, cb_w):
    all_movies  = ratings['movieId'].unique()
    watched     = ratings[ratings['userId'] == user_id]['movieId'].values
    not_watched = [m for m in all_movies if m not in watched]

    # CF scores normalized 0 to 1
    cf_scores = {m: model.predict(user_id, m).est for m in not_watched}
    cf_min, cf_max = min(cf_scores.values()), max(cf_scores.values())
    cf_norm = {m: (s - cf_min) / (cf_max - cf_min + 1e-9) for m, s in cf_scores.items()}

    # CB scores normalized 0 to 1
    cb_scores = {}
    if title in indices:
        idx = indices[title]
        for i, score in enumerate(cosine_sim[idx]):
            mid = movies_content.iloc[i]['movieId']
            if mid in cf_norm:
                cb_scores[mid] = score

    cb_vals = list(cb_scores.values()) if cb_scores else [0]
    cb_min, cb_max = min(cb_vals), max(cb_vals)
    cb_norm = {m: (s - cb_min) / (cb_max - cb_min + 1e-9) for m, s in cb_scores.items()}

    # combine scores
    hybrid_scores = {
        m: cf_w * cf_norm[m] + cb_w * cb_norm.get(m, 0)
        for m in cf_norm
    }

    top = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]

    result = []
    for mid, score in top:
        info = movies[movies['movieId'] == mid]
        if len(info) == 0:
            continue
        result.append({
            'Title':        info['title'].values[0],
            'Genres':       info['genres'].values[0],
            'Hybrid Score': round(score, 4)
        })

    return pd.DataFrame(result)

# user inputs
user_id     = st.number_input("User ID", min_value=1, max_value=943, value=1, step=1)
movie_title = st.selectbox("Select Movie", sorted(indices.index.tolist()))
n_recs      = st.slider("Number of Recommendations", 5, 20, 10)


# tabs
tab1, tab2 = st.tabs(["Recommendations", "Evaluation"])

with tab1:
    if st.button("Get Recommendations"):
        with st.spinner("Generating..."):
            df = recommend_hybrid(user_id, movie_title, n_recs, cf_w=0.6, cb_w=0.4)
        st.dataframe(df, use_container_width=True)

with tab2:
    actual    = [pred.r_ui for pred in predictions]
    estimated = [pred.est  for pred in predictions]
    rmse = root_mean_squared_error(actual, estimated)
    mae  = mean_absolute_error(actual, estimated)

    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("MAE",  f"{mae:.4f}")