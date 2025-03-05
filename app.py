import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load IMDb Movie Dataset (From Kaggle CSV)
@st.cache_data
def load_movie_data():
    return pd.read_csv("imdb_movies.csv")  # Make sure this file is in your repo

movies_df = load_movie_data()

# ✅ Function to Get Movie Details
def get_movie_details(movie_name):
    movie = movies_df[movies_df["title"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]  # Return first match
    return None

# ✅ Sentiment Analysis on Reviews
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

# ✅ Recommend Similar Movies
def recommend_movies(movie_title, num_recommendations=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies_df["description"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = movies_df[movies_df["title"].str.contains(movie_title, case=False, na=False)].index
    if not idx.empty:
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        recommended_movies = [movies_df.iloc[i[0]]["title"] for i in sim_scores]
        return recommended_movies
    return []

# ✅ Streamlit UI
st.title("🎬 AI Smart Movie Assistant")
st.write("Search for a movie and get details, reviews, sentiment analysis, and recommendations!")

# ✅ User Input
movie_name = st.text_input("Enter a movie name", "")

if st.button("Search"):
    if movie_name:
        movie_details = get_movie_details(movie_name)

        if movie_details is not None:
            st.subheader("📌 Movie Details")
            st.write(f"**Title:** {movie_details['title']}")
            st.write(f"**Year:** {movie_details['year']}")
            st.write(f"**IMDb Rating:** {movie_details['imdb_rating']}")
            st.write(f"**Cast:** {movie_details['cast']}")
            st.write(f"**Description:** {movie_details['description']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["reviews"])
            st.write(f"**Review Sentiment Score:** {sentiment_score:.2f}")

            # ✅ Recommendations
            similar_movies = recommend_movies(movie_name)
            if similar_movies:
                st.subheader("🎥 Similar Movies")
                st.write(", ".join(similar_movies))
            else:
                st.write("❌ No similar movies found.")

        else:
            st.error("❌ Movie not found! Showing similar movies...")
            similar_movies = recommend_movies(movie_name)
            if similar_movies:
                st.write("🎥 Recommended Similar Movies:")
                st.write(", ".join(similar_movies))
            else:
                st.write("❌ No recommendations available.")

    else:
        st.warning("⚠️ Please enter a movie name.")

