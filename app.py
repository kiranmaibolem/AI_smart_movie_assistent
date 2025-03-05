import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load IMDb Movie Dataset
@st.cache_data
def load_movie_data():
    return pd.read_csv("imdb_movies.csv")

movies_df = load_movie_data()

# ✅ Function to Get Movie Details
def get_movie_details(movie_name):
    movie = movies_df[movies_df["names"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]  # Return first match
    return None

# ✅ Sentiment Analysis on Movie Overview
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if pd.notna(text) else 0

# ✅ Recommend Similar Movies if No Exact Match
def recommend_movies(movie_title, num_recommendations=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies_df["overview"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = movies_df[movies_df["names"].str.contains(movie_title, case=False, na=False)].index
    if not idx.empty:
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        recommended_movies = [movies_df.iloc[i[0]]["names"] for i in sim_scores]
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
            st.write(f"**Title:** {movie_details['names']}")
            st.write(f"**Release Date:** {movie_details['date_x']}")
            st.write(f"**IMDb Score:** {movie_details['score']}")
            st.write(f"**Genre:** {movie_details['genre']}")
            st.write(f"**Overview:** {movie_details['overview']}")
            st.write(f"**Crew:** {movie_details['crew']}")
            st.write(f"**Original Title:** {movie_details['orig_title']}")
            st.write(f"**Status:** {movie_details['status']}")
            st.write(f"**Original Language:** {movie_details['orig_lang']}")
            st.write(f"**Budget:** {movie_details['budget_x']}")
            st.write(f"**Revenue:** {movie_details['revenue']}")
            st.write(f"**Country:** {movie_details['country']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["overview"])
            st.write(f"**Overview Sentiment Score:** {sentiment_score:.2f}")

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
                st.subheader("🎥 Recommended Similar Movies:")
                st.write(", ".join(similar_movies))
            else:
                st.write("❌ No recommendations available.")

    else:
        st.warning("⚠️ Please enter a movie name.")
