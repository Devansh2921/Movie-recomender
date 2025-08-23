from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from rapidfuzz import process
from difflib import get_close_matches
import requests

# Load saved objects
content_sim = np.load("content_sim.npy")
df = pd.read_pickle("movies.pkl")

def get_poster(title):
    api_key = "62936bb9"
    url = f"http://www.omdbapi.com/?s={title}&apikey={api_key}"
    data = requests.get(url).json()
    
    print(title, "->", data)  # debug: see raw API response
    
    if data.get("Search"):
        poster = data["Search"][0].get("Poster")
        if poster and poster != "N/A":
            return poster
    return "/static/no_poster.png"

# ---- Your functions ----
def get_best_match(movie_title, df):
    titles = df['title'].tolist()
    match = process.extractOne(movie_title, titles)  # (match, score, index)
    if match:
        matched_title, score, idx = match
        if score > 70:  # confidence threshold
            return matched_title
    return None

def recommend_content(movie_title, df, content_sim, top_n=5):
    # Try RapidFuzz first
    matched_title = get_best_match(movie_title, df)

    # If no good match → fallback to Difflib
    if not matched_title:
        titles = df['title'].tolist()
        best_match = get_close_matches(movie_title, titles, n=1, cutoff=0.6)
        if best_match:
            matched_title = best_match[0]
        else:
            return None

    # Find index of matched title
    idx = df[df['title'] == matched_title].index[0]

    # Compute similarity scores
    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip self

    # Get recommendations
    movie_indices = [i[0] for i in sim_scores]
    recs = df.iloc[movie_indices][['title', 'genre', 'rating', 'description']]
    recs['poster_url'] = recs['title'].apply(get_poster)
    return matched_title, recs.to_dict(orient="records")

# ---- Flask routes ----
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    matched_title = None
    query = ""

    if request.method == "POST":
        query = request.form["movie"]
        result = recommend_content(query, df, content_sim, top_n=5)
        if result:
            matched_title, recommendations = result

    # Debug: print recommendations to see what's being passed
    if recommendations:
        print("Debug - First recommendation:", recommendations[0])
    
    return render_template("index.html", query=query, matched_title=matched_title, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
