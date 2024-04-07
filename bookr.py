from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Initialise Flask App
app = Flask(__name__)

df = pd.read_csv("books.csv")  

# Keep only the necessary columns and drop rows with missing values.
df = df[['title', 'author', 'rating', 'price', 'description', 'coverImg', 'genres']].dropna()


# We combine the title and description into a single column and process the text.
df['title_description'] = df['title'] + ' ' + df['description']
df['title_description'] = df['title_description'].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x.lower()))

# Convert the genres column from string representation of list to actual list, then to a space-separated string.
df['genres'] = df['genres'].apply(ast.literal_eval)
df['genres_str'] = df['genres'].apply(lambda x: ' '.join([genre.lower() for genre in x]))

# Combine the title_description and genres_str into a single column for the TF-IDF vectorizer.
df['combined_features'] = df['title_description'] + ' ' + df['genres_str']

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the combined features of the books and transform the text into numerical data.
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])


@app.route('/')
def quiz():
    return render_template('quiz.html')

@app.route('/recommender', methods=['POST'])
def get_recommendations():
    # Extract the selected genres and plot keywords from the form submission.
    selected_genres = request.form.getlist('genre')
    plot_keywords = request.form['plot']
    # Combine the genres and plot keywords into a single query.
    combined_query = ' '.join(selected_genres) + ' ' + plot_keywords
    # Get book recommendations based on the combined query.
    recommendations = recommend_books(combined_query)
    # Serve the recommendations.html template, passing the recommendations to it.
    return render_template('recommendations.html', recommendations=recommendations)

def recommend_books(query):
    # Process the query to remove any special characters and convert it to lowercase.
    processed_query = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    # Transform the query using the TF-IDF vectorizer.
    query_vec = vectorizer.transform([processed_query])
    # Compute the cosine similarity between the query and all books.
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Get the indices of the books with the highest similarity scores.
    indices = np.argpartition(similarity, -10)[-20:]
    # Sort the books by rating in descending order.
    results = df.iloc[indices].sort_values("rating", ascending=False)
    # Convert the results into a list of dictionaries to pass to the template.
    recommendations = results.to_dict('records')
    return recommendations

if __name__ == '__main__':
    app.run(port=6100)
