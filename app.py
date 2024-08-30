import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv('charity_navigator.csv')

# Fill NaN values
df['mission'] = df['mission'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# Encode categorical variables
le_category = LabelEncoder()
le_cause = LabelEncoder()
df['category_encoded'] = le_category.fit_transform(df['category'])
df['cause_encoded'] = le_cause.fit_transform(df['cause'])

# Combine features
df['combined_features'] = df['mission'] + ' ' + df['tagline'] + ' ' + df['category'] + ' ' + df['cause']

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save processed data
df[['charityid', 'category', 'cause', 'mission', 'tagline']].to_csv('processed_charity_data.csv', index=False)

# Save cosine similarity matrix
with open('cosine_sim.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

print("Preprocessing complete. Files 'processed_charity_data.csv' and 'cosine_sim.pkl' have been created.")