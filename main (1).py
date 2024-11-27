from dotenv import load_dotenv
import os
import pandas as pd
from transformers import pipeline
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Load the recipe dataset
data_path = os.path.join("data", "recepie_dataset.csv")
recipe_df = pd.read_csv(data_path)

# Initialize the language model
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

def preprocess_recipes(df):
    """Combines title, ingredients, and directions into a single text field for vectorization."""
    df['text'] = df['title'] + " " + df['ingredients'].apply(lambda x: ' '.join(eval(x))) + " " + df['directions']
    df['ingredients_list'] = df['ingredients'].apply(lambda x: set(eval(x)))  # To match ingredients precisely
    return df

# Preprocess the recipes for vectorization
recipe_df = preprocess_recipes(recipe_df)

# Initialize the TF-IDF vectorizer and create embeddings
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(recipe_df['text'])

def retrieve_recipes(query, df, n=3):
    """Retrieve top n recipes based on presence of all ingredients and cosine similarity."""
    # Split the query into individual ingredient words
    ingredients = set(re.findall(r'\b\w+\b', query.lower()))
    
    # Filter recipes to contain all specified ingredients as substrings
    def contains_all_ingredients(ingredient_list):
        return all(any(ingredient in ingr.lower() for ingr in ingredient_list) for ingredient in ingredients)
    
    filtered_df = df[df['ingredients_list'].apply(contains_all_ingredients)]
    
    if filtered_df.empty:
        # Fallback if no exact matches are found
        partial_matches = df[df['ingredients_list'].apply(lambda x: any(ingredient in x for ingredient in ingredients))]
        if not partial_matches.empty:
            return partial_matches.head(n)
        return None  # No partial matches found either

    # Calculate TF-IDF similarity within the filtered recipes
    query_vec = vectorizer.transform([query])
    filtered_tfidf = vectorizer.transform(filtered_df['text'])
    cosine_similarities = cosine_similarity(query_vec, filtered_tfidf).flatten()
    
    # Get top n matches
    top_indices = cosine_similarities.argsort()[-n:][::-1]
    matches = filtered_df.iloc[top_indices]
    
    return matches if not matches.empty else None

def generate_response(recipes):
    """Generates a friendly response with recipe suggestions."""
    if recipes is None or recipes.empty:
        return "I'm sorry, I couldn't find any recipes matching your request. Could you provide more details or try a different query?"

    response = "Here are some recipes you might enjoy:\n\n"
    for _, row in recipes.iterrows():
        response += f"- **{row['title']}**\n"
        response += f"  Ingredients: {', '.join(eval(row['ingredients']))[:200]}...\n"
        response += f"  Directions (partial): {row['directions'][:200]}...\n\n"
    return response

# Main interactive loop for user input
print("Welcome to the Recipe Recommender! You can ask for recipes with specific ingredients, or dietary needs like 'low carb'. Type 'q' to quit.")

while True:
    prompt = input("Enter your query: ")
    if prompt.lower() == "q":
        print("Thank you for using the Recipe Recommender. Goodbye!")
        break
    try:
        # Retrieve and respond to recipes
        retrieved_recipes = retrieve_recipes(prompt, recipe_df)
        response = generate_response(retrieved_recipes)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
