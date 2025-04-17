from pymongo import MongoClient
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client['mahabharata-chatbotdb']

# List of collections corresponding to categories
collections = [
    'api_character', 'api_comparison', 'api_creatureorspecies', 'api_culturalorhistorical',
    'api_mythologyera', 'api_mythologysystem', 'api_objectorartifact', 'api_placeorlocation',
    'api_prophecyorfate', 'api_riddleorpuzzle', 'api_sceneorincident', 'api_storyorevent', 'api_themeormoral'
]

def get_all_queries(user_query):
    """Retrieve all queries and their associated answers using regex matching for the user's query."""
    all_queries = []
    # Updated regex pattern for partial matches (allows for any text before and after)
    pattern = re.compile(re.escape(user_query), re.IGNORECASE)
    
    print(f"Searching for queries matching the pattern: {pattern}")  # Debugging line

    for collection_name in collections:
        collection = db[collection_name]

        # Search for documents where at least one query matches the regex pattern
        documents = collection.find({'queries': {'$elemMatch': {'$regex': pattern}}})
        
        print(f"Checking collection: {collection_name}")  # Debugging line

        for doc in documents:
            queries = doc.get('queries', [])
            print(f"Found document: {doc}")  # Debugging line

            # Ensure 'answers' is a list
            answers = doc.get('answers', [])
            answer = None  # Default answer to None
            
            if isinstance(answers, list) and len(answers) > 0:
                answer = answers[0]  # Get the first answer directly
            
            if answer:  # Check if there's an answer available
                for query in queries:
                    all_queries.append((query, answer, doc['_id']))  # Store query and corresponding answer

    print(f"Total queries retrieved using regex: {len(all_queries)}")  # Debugging line
    
    return all_queries

def find_best_match(user_query, all_queries):
    """Find the best matching query using Cosine Similarity."""
    
    # Check if there are queries to match
    if not all_queries:
        return "No queries found in the database."
    
    # Extract queries from the dataset
    queries = [q[0] for q in all_queries]
    
    # Combine user query with dataset queries for vectorization
    all_texts = queries + [user_query]
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()
    
    user_vector = vectors[-1].reshape(1, -1)
    
    # Compute cosine similarity between the user query and the dataset queries
    similarities = cosine_similarity(user_vector, vectors[:-1]).flatten()
    
    # Find the best match based on the highest similarity score
    best_match_idx = np.argmax(similarities)
    
    return all_queries[best_match_idx]

def simulatebotresponse(user_query):
    """Simulate the chatbot response using regex-based search and cosine similarity."""
    
    # Get all queries from the database using regex search
    all_queries = get_all_queries(user_query)
    
    # Find the best match for the user query
    best_match = find_best_match(user_query, all_queries)
    
    # Return the corresponding answer
    if isinstance(best_match, str):
        # If best_match is a string, return the error message
        return best_match
    else:
        matched_query, answer, doc_id = best_match
        return answer  # Return only the answer

# Example Usage:
user_query = "who is arjuna?"
response = simulatebotresponse(user_query)
print(response)
