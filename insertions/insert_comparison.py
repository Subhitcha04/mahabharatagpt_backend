import json
from pymongo import MongoClient
import uuid  # Import the uuid module to generate unique IDs

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]
collection = db["api_comparison"]

# Function to insert data into MongoDB
def insert_comparison_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Iterate over each element in the 'queries' array
    for query in data["queries"]:
        # Prepare the document to match the model schema
        comparison_data = {
            "id": str(uuid.uuid4()),  # Generate a unique ID for the document
            "queries": query["queries"],  # Store the queries sub-array for each entry
            "answer": query.get("answer", ""),  # Use the answer field from the query
            "characters": query.get("characters", []),  # Store characters if present
            "comparison": query.get("comparison", ""),  # Store the comparison text if present
            "image": None,  # Image field is left null for now
            "manual_entry": data.get("manual_entry", False)
        }

        # Insert the document into the collection
        collection.insert_one(comparison_data)
        print(f"Inserted comparison with ID: {comparison_data['id']}")

# Provide the path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/comparison.json'

# Call the function to insert data
insert_comparison_data(json_file_path)
