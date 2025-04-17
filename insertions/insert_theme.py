import json
from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]

# Function to insert data into the api_themeormoral collection
def insert_theme_or_moral(json_file_path):
    collection = db["api_themeormoral"]

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    for item in data["themes"]:  # Accessing the list of themes
        document = {
            "id": str(ObjectId()),  # Generate a unique id using ObjectId
            "queries": item["queries"],  # Store the queries array
            "answers": item["answers"],  # Store the answers array
            "image": None,  # Image field is left null for now
            "manual_entry": item.get("manual_entry", False),  # Use provided value or default to False
            "theme": item["theme"]  # Accessing the theme field
        }

        # Check for existing entry
        existing_entry = collection.find_one({"theme": document["theme"]})
        if existing_entry:
            print(f"Theme or Moral '{document['theme']}' already exists. Skipping insertion.")
            continue  # Skip to the next iteration if it already exists

        # Insert the document into the collection
        collection.insert_one(document)
        print(f"Inserted Theme or Moral: {document['theme']}")

# Define the path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/themes.json'
insert_theme_or_moral(json_file_path)
