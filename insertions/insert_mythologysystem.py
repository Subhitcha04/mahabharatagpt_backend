import json
from pymongo import MongoClient
from bson.objectid import ObjectId  # Import ObjectId for unique ids

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]
collection = db["api_mythologysystem"]  # Updated the collection name

# Function to insert mythology systems into MongoDB
def insert_mythology_system(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Iterate over each mythology system in the 'mythology_systems' array
    for system in data["mythology_systems"]:
        # Prepare the document to match the model schema
        mythology_system_data = {
            "id": str(ObjectId()),  # Generate a unique id using ObjectId
            "system_name": system["system_name"],
            "queries": system["queries"],  # Store the queries array
            "answers": system["answers"],  # Store the answers array
            "image": system["image"],  # Store the image (could be null)
            "manual_entry": system.get("manual_entry", False)  # Default to False if not provided
        }

        # Check for existing system_name
        existing_system = collection.find_one({"system_name": mythology_system_data["system_name"]})
        if existing_system:
            print(f"Mythology system '{mythology_system_data['system_name']}' already exists. Skipping insertion.")
            continue  # Skip to the next iteration if it already exists

        # Insert the document into the collection
        collection.insert_one(mythology_system_data)
        print(f"Inserted mythology system: {mythology_system_data['system_name']}")

# Provide the path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/mythology_systems.json'

# Call the function to insert data
insert_mythology_system(json_file_path)
