import json
from pymongo import MongoClient
from bson.objectid import ObjectId  # Import ObjectId for unique ids

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]
collection = db["api_objectorartifact"]

# Function to insert object or artifact data into MongoDB
def insert_object_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Iterate over each object in the 'objects' array
    for obj in data["objects"]:
        # Prepare the document to match the model schema
        object_data = {
            "id": str(ObjectId()),  # Generate a unique id using ObjectId
            "object_name": obj["object_name"],
            "queries": obj["queries"],  # Store the queries array for each entry
            "answers": obj["answers"],  # Store the answers array
            "image": None,  # Image field is left null for now; update this as needed
            "manual_entry": obj.get("manual_entry", False)  # Use provided value or default to False
        }

        # Check for existing object_name
        existing_object = collection.find_one({"object_name": object_data["object_name"]})
        if existing_object:
            print(f"Object '{object_data['object_name']}' already exists. Skipping insertion.")
            continue  # Skip to the next iteration if it already exists

        # Insert the document into the collection
        collection.insert_one(object_data)
        print(f"Inserted object: {object_data['object_name']}")

# Provide the path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/objects.json'

# Call the function to insert data
insert_object_data(json_file_path)
