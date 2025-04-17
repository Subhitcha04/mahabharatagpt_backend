import json
from pymongo import MongoClient
from bson.objectid import ObjectId  # Import ObjectId for unique ids

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]
collection = db["api_creatureorspecies"]

# Function to insert creature or species data into MongoDB
def insert_creature_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Iterate over each creature in the 'creatures' array
    for creature in data["creatures"]:
        # Prepare the document to match the model schema
        creature_data = {
            "id": str(ObjectId()),  # Generate a unique id using ObjectId
            "species_name": creature["species"],
            "queries": creature["queries"],  # Store the queries array for each entry
            "answers": creature.get("answer", ""),  # Use the answer field from the creature
            "image": None,  # Image field is left null for now
            "manual_entry": data.get("manual_entry", False)  # Manual entry field, if needed
        }

        # Check for existing species_name
        existing_creature = collection.find_one({"species_name": creature_data["species_name"]})
        if existing_creature:
            print(f"Creature '{creature_data['species_name']}' already exists. Skipping insertion.")
            continue  # Skip to the next iteration if it already exists

        # Insert the document into the collection
        collection.insert_one(creature_data)
        print(f"Inserted creature: {creature_data['species_name']}")

# Provide the path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/creatures.json'

# Call the function to insert data
insert_creature_data(json_file_path)
