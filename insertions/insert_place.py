import json
from pymongo import MongoClient
from bson.objectid import ObjectId  # Import ObjectId for unique ids

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]
collection = db["api_placeorlocation"]

# Function to insert place or location data into MongoDB
def insert_place_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Iterate over each place in the 'places' array
    for place in data["places"]:
        # Prepare the document to match the model schema
        place_data = {
            "id": str(ObjectId()),  # Generate a unique id using ObjectId
            "place_name": place["place_name"],
            "queries": place["queries"],  # Store the queries array for each entry
            "answers": place["answers"],  # Store the answers string
            "image": None,  # Image field is left null for now; update this as needed
            "manual_entry": False  # Defaulting manual_entry to False
        }

        # Check for existing place_name
        existing_place = collection.find_one({"place_name": place_data["place_name"]})
        if existing_place:
            print(f"Place '{place_data['place_name']}' already exists. Skipping insertion.")
            continue  # Skip to the next iteration if it already exists

        # Insert the document into the collection
        collection.insert_one(place_data)
        print(f"Inserted place: {place_data['place_name']}")

# Provide the path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/places.json'

# Call the function to insert data
insert_place_data(json_file_path)
