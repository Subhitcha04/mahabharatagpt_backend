import json
from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]

def insert_scenes(json_file_path):
    collection = db["api_sceneorincident"]

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Access the "scenes_or_incidents" key
    for item in data["scenes_or_incidents"]:
        document = {
            "id": str(ObjectId()),
            "scene_title": item["scene_title"],
            "queries": item["queries"],
            "answers": item["answers"],
            "image": None,  # Set to None, or handle it if you have image data.
            "manual_entry": item.get("manual_entry", False)
        }

        existing_entry = collection.find_one({"scene_title": document["scene_title"]})
        if existing_entry:
            print(f"Scene '{document['scene_title']}' already exists. Skipping insertion.")
            continue

        collection.insert_one(document)
        print(f"Inserted Scene: {document['scene_title']}")

# Path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/scenes.json'
insert_scenes(json_file_path)
