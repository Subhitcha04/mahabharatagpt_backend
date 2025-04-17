import json
from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]

def insert_cultural_events(json_file_path):
    collection = db["api_culturalorhistorical"]

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Access the "cultural_events" key
    for item in data["cultural_events"]:
        document = {
            "id": str(ObjectId()),
            "queries": item["queries"],
            "answers": item["answers"],
            "image": None,
            "manual_entry": item.get("manual_entry", False),
            "culture_event": item["culture_event"]
        }

        existing_entry = collection.find_one({"culture_event": document["culture_event"]})
        if existing_entry:
            print(f"Cultural event '{document['culture_event']}' already exists. Skipping insertion.")
            continue

        collection.insert_one(document)
        print(f"Inserted Cultural Event: {document['culture_event']}")

# Path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/cultural_events.json'
insert_cultural_events(json_file_path)
