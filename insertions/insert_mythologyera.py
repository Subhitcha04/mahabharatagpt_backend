import json
from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]

def insert_mythology_era(json_file_path):
    collection = db["api_mythologyera"]

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Access the "mythology_eras" key
    for item in data["mythology_eras"]:
        document = {
            "id": str(ObjectId()),
            "queries": item["queries"],
            "answers": item["answers"],
            "image": None,
            "manual_entry": item.get("manual_entry", False),
            "era_name": item["era_name"]
        }

        existing_entry = collection.find_one({"era_name": document["era_name"]})
        if existing_entry:
            print(f"Mythology Era '{document['era_name']}' already exists. Skipping insertion.")
            continue

        collection.insert_one(document)
        print(f"Inserted Mythology Era: {document['era_name']}")

# Path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/mythology_eras.json'
insert_mythology_era(json_file_path)
