import json
from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]

def insert_prophecy_or_fate(json_file_path):
    collection = db["api_prophecyorfate"]

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Access the "prophecies" key
    for item in data["prophecies"]:
        document = {
            "id": str(ObjectId()),
            "queries": item["queries"],
            "answers": item["answers"],
            "image": None,
            "manual_entry": item.get("manual_entry", False),
            "prophecy_title": item["prophecy_title"]
        }

        existing_entry = collection.find_one({"prophecy_title": document["prophecy_title"]})
        if existing_entry:
            print(f"Prophecy '{document['prophecy_title']}' already exists. Skipping insertion.")
            continue

        collection.insert_one(document)
        print(f"Inserted Prophecy: {document['prophecy_title']}")

# Path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/prophecies.json'
insert_prophecy_or_fate(json_file_path)
