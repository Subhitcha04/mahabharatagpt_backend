import json
from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mahabharata-chatbotdb"]

def insert_riddles(json_file_path):
    collection = db["api_riddleorpuzzle"]

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Access the "riddles" key
    for item in data["riddles"]:
        document = {
            "id": str(ObjectId()),
            "queries": item["queries"],
            "answers": item["answers"],
            "image": None,
            "manual_entry": item.get("manual_entry", False),
            "riddle": item["riddle"]
        }

        existing_entry = collection.find_one({"riddle": document["riddle"]})
        if existing_entry:
            print(f"Riddle '{document['riddle']}' already exists. Skipping insertion.")
            continue

        collection.insert_one(document)
        print(f"Inserted Riddle: {document['riddle']}")

# Path to your JSON file
json_file_path = 'C:/cit/nlpcat/mythology_chatbot/backend/data/riddles.json'
insert_riddles(json_file_path)
