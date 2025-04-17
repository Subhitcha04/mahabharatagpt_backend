import requests
from pymongo import MongoClient
from bson import ObjectId

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "mahabharata-chatbotdb"
COLLECTION_NAME = "api_character"

def connect_to_mongodb():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection

def fetch_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except Exception as e:
        print(f"Failed to read the file: {e}")
        return None

def parse_text_file_and_save_to_mongodb(file_path):
    # Fetch the text content from the file
    lines = fetch_text_file(file_path)
    if lines is None:
        return

    # Initialize MongoDB collection
    collection = connect_to_mongodb()
    inserted_count = 0

    # Start processing from line 1480 (glossary starts here)
    for line in lines[1480:]:
        # Skip empty lines and lines that don't contain character data
        if not line.strip() or not line[0].isalpha() or '.' not in line:
            continue

        # Split the line into name and description based on the first period
        try:
            name, description = line.split('.', 1)
            name = name.strip()
            description = description.strip()

            # Check for duplicate name before inserting
            existing_entry = collection.find_one({'name': name})
            if existing_entry:
                print(f"Duplicate entry found for character '{name}'. Skipping insertion.")
                continue

            # Prepare the document to insert into MongoDB
            document = {
                'id': str(ObjectId()),  # Ensure a unique 'id' is generated
                'name': name,
                'queries': [
                    f"What is the role of {name} in Mahabharata?",
                    f"Can you tell me about {name}?",
                    f"Who is {name}?",
                    f"What are the key events involving {name}?",
                    f"What is the significance of {name} in the story?"
                ],
                'answers': [description],
                'image': None,  # Placeholder for image URL
                'manual_entry': False
            }

            # Insert the document into MongoDB
            collection.insert_one(document)
            inserted_count += 1

        except ValueError as ve:
            print(f"Skipping entry due to error: {ve}")
            continue

    print(f"{inserted_count} documents were inserted into the database.")

if __name__ == "__main__":
    # Path to the local text file containing character information
    file_path = r'C:/cit/nlpcat/mythology_chatbot/backend/data/mahabharata.txt'
    parse_text_file_and_save_to_mongodb(file_path)
