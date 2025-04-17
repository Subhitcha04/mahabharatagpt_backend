import re
from pymongo import MongoClient
from bson import ObjectId

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "mahabharata-chatbotdb"
COLLECTION_NAME = "api_storyorevent"

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

def clean_title(title):
    """Remove numeric prefixes (e.g., '2. ') from the title."""
    return re.sub(r'^\d+\.\s*', '', title).strip()

def parse_text_file_and_save_to_mongodb(file_path):
    # Fetch the text content from the file
    lines = fetch_text_file(file_path)
    if lines is None:
        return

    # Initialize MongoDB collection
    collection = connect_to_mongodb()
    inserted_count = 0

    # Start processing from line 16 (story starts here)
    story_data = lines[15:]  # Skip first 15 lines, 16th line onwards

    story_title = None
    story_description = []
    in_story_section = False

    for line in story_data:
        # Skip empty lines
        if not line.strip():
            continue

        # Detect a new chapter or story by looking for markers like "~" or "CHAPTER"
        if re.match(r"^~|\*\*CHAPTER", line):
            if story_title:  # If there's an existing story being processed
                # Clean the title by removing numeric prefixes
                cleaned_title = clean_title(story_title)

                # Save the previous story to MongoDB
                document = {
                    'id': str(ObjectId()),  # Ensure a unique 'id' is generated
                    'title': story_title.strip(),
                    'queries': [
                        f"What is the story of {cleaned_title}?",
                        f"What are the main themes of {cleaned_title}?",
                        f"Who are the key characters in {cleaned_title}?",
                        f"How does {cleaned_title} relate to other events in mythology?",
                        f"What moral lessons can be drawn from {cleaned_title}?",
                        f"What are the significant incidents in {cleaned_title}?",
                        f"How does {cleaned_title} influence cultural beliefs?",
                        f"What are the different interpretations of {cleaned_title}?",
                        f"What are the consequences of events in {cleaned_title}?",
                        f"What rituals are associated with {cleaned_title}?"
                    ],
                    'answers': [' '.join(story_description).strip()],
                    'source': file_path,
                    'manual_entry': False
                }

                # Insert the document into MongoDB
                collection.insert_one(document)
                inserted_count += 1

            # Reset for new story
            story_title = re.sub(r"[\*\*~]", "", line).strip()
            story_description = []
            in_story_section = True
        elif in_story_section:
            # Append the story content to the description
            story_description.append(line.strip())

    # Insert the last story if it exists
    if story_title:
        cleaned_title = clean_title(story_title)
        document = {
            'id': str(ObjectId()),
            'title': story_title.strip(),
            'queries': [
                f"What is the story of {cleaned_title}?",
                f"What are the main themes of {cleaned_title}?",
                f"Who are the key characters in {cleaned_title}?",
                f"How does {cleaned_title} relate to other events in mythology?",
                f"What moral lessons can be drawn from {cleaned_title}?",
                f"What are the significant incidents in {cleaned_title}?",
                f"How does {cleaned_title} influence cultural beliefs?",
                f"What are the different interpretations of {cleaned_title}?",
                f"What are the consequences of events in {cleaned_title}?",
                f"What rituals are associated with {cleaned_title}?"
            ],
            'answers': [' '.join(story_description).strip()],
            'source': file_path,
            'manual_entry': False
        }
        collection.insert_one(document)
        inserted_count += 1

    print(f"{inserted_count} stories were inserted into the database.")

if __name__ == "__main__":
    # Path to the local text file containing the story information
    file_path = r'C:/cit/nlpcat/mythology_chatbot/backend/data/mahabharata.txt'
    parse_text_file_and_save_to_mongodb(file_path)
