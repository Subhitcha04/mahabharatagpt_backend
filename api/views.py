from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.contrib.auth.hashers import check_password, make_password

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import spacy
import logging
import json
import re
import numpy as np
import nltk
import string
import uuid
from collections import defaultdict, Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from textblob import TextBlob

from pymongo import MongoClient
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .serializers import UserSerializer, UserQuerySerializer
from .models import AppUser, UserQuery

from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Spacy model not found. Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define categories for classification
CATEGORIES = [
    "Character", "StoryOrEvent", "SceneOrIncident", "PlaceOrLocation",
    "ObjectOrArtifact", "ThemeOrMoral", "MythologySystem", "MythologyEra",
    "CreatureOrSpecies", "ProphecyOrFate", "Comparison", "CulturalOrHistorical",
    "RiddleOrPuzzle"
]

# Define mapping from categories to collection names
CATEGORY_TO_COLLECTION = {
    "Character": "api_character",
    "StoryOrEvent": "api_storyorevent",
    "SceneOrIncident": "api_sceneorincident",
    "PlaceOrLocation": "api_placeorlocation",
    "ObjectOrArtifact": "api_objectorartifact",
    "ThemeOrMoral": "api_themeormoral",
    "MythologySystem": "api_mythologysystem",
    "MythologyEra": "api_mythologyera",
    "CreatureOrSpecies": "api_creatureorspecies",
    "ProphecyOrFate": "api_prophecyorfate",
    "Comparison": "api_comparison",
    "CulturalOrHistorical": "api_culturalorhistorical",
    "RiddleOrPuzzle": "api_riddleorpuzzle"
}

# Define common question words and their associated intents
QUESTION_WORD_INTENTS = {
    "who": "PERSON",
    "whom": "PERSON",
    "whose": "PERSON_POSSESSION",
    "what": "DEFINITION",
    "which": "SELECTION",
    "when": "TIME",
    "where": "LOCATION",
    "why": "REASON",
    "how": "METHOD"
}

# Common semantic frames for mythology-related queries
SEMANTIC_FRAMES = {
    "origin": ["origin", "originate", "begin", "start", "source", "create", "birth", "born"],
    "power": ["power", "ability", "capable", "strength", "magic", "supernatural", "divine"],
    "relationship": ["relation", "related", "connection", "connected", "family", "marry", "marriage", "spouse", "child", "parent", "sibling"],
    "event": ["event", "happen", "occurred", "battle", "war", "fight", "conflict", "ceremony"],
    "attribute": ["attribute", "quality", "trait", "character", "nature", "property"],
    "comparison": ["compare", "comparison", "versus", "vs", "difference", "similar", "alike", "same", "different"],
    "symbolism": ["symbol", "symbolize", "represent", "meaning", "signify", "stand for"],
    "consequence": ["result", "consequence", "outcome", "effect", "impact", "influence"],
    "purpose": ["purpose", "goal", "aim", "reason", "motive", "intention", "function"]
}

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Connect to MongoDB with error handling
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client.server_info()  # Will raise exception if connection fails
    db = client['mahabharata-chatbotdb']
    logging.info("Successfully connected to MongoDB")
except Exception as e:
    logging.error(f"MongoDB connection error: {str(e)}")
    db = None

# Initialize models
vectorizer = CountVectorizer()
classifier = MultinomialNB()
tfidf_vectorizer = TfidfVectorizer()
word2vec_model = None
fasttext_model = None
query_embeddings_cache = {}

# Track model accuracy for real-time monitoring
model_accuracy = 0.0

# Initialize in-memory query cache for faster matching
query_cache = defaultdict(list)

# Define thresholds
HIGH_RELEVANCE_THRESHOLD = 0.85
CONTEXT_WINDOW_SIZE = 3  # Number of neighboring tokens to consider for context

# Define a simple in-memory knowledge graph for relationship mapping
knowledge_graph = defaultdict(dict)

# Dictionary to store common entity coreferences
coreference_map = {
    "he": None,
    "she": None,
    "it": None,
    "they": None,
    "his": None,
    "her": None,
    "its": None,
    "their": None,
    "him": None,
    "them": None,
    "this": None,
    "that": None,
    "these": None,
    "those": None
}


def initialize_models():
    """Initialize and train all the models"""
    global vectorizer, classifier, tfidf_vectorizer, word2vec_model, fasttext_model, query_cache, model_accuracy
    
    # Fetch data from MongoDB
    data, labels, all_queries = fetch_data(with_queries=True)
    
    if not data or not labels:
        logging.warning("No training data found in MongoDB")
        return
    
    # Train classification model
    model_accuracy = train_classification_model(data, labels)
    
    # Print model accuracy
    logging.info(f"Model accuracy: {model_accuracy * 100:.2f}%")
    print(f"Model accuracy: {model_accuracy * 100:.2f}%")
    
    # Train TF-IDF vectorizer
    if all_queries:
        processed_queries = [preprocess_text(query) for query, _ in all_queries]
        tfidf_vectorizer.fit(processed_queries)
        
        # Build query cache for faster retrieval
        for query, answer in all_queries:
            processed = preprocess_text(query)
            query_cache[processed].append((query, answer))
        
        # Train word embedding models if enough data is available
        if len(processed_queries) > 50:  # Minimum threshold for meaningful embeddings
            train_word_embeddings(processed_queries)
        
        # Build initial knowledge graph
        build_knowledge_graph(all_queries)


def build_knowledge_graph(all_queries):
    """Build a simple knowledge graph from existing query-answer pairs"""
    global knowledge_graph
    
    for query, answer in all_queries:
        doc = nlp(query)
        
        # Extract potential entities and relations
        entities = [ent.text.lower() for ent in doc.ents]
        
        # Add known entities to the knowledge graph
        for entity in entities:
            if entity not in knowledge_graph:
                knowledge_graph[entity] = {
                    "mentions": 1,
                    "associated_answers": [answer],
                    "related_entities": []
                }
            else:
                knowledge_graph[entity]["mentions"] += 1
                if answer not in knowledge_graph[entity]["associated_answers"]:
                    knowledge_graph[entity]["associated_answers"].append(answer)
        
        # Connect entities that appear in the same query
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity2 not in knowledge_graph[entity1]["related_entities"]:
                    knowledge_graph[entity1]["related_entities"].append(entity2)
                if entity1 not in knowledge_graph[entity2]["related_entities"]:
                    knowledge_graph[entity2]["related_entities"].append(entity1)


def train_classification_model(data, labels):
    """Train the Naive Bayes classifier for query categorization"""
    global vectorizer, classifier
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Fit vectorizer and transform data
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    # Train classifier
    classifier.fit(X_train_counts, y_train)
    
    # Evaluate model
    y_pred = classifier.predict(X_test_counts)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Classification model accuracy: {accuracy * 100:.2f}%")
    
    return accuracy


def train_word_embeddings(processed_queries):
    """Train Word2Vec and FastText models on the query corpus"""
    global word2vec_model, fasttext_model
    
    # Tokenize each query for training
    tokenized_queries = [query.split() for query in processed_queries]
    
    # Train Word2Vec model
    try:
        word2vec_model = Word2Vec(
            sentences=tokenized_queries,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4
        )
        logging.info("Word2Vec model trained successfully")
    except Exception as e:
        logging.error(f"Error training Word2Vec model: {str(e)}")
        word2vec_model = None
    
    # Train FastText model
    try:
        fasttext_model = FastText(
            sentences=tokenized_queries,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4
        )
        logging.info("FastText model trained successfully")
    except Exception as e:
        logging.error(f"Error training FastText model: {str(e)}")
        fasttext_model = None


def fetch_data(with_queries=False):
    """Fetches data from the MongoDB collections for training"""
    if db is None:
        logging.error("MongoDB connection not available")
        return [], [], [] if with_queries else [], []
        
    data = []
    labels = []
    all_queries = []
    
    # Get all collections based on category mapping
    collections = list(CATEGORY_TO_COLLECTION.values())

    for idx, collection_name in enumerate(collections):
        try:
            collection = db[collection_name]
            documents = collection.find()

            for doc in documents:
                # Extract queries from the document
                queries = doc.get('queries', [])
                
                # Handle both old and new document structures
                answers = []
                if 'answers' in doc and doc['answers']:
                    answers = doc['answers']
                elif 'answer' in doc and doc['answer']:
                    answers = [doc['answer']]
                
                if answers and len(answers) > 0:
                    answer = answers[0]
                    
                    for query in queries:
                        if query and isinstance(query, str):
                            data.append(query)
                            labels.append(CATEGORIES[idx])
                            
                            if with_queries:
                                all_queries.append((query, answer))
        except Exception as e:
            logging.error(f"Error fetching data from collection {collection_name}: {str(e)}")
    
    if with_queries:
        return data, labels, all_queries
    return data, labels


def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS tag for lemmatization"""
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)


def get_synonyms(word, pos=None):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    
    # If POS is provided, only look for that specific POS
    if pos:
        synsets = wordnet.synsets(word, pos=pos)
    else:
        synsets = wordnet.synsets(word)
        
    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    
    return list(synonyms)[:3]  # Limit to top 3 synonyms to avoid noise


def preprocess_text(text):
    """Enhanced preprocessing: consider context-aware lemmatization and keep important terms"""
    if not isinstance(text, str) or not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Tag parts of speech for better lemmatization
    pos_tags = nltk.pos_tag(tokens)
    
    # Lemmatize with proper POS and keep important words even if they're stopwords
    processed_tokens = []
    important_stopwords = {"who", "what", "when", "where", "why", "how", "which"}
    mythology_stopwords = {"god", "goddess", "myth", "legend", "epic", "hero", "divine", "sacred"}
    
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
        
        # Keep the token if it's not a stopword or if it's an important question word
        if token not in stop_words or token in important_stopwords or token in mythology_stopwords:
            processed_tokens.append(lemma)
    
    return " ".join(processed_tokens)


def expand_query(query):
    """Expand query with relevant synonyms based on context"""
    doc = nlp(query)
    expanded_terms = []
    
    # Get POS tags
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    # Only expand nouns, verbs, and adjectives to avoid introducing noise
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
            # Convert spaCy POS to WordNet POS
            if token.pos_ == 'NOUN':
                wordnet_pos = wordnet.NOUN
            elif token.pos_ == 'VERB':
                wordnet_pos = wordnet.VERB
            elif token.pos_ == 'ADJ':
                wordnet_pos = wordnet.ADJ
            else:
                wordnet_pos = None
                
            # Get contextually appropriate synonyms
            synonyms = get_synonyms(token.text, wordnet_pos)
            
            # Add original term plus synonyms
            expanded_terms.append(token.text)
            expanded_terms.extend(synonyms)
    
    # Return original terms plus expanded terms
    return list(set(expanded_terms))


def extract_query_structure(query):
    """Extract the semantic structure of a query including subject, verb, object relationships"""
    doc = nlp(query)
    query_structure = {
        "subject": None,
        "verb": None,
        "object": None,
        "relations": []
    }
    
    # Extract subject-verb-object relations
    for token in doc:
        # Find the root verb
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            query_structure["verb"] = token.text
            
            # Find subject
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    # Include the full noun phrase
                    subject_phrase = [child.text]
                    for descendant in child.subtree:
                        if descendant != child:
                            subject_phrase.append(descendant.text)
                    query_structure["subject"] = " ".join(subject_phrase)
                
                # Find object
                if child.dep_ in ["dobj", "pobj"]:
                    # Include the full noun phrase
                    object_phrase = [child.text]
                    for descendant in child.subtree:
                        if descendant != child:
                            object_phrase.append(descendant.text)
                    query_structure["object"] = " ".join(object_phrase)
    
    # Extract all dependency relations as triples
    triples = []
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
            subj = token.text
            verb = token.head.text
            obj = None
            
            for child in token.head.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    obj = child.text
                    triples.append((subj, verb, obj))
    
    query_structure["relations"] = triples
    
    return query_structure


def identify_query_intent(query):
    """Identify the intent of the query based on question words and structure"""
    doc = nlp(query.lower())
    
    # Default intent
    intent = "INFORMATION"
    confidence = 0.5
    frame = None
    
    # Check for question words and their associated intents
    for token in doc:
        if token.text in QUESTION_WORD_INTENTS:
            intent = QUESTION_WORD_INTENTS[token.text]
            confidence = 0.7
            break
    
    # Check for semantic frames in mythology domain
    for frame_name, frame_words in SEMANTIC_FRAMES.items():
        for token in doc:
            if token.lemma_.lower() in frame_words:
                frame = frame_name
                confidence = max(confidence, 0.8)
                break
        if frame:
            break
    
    # Adjust intent based on frame if found
    if frame:
        if frame == "origin" and intent == "INFORMATION":
            intent = "ORIGIN"
        elif frame == "power" and intent == "INFORMATION":
            intent = "POWER"
        elif frame == "relationship" and intent == "INFORMATION":
            intent = "RELATIONSHIP"
        elif frame == "event" and intent == "INFORMATION":
            intent = "EVENT"
        elif frame == "comparison" and intent == "INFORMATION":
            intent = "COMPARISON"
    
    # Check for yes/no questions
    if doc[0].pos_ == "AUX" or doc[0].pos_ == "VERB":
        intent = "YES_NO"
        confidence = 0.8
    
    # Check for commands or instructions
    if doc[0].pos_ == "VERB" and doc[0].tag_ == "VB":
        intent = "COMMAND"
        confidence = 0.7
    
    # Check for list requests
    list_indicators = ["list", "name", "enumerate", "tell me all", "what are"]
    for indicator in list_indicators:
        if indicator in query.lower():
            intent = "LIST"
            confidence = 0.8
            break
    
    return {
        "intent": intent,
        "confidence": confidence,
        "frame": frame
    }


def resolve_coreferences(query, previous_query=None, previous_entities=None):
    """Perform simple rule-based coreference resolution"""
    global coreference_map
    
    # Start with the current query
    resolved_query = query
    
    # If no previous entities, just return the original query
    if not previous_entities:
        return resolved_query
    
    doc = nlp(query)
    
    # Look for pronouns that might need resolution
    for token in doc:
        if token.text.lower() in coreference_map and previous_entities:
            # Use the most recent entity that matches the gender if possible
            for entity in previous_entities:
                # Simple gender matching based on known pronouns
                if (token.text.lower() in ["he", "his", "him"] and entity.get("gender") == "MALE") or \
                   (token.text.lower() in ["she", "her", "hers"] and entity.get("gender") == "FEMALE") or \
                   (token.text.lower() in ["it", "its"] and entity.get("gender") == "NEUTRAL") or \
                   (token.text.lower() in ["they", "them", "their", "theirs"]):
                    
                    # Replace the pronoun with the entity name
                    resolved_query = resolved_query.replace(token.text, entity.get("text", ""))
                    break
            
            # If no gender match but we have entities, use the most recent one as fallback
            if token.text in resolved_query and previous_entities:
                resolved_query = resolved_query.replace(token.text, previous_entities[0].get("text", ""))
    
    return resolved_query


def extract_entities(query):
    """Extract entities from the query"""
    doc = nlp(query)
    
    entities = []
    for ent in doc.ents:
        # Assign gender based on entity type (very simplistic approach)
        gender = "NEUTRAL"
        if ent.label_ == "PERSON":
            # Try to guess gender based on first name (extremely simplistic)
            first_token = ent.text.split()[0].lower()
            if first_token in ["he", "his", "him", "mr", "sir", "king", "prince", "lord", "father", "son", "brother"]:
                gender = "MALE"
            elif first_token in ["she", "her", "mrs", "miss", "ms", "queen", "princess", "lady", "mother", "daughter", "sister"]:
                gender = "FEMALE"
        
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "gender": gender
        })
    
    # Add simple noun phrases if they weren't caught as named entities
    for chunk in doc.noun_chunks:
        # Check if this chunk overlaps with any existing entities
        overlaps = False
        for entity in entities:
            if (chunk.start_char >= entity["start"] and chunk.start_char <= entity["end"]) or \
               (chunk.end_char >= entity["start"] and chunk.end_char <= entity["end"]):
                overlaps = True
                break
        
        if not overlaps:
            entities.append({
                "text": chunk.text,
                "label": "NOUN_PHRASE",
                "start": chunk.start_char,
                "end": chunk.end_char,
                "gender": "NEUTRAL"
            })
    
    return entities


def extract_relation_triples(query):
    """Extract subject-verb-object triples from the query"""
    doc = nlp(query)
    triples = []
    
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
            # Extract subject phrase
            subject = [t.text for t in token.subtree if t.dep_ in ["compound", "amod", "det"] or t == token]
            subject = " ".join(subject)
            
            # Get the predicate/verb
            predicate = token.head.text
            
            # Find object(s) connected to this verb
            for child in token.head.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    # Extract object phrase
                    obj = [t.text for t in child.subtree if t.dep_ in ["compound", "amod", "det"] or t == child]
                    obj = " ".join(obj)
                    
                    triples.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj
                    })
    
    return triples


def classify_query(query):
    """Classifies the query using the Naive Bayes classifier."""
    if not isinstance(query, str) or not query:
        return "Unknown"
        
    try:
        # Preprocess the query
        processed_query = preprocess_text(query)
        
        # Transform the query using the vectorizer
        query_transformed = vectorizer.transform([processed_query])
        
        # Predict the category
        category = classifier.predict(query_transformed)[0]
        
        # Get all prediction probabilities
        probabilities = classifier.predict_proba(query_transformed)[0]
        
        # Get the confidence score for the predicted category
        confidence = max(probabilities)
        
        # If confidence is low, do additional checks based on keywords
        if confidence < 0.6:
            # Simple keyword matching for categories
            category_keywords = {
                "Character": ["character", "person", "who", "whom", "king", "queen", "hero", "protagonist", "antagonist", "villain"],
                "StoryOrEvent": ["story", "event", "episode", "narrative", "tale", "happen", "occur", "plot"],
                "PlaceOrLocation": ["place", "location", "where", "kingdom", "city", "mountain", "river", "palace", "temple"],
                "ObjectOrArtifact": ["object", "artifact", "weapon", "item", "tool", "what", "bow", "arrow", "sword"],
                "ThemeOrMoral": ["theme", "moral", "lesson", "meaning", "significance", "message", "value"],
                "MythologySystem": ["mythology", "pantheon", "system", "cosmology", "belief", "tradition"],
                "CreatureOrSpecies": ["creature", "species", "being", "monster", "animal", "divine", "demon"],
                "Comparison": ["compare", "comparison", "versus", "vs", "difference", "similar", "different"]
            }
            
            # Check for category keywords in the query
            for cat, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword in query.lower():
                        return cat
        
        return category
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        return "Unknown"


def analyze_query(query, previous_query=None):
    """Enhanced query analysis with semantic role labeling and intent recognition"""
    if query is None or not isinstance(query, str):
        raise ValueError("The query must be a valid string.")
    
    # Process with spaCy
    doc = nlp(query)
    
    # Original spaCy analysis
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    tokens = [token.text for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]
    noun_chunks = [(chunk.text, chunk.root.text, chunk.root.dep_) for chunk in doc.noun_chunks]
    syntax_tags = [(token.text, token.dep_) for token in doc]

    # Sentiment analysis with TextBlob
    textblob_sentiment = TextBlob(query).sentiment
    sentiment_score = textblob_sentiment.polarity
    
    # Complementary sentiment analysis with VADER
    try:
        sia = SentimentIntensityAnalyzer()
        vader_sentiment = sia.polarity_scores(query)
        compound_score = vader_sentiment['compound']
        
        # Combine both sentiment analyses
        sentiment_label = "positive" if (sentiment_score > 0 and compound_score > 0) else \
                          "negative" if (sentiment_score < 0 and compound_score < 0) else \
                          "neutral"
    except Exception as e:
        logging.warning(f"VADER sentiment analysis failed: {str(e)}")
        sentiment_label = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"

    # Extract detailed entities with gender for coreference resolution
    detailed_entities = extract_entities(query)
    
    # Perform coreference resolution if we have previous queries
    resolved_query = query
    previous_entities = []
    
    if previous_query:
        # Extract entities from previous query
        previous_entities = extract_entities(previous_query)
        
        # Try to resolve coreferences
        resolved_query = resolve_coreferences(query, previous_query, previous_entities)

    # Expanded query terms using synonyms
    expanded_terms = expand_query(query)
    
    # Extract query structure (subject-verb-object)
    query_structure = extract_query_structure(query)
    
    # Identify query intent
    intent_info = identify_query_intent(query)
    
    # Extract relation triples for knowledge graph applications
    relation_triples = extract_relation_triples(query)
    
    # Enhanced analysis with all the new components
    analysis = {
        "entities": entities,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "noun_chunks": noun_chunks,
        "syntax_tags": syntax_tags,
        "sentiment": {
            "label": sentiment_label,
            "textblob_score": float(sentiment_score),
            "vader_score": float(vader_sentiment["compound"]) if 'vader_sentiment' in locals() else None
        },
        "query_understanding": {
            "expanded_terms": expanded_terms,
            "structure": query_structure,
            "intent": intent_info,
            "detailed_entities": detailed_entities,
            "relation_triples": relation_triples
        },
        "coreference_resolution": {
            "original_query": query,
            "resolved_query": resolved_query
        }
    }
    
    return analysis


def get_fuzzy_matches(user_query, max_distance=2):
    """Find potential matches using Levenshtein distance (fuzzy matching)"""
    if db is None:
        logging.error("MongoDB connection not available")
        return []
    
    if not user_query or not isinstance(user_query, str):
        return []
    
    # Preprocess the user query
    processed_query = preprocess_text(user_query)
    tokens = processed_query.split()
    
    # Check the query cache first for exact matches
    if processed_query in query_cache:
        return query_cache[processed_query]
    
    # If no exact match, try fuzzy matching
    fuzzy_matches = []
    
    try:
        collections = list(CATEGORY_TO_COLLECTION.values())

        for collection_name in collections:
            collection = db[collection_name]
            documents = collection.find()

            for doc in documents:
                # Extract queries from the document
                queries = doc.get('queries', [])
                
                # Handle both old and new document structures
                answers = []
                if 'answers' in doc and doc['answers']:
                    answers = doc['answers']
                elif 'answer' in doc and doc['answer']:
                    answers = [doc['answer']]
                
                if answers and len(answers) > 0:
                    answer = answers[0]
                    
                    for query in queries:
                        if not query or not isinstance(query, str):
                            continue
                            
                        # Preprocess the stored query
                        processed_stored = preprocess_text(query)
                        
                        # Check for fuzzy matches on the whole query
                        if edit_distance(processed_query, processed_stored) <= max_distance:
                            fuzzy_matches.append((query, answer))
                            continue
                        
                        # Check for fuzzy matches on individual tokens
                        stored_tokens = processed_stored.split()
                        token_matches = 0
                        
                        for token in tokens:
                            for stored_token in stored_tokens:
                                if edit_distance(token, stored_token) <= 1:
                                    token_matches += 1
                                    break
                        
                        # If a significant portion of tokens match, consider it a match
                        token_match_ratio = token_matches / max(len(tokens), 1)
                        if token_match_ratio >= 0.5 and len(tokens) >= 2:
                            fuzzy_matches.append((query, answer))
                        
        return fuzzy_matches
    except Exception as e:
        logging.error(f"Error in fuzzy matching: {str(e)}")
        return []

def find_matches_tfidf(user_query, all_queries):
    """Find matching queries using TF-IDF vectorization and cosine similarity"""
    if not all_queries:
        return []
    
    if not user_query or not isinstance(user_query, str):
        return []
    
    try:
        # Preprocess the user query
        processed_query = preprocess_text(user_query)
        
        # Get the stored queries
        stored_queries = [preprocess_text(q[0]) for q in all_queries]
        
        # Transform queries into TF-IDF vectors
        query_vectors = tfidf_vectorizer.transform(stored_queries)
        user_vector = tfidf_vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, query_vectors).flatten()
        
        # Find matches above threshold
        threshold = 0.4
        matches = []
        
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                # Convert NumPy float to Python float
                matches.append((all_queries[i], float(sim)))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top matches with their scores
        return matches[:5]
    except Exception as e:
        logging.error(f"Error in TF-IDF matching: {str(e)}")
        return []

def get_embedding(text, model):
    """Get the embedding for a text using a word embedding model"""
    if not text or not model:
        return None
    
    words = preprocess_text(text).split()
    word_vectors = []
    
    for word in words:
        try:
            if word in model.wv:
                word_vectors.append(model.wv[word])
        except:
            continue
    
    if word_vectors:
        # Average the word vectors to get a document vector
        return np.mean(word_vectors, axis=0)
    
    return None

def find_matches_word_embeddings(user_query, all_queries):
    """Find matching queries using word embeddings (Word2Vec or FastText)"""
    if not all_queries or (word2vec_model is None and fasttext_model is None):
        return []
    
    if not user_query or not isinstance(user_query, str):
        return []
    
    try:
        # Choose the model (prefer FastText if available as it handles OOV words better)
        model = fasttext_model if fasttext_model is not None else word2vec_model
        
        # Get user query embedding
        user_embedding = get_embedding(user_query, model)
        
        if user_embedding is None:
            return []
        
        # Calculate similarities with stored queries
        similarities = []
        
        for i, (query, answer) in enumerate(all_queries):
            # Check if we have a cached embedding
            if query in query_embeddings_cache:
                query_embedding = query_embeddings_cache[query]
            else:
                query_embedding = get_embedding(query, model)
                if query_embedding is not None:
                    query_embeddings_cache[query] = query_embedding
            
            if query_embedding is not None:
                # Calculate cosine similarity
                similarity = cosine_similarity([user_embedding], [query_embedding])[0][0]
                # Convert NumPy float to Python float
                similarities.append(((query, answer), float(similarity)))
        
        # Find matches above threshold
        threshold = 0.6
        matches = []
        
        for item, sim in similarities:
            if sim >= threshold:
                matches.append((item, sim))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top matches with their scores
        return matches[:5]
    except Exception as e:
        logging.error(f"Error in word embedding matching: {str(e)}")
        return []

def add_query_to_kb(user_query, best_match_answer, category, similarity_score):
    """Add a highly relevant user query to the knowledge base"""
    if db is None:
        logging.error("MongoDB connection not available")
        return False
    
    if not user_query or not best_match_answer or not category:
        logging.error("Missing parameters for adding query to KB")
        return False
    
    try:
        # Only add queries with very high similarity scores
        if similarity_score < HIGH_RELEVANCE_THRESHOLD:
            logging.info(f"Similarity score {similarity_score} below threshold {HIGH_RELEVANCE_THRESHOLD}")
            return False
        
        # Get collection name from category
        collection_name = CATEGORY_TO_COLLECTION.get(category)
        
        if not collection_name:
            logging.error(f"Unknown category: {category}")
            return False
        
        collection = db[collection_name]
        added = False
        
        # First, search for documents with this specific answer
        # Try both 'answers' array and 'answer' field formats
        answer_doc = collection.find_one({"answers": {"$elemMatch": {"$eq": best_match_answer}}})
        
        # If not found, try the newer format with 'answer' field
        if not answer_doc:
            answer_doc = collection.find_one({"answer": best_match_answer})
        
        if answer_doc:
            # Check if the query already exists
            queries = answer_doc.get('queries', [])
            if user_query in queries:
                logging.info(f"Query already exists in knowledge base: {user_query}")
                return False
            
            # Add the query to the existing document
            result = collection.update_one(
                {"_id": answer_doc["_id"]},
                {"$push": {"queries": user_query}}
            )
            
            if result.modified_count > 0:
                logging.info(f"Added query to existing document: {user_query} (Score: {similarity_score:.4f})")
                added = True
        else:
            # No matching document found, create a new one
            # Determine the document structure based on collection format
            new_doc = {
                "id": str(uuid.uuid4()),  # Generate a unique ID
                "queries": [user_query],
                "manual_entry": False
            }
            
            # Check the collection's document format
            sample_doc = collection.find_one()
            if sample_doc:
                if 'answers' in sample_doc:
                    # Old format with 'answers' array
                    new_doc["answers"] = [best_match_answer]
                else:
                    # New format with single 'answer' field
                    new_doc["answer"] = best_match_answer
                
                # Copy additional fields from the sample document for completeness
                for key in sample_doc:
                    if key not in ['_id', 'id', 'queries', 'answers', 'answer', 'manual_entry']:
                        new_doc[key] = None
            else:
                # No existing documents, use new format
                new_doc["answer"] = best_match_answer
            
            # Insert the new document
            result = collection.insert_one(new_doc)
            if result.inserted_id:
                logging.info(f"Created new document for query: {user_query} (Score: {similarity_score:.4f})")
                added = True
        
        if added:
            # Update the query cache
            processed_query = preprocess_text(user_query)
            query_cache[processed_query].append((user_query, best_match_answer))
            return True
            
        return False
    except Exception as e:
        logging.error(f"Error adding query to knowledge base: {str(e)}")
        return False

def find_best_match(user_query):
    """Find the best matching query using multiple matching strategies"""
    if not user_query or not isinstance(user_query, str):
        return "Please provide a valid query.", "Unknown", 0.0
    
    try:
        # Classify the query to determine the category
        query_category = classify_query(user_query)
        best_match = None
        best_match_similarity = 0.0
        
        # Strategy 1: Try fuzzy matching first (fastest)
        fuzzy_matches = get_fuzzy_matches(user_query)
        
        if fuzzy_matches:
            logging.info(f"Found {len(fuzzy_matches)} fuzzy matches")
            
            # If there's only one fuzzy match, return it directly
            if len(fuzzy_matches) == 1:
                # Assign a moderate similarity score for fuzzy matches
                best_match = fuzzy_matches[0][1]
                best_match_similarity = 0.7
                return best_match, query_category, best_match_similarity
            
            # If there are multiple matches, use TF-IDF to rank them
            tfidf_matches = find_matches_tfidf(user_query, fuzzy_matches)
            
            if tfidf_matches:
                best_match = tfidf_matches[0][0][1]
                best_match_similarity = tfidf_matches[0][1]
                return best_match, query_category, best_match_similarity
            
            # If TF-IDF didn't help, return the first fuzzy match
            best_match = fuzzy_matches[0][1]
            best_match_similarity = 0.7
            return best_match, query_category, best_match_similarity
        
        # Strategy 2: If no fuzzy matches, get all queries and use TF-IDF
        all_queries = get_all_queries()
        
        if not all_queries:
            return "I don't have any information on that topic yet.", query_category, 0.0
        
        tfidf_matches = find_matches_tfidf(user_query, all_queries)
        
        if tfidf_matches:
            logging.info(f"Found {len(tfidf_matches)} TF-IDF matches")
            best_match = tfidf_matches[0][0][1]
            best_match_similarity = tfidf_matches[0][1]
            return best_match, query_category, best_match_similarity
        
        # Strategy 3: Try word embeddings if available
        if word2vec_model is not None or fasttext_model is not None:
            embedding_matches = find_matches_word_embeddings(user_query, all_queries)
            
            if embedding_matches:
                logging.info(f"Found {len(embedding_matches)} embedding matches")
                best_match = embedding_matches[0][0][1]
                best_match_similarity = embedding_matches[0][1]
                return best_match, query_category, best_match_similarity
        
        # If no matches found through any method
        return "I don't have enough information to answer that question yet.", query_category, 0.0
    except Exception as e:
        logging.error(f"Error finding best match: {str(e)}")
        return "I'm having trouble processing your request right now.", "Unknown", 0.0

def get_all_queries():
    """Retrieve all queries and their associated answers from the database"""
    if db is None:
        logging.error("MongoDB connection not available")
        return []
    
    all_queries = []
    
    try:
        collections = list(CATEGORY_TO_COLLECTION.values())

        for collection_name in collections:
            collection = db[collection_name]
            documents = collection.find()

            for doc in documents:
                queries = doc.get('queries', [])
                
                # Handle both old and new document structures
                answers = []
                if 'answers' in doc and doc['answers']:
                    answers = doc['answers']
                elif 'answer' in doc and doc['answer']:
                    answers = [doc['answer']]
                
                if answers and len(answers) > 0:
                    answer = answers[0]
                    
                    for query in queries:
                        if query and isinstance(query, str):
                            all_queries.append((query, answer))
    except Exception as e:
        logging.error(f"Error retrieving all queries: {str(e)}")
    
    return all_queries

def simulate_bot_response(user_query):
    """Generate the chatbot response using the best matching algorithm"""
    if not user_query or not isinstance(user_query, str):
        return "Please provide a valid query.", "Unknown", 0.0
    
    # Find the best match using the hybrid approach
    response, category, similarity_score = find_best_match(user_query)
    
    # Log the match details
    logging.info(f"Query: '{user_query}'")
    logging.info(f"Category: {category}")
    logging.info(f"Best match similarity: {similarity_score:.4f}")
    
    # If the match has very high similarity, add the query to the knowledge base
    if similarity_score >= HIGH_RELEVANCE_THRESHOLD:
        success = add_query_to_kb(user_query, response, category, similarity_score)
        if success:
            logging.info(f"Added high relevance query to KB: {similarity_score:.4f}")
        else:
            logging.info(f"Failed to add query to KB despite high relevance: {similarity_score:.4f}")
    
    return response, category, similarity_score

@require_http_methods(["POST"])
@csrf_exempt
def handle_query(request):
    """Handles POST requests containing a user query."""
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)
        
    try:
        data = json.loads(request.body)
        query = data.get('query')
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON input."}, status=400)

    if not query:
        return JsonResponse({"error": "No query provided. Please enter a valid query."}, status=400)

    try:
        # Perform syntactic and semantic analysis on the query
        analysis = analyze_query(query)
        
        # Get the chatbot response and metadata
        chatbot_response, query_classification, similarity_score = simulate_bot_response(query)

        logging.info(f"Query: {query}")
        logging.info(f"Classification: {query_classification}")
        logging.info(f"Similarity Score: {similarity_score:.4f}")
        
        # Track if the query was added to the knowledge base
        was_added_to_kb = similarity_score >= HIGH_RELEVANCE_THRESHOLD
        
        # Convert all potential NumPy types to Python native types for JSON serialization
        return JsonResponse({
            "analysis": analysis,
            "classification": query_classification,
            "chatbot_response": chatbot_response,
            "similarity_score": float(similarity_score),
            "added_to_kb": bool(was_added_to_kb),
            "model_accuracy": float(model_accuracy * 100)
        })
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return JsonResponse({"error": f"An error occurred while processing the query: {str(e)}"}, status=500)

@csrf_exempt
@api_view(['POST'])
def save_query(request):
    username = request.data.get('username')
    query_text = request.data.get('query')

    if not username or not query_text:
        return Response({'error': 'Username and query are required'}, status=400)

    # Get the user or return 404 if not found
    user = get_object_or_404(AppUser, username=username)

    # Create and save the UserQuery instance
    user_query = UserQuery(user=user, query=query_text)
    user_query.save()

    return Response({'message': 'Query saved successfully'}, status=201)

@api_view(['GET'])
def get_user_queries(request, username):
    """Retrieve queries for a given username"""
    user = get_object_or_404(AppUser, username=username)
    queries = UserQuery.objects.filter(user=user).order_by('-created_at')
    
    response_data = [
        {
            "id": str(q.id),
            "user_id": str(q.user.id),
            "query": q.query,
            "created_at": q.created_at.isoformat()
        }
        for q in queries
    ]
    
    return Response(response_data, status=200)

@api_view(['GET'])
def get_model_statistics(request):
    """Return current model statistics"""
    return Response({
        "model_accuracy": float(model_accuracy * 100),
        "kb_entries": len(query_cache),
        "has_embeddings": word2vec_model is not None or fasttext_model is not None
    }, status=200)

@api_view(['POST'])
def register(request):
    """Registers a new user."""
    serializer = UserSerializer(data=request.data)

    if serializer.is_valid():
        username = serializer.validated_data['username']
        password = serializer.validated_data['password']

        if AppUser.objects.filter(username=username).exists():
            return Response({"error": "Username already exists"}, status=status.HTTP_400_BAD_REQUEST)

        user = AppUser(
            username=username, 
            email=serializer.validated_data.get("email", "")
        )
        user.password = make_password(password)  # Hash the password manually
        user.save()

        return Response({"message": "User registered successfully"}, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
@api_view(['POST'])
def login_view(request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({"success": False, "message": "Username and password are required"}, status=400)

    user = get_object_or_404(AppUser, username=username)

    if check_password(password, user.password):  # Compare hashed password
        return Response({"success": True, "username": user.username}, status=200)

    return Response({"success": False, "message": "Invalid credentials"}, status=401)

@api_view(['GET'])
def get_logged_in_username(request):
    """Returns the username of the logged-in user."""
    return Response({"username": request.user.username}, status=200)

@ensure_csrf_cookie
def home_view(request):
    """Renders the home.html template and ensures CSRF token is set."""
    return render(request, 'home.html')

# Initialize models when the module is loaded
initialize_models()