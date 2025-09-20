import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_text(text):
    doc = nlp(text or "")
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def extract_text_features(data):
    processed = [preprocess_text(desc) for desc in data["description"].fillna("")]
    vectorizer = CountVectorizer(max_features=5000)
    text_features = vectorizer.fit_transform(processed).toarray()
    return text_features, vectorizer
