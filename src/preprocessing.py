import re
import html
import unicodedata

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag


# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

negation_words = {
    "not", "no", "nor", "never",
    "cannot", "without", "neither"
}


# 1. Text cleaning and normalization
def clean_text(text):

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.lower()
    text = text.replace("’", "'")

    # Remove HTML tags and URLs
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Expand contractions
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'m\b", " am", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'ll\b", " will", text)

    # Remove punctuation and extra whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 2. Word tokenization
def tokenize_text(text):
    return word_tokenize(clean_text(text))


# 3. Sentence tokenization
def tokenize_sentences(text):
    return sent_tokenize(str(text))


# 4. Stopword removal
def remove_stopwords(tokens):

    stop_words = set(stopwords.words("english"))
    stop_words -= negation_words

    return [
        word for word in tokens
        if word not in stop_words
    ]


# 5. Stemming
def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]


# 6. Convert POS tags to WordNet format
def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


# 7. Lemmatization
def apply_lemmatization(tokens):

    tagged_tokens = pos_tag(tokens)

    return [
        lemmatizer.lemmatize(
            word,
            get_wordnet_pos(tag)
        )
        for word, tag in tagged_tokens
    ]


# 8. Negation handling
def handle_negation(tokens):

    result = []
    remaining = 0

    for word in tokens:

        if word in negation_words:
            result.append(word)
            remaining = 2

        elif word in {"but", "however", "yet"}:
            remaining = 0
            result.append(word)

        elif remaining > 0:
            result.append("NOT_" + word)
            remaining -= 1

        else:
            result.append(word)

    return result


# 9. Machine learning preprocessing
def normalize_ml_text(text):

    text = re.sub(
        r"\{\{[^{}]+\}\}",
        " placeholder ",
        str(text)
    )

    text = re.sub(
        r"\d+",
        " numbertoken ",
        text
    )

    return clean_text(text)


# 10. Keyword extraction preprocessing
def preprocess_query(text):

    tokens = tokenize_text(text)
    tokens = apply_lemmatization(tokens)
    tokens = remove_stopwords(tokens)

    return " ".join(tokens)


# 11. Domain-specific entity extraction
def extract_domain_entities(text):

    entity_patterns = {
        "ORDER_ID": r"\b(?:order|purchase)\s*(?:number|id|no\.?)?\s*#?\s*(\d+)\b",

        "INVOICE_ID": r"\b(?:invoice|bill)\s*(?:number|id|no\.?)?\s*#?\s*(\d+)\b",

        "ACCOUNT_ID": r"\baccount\s*(?:number|id|no\.?)?\s*#?\s*(\d+)\b",

        "TRACKING_ID": r"\btracking\s*(?:number|id|no\.?)?\s*#?\s*(\d+)\b",

        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

        "MONEY": r"(?:₹|\$|€|£|Rs\.?)\s*\d+(?:,\d{3})*(?:\.\d{1,2})?",

        "PLACEHOLDER": r"\{\{[^{}]+\}\}"
    }

    entities = []

    for label, pattern in entity_patterns.items():

        matches = re.finditer(
            pattern,
            str(text),
            flags=re.IGNORECASE
        )

        for match in matches:
            entities.append((match.group(), label))

    return entities