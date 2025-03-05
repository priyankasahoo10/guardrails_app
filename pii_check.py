import re
import spacy

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Regex patterns for structured PII
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
credit_card_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'

def pii_check(text):
    # Rule-based PII detection
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    credit_cards = re.findall(credit_card_pattern, text)
    ssns = re.findall(ssn_pattern, text)

    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    # Check if any PII is detected
    if emails or phones or credit_cards or ssns or names or locations:
        return 'PII Detected', 1
    else:
        return 'No PII Detected', 0 