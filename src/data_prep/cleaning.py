import re

def mask_pii(text):
    if not text:
        return text
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
    return text

def normalize_whitespace(text):
    if not text:
        return text
    return " ".join(text.split())

def truncate_long_text(text, max_chars=3000):
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars//2] + "\n...[TRUNCATED]...\n" + text[-max_chars//2:]

def clean_text(text):
    text = mask_pii(text)
    text = normalize_whitespace(text)
    text = truncate_long_text(text)
    return text
