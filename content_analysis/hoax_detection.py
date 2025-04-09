def is_hoax_related(text, keywords=None):
    if keywords is None:
        keywords = ['hoax', 'bomb threat', 'fake call', 'false alarm']
    return any(kw in text.lower() for kw in keywords)

