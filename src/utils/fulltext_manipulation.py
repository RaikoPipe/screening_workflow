import re

def remove_references_section(fulltext):
    text_split = re.split("References", fulltext, flags=re.IGNORECASE) # should cover most cases
    text_split.pop()
    return "".join(text_split)