import re

def remove_section(fulltext, section_title="References"):
    text_split = re.split(section_title, fulltext, flags=re.IGNORECASE) # should cover most cases
    text_split.pop()
    return "".join(text_split)