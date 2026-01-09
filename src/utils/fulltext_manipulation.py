import re
from rapidfuzz import fuzz

def remove_section(fulltext, section_title="References"):
    text_split = re.split(section_title, fulltext, flags=re.IGNORECASE) # should cover most cases
    text_split.pop()
    return "".join(text_split)

def omit_sections_markdown(fulltext, omit_sections=None, similarity_threshold=80):
    """
    Omit sections from markdown text using fuzzy matching.

    Args:
        fulltext: Markdown text to filter
        omit_sections: List of section titles to omit
        similarity_threshold: Minimum similarity score (0-100) to consider a match
    """
    if omit_sections is None:
        omit_sections = ["References", "Acknowledgments", "Funding", "Conflict of Interest"]

    # Normalize omit_sections for better matching
    normalized_omit = [s.lower().strip() for s in omit_sections]

    pattern = r'^(#{1,6}) (.+)$'
    sections = re.split(pattern, fulltext, flags=re.MULTILINE)

    filtered_sections = []
    skip_prefix = None
    i = 0

    while i < len(sections):
        section = sections[i]

        # Check if this looks like hash marks (only contains #)
        if re.match(r'^#{1,6}$', section.strip()):
            # Next section should be the heading text
            if i + 1 < len(sections):
                heading_text = sections[i + 1]

                # Clean heading text for matching (remove numbers, special chars)
                cleaned_heading = re.sub(r'^[^\w]*[\d.]+\s*', '', heading_text.strip()).lower()

                # Extract number prefix from heading
                match = re.match(r'^[^\d]*(\d+(?:\.\d+)*)', heading_text.strip())
                current_prefix = match.group(1) if match else None

                # If we're skipping, check if we can stop
                if skip_prefix is not None:
                    if current_prefix and current_prefix.startswith(skip_prefix + '.'):
                        # Skip this heading and its content
                        i += 3 if i + 2 < len(sections) else 2
                        continue
                    else:
                        skip_prefix = None

                # Fuzzy match against omit_sections
                should_omit = False
                for omit_term in normalized_omit:
                    # Use partial ratio for substring matching
                    similarity = fuzz.partial_ratio(omit_term, cleaned_heading)
                    if similarity >= similarity_threshold:
                        should_omit = True
                        break

                if should_omit:
                    skip_prefix = current_prefix
                    # Skip this heading and its content
                    i += 3 if i + 2 < len(sections) else 2
                    continue

                # Keep hash marks and heading
                filtered_sections.append(section)
                filtered_sections.append(heading_text)
                i += 2
                continue

        # Not a heading or we're not skipping - keep the section
        if skip_prefix is None:
            filtered_sections.append(section)

        i += 1

    return "".join(filtered_sections)