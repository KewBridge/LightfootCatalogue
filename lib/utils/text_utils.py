import re


def clean_text(text):

    result = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.MULTILINE) # Remove any markdown (bold) on strings
    result = re.sub(r"```", "", result, flags=re.MULTILINE) # Remove any markdown
    result = re.sub(r"^Catalogue|catalogue$", "", result, flags=re.MULTILINE) # Remove Catalogue/catalogue
    result = re.sub(r"^[0-9]+$", "", result, flags=re.MULTILINE) # Remove page numbers
    result = re.sub(r"^John Lightfoot$", "", result, flags=re.MULTILINE) # Remove John Lightfoot name if on its own
    result = re.sub(r"\n{3,}", "", result)
    
    return result

def split_division(text, divisions=["Dicotyledones", "Monocotyledone", "Pteridophyta", "Hepaticae", "Algae"]):

    if divisions is None:
        return [("MAIN", text)]
    
    division_str = "|".join(divisions)
    regex = re.compile(f"({division_str})")
    result = re.split(regex, text)

    result = list(filter(None,result))

    return list(zip(result[::2], result[1::2]))

def split_family(text):

    regex = re.compile("\n\n(?=[A-Z ]+\n)")

    result = re.split(regex, text)

    return list(filter(None,result))

def convertToTextBlocks(text, divisions=["Dicotyledones", "Monocotyledone", "Pteridophyta", "Hepaticae", "Algae"]):

    cleaned_text = clean_text(text)

    division_splits = split_division(cleaned_text, divisions)

    splits = {}

    for division, division_text in division_splits:

        if division in splits:
            splits[division].extend(split_family(division_text))
        else:
            splits[division] = split_family(division_text)
    
    return splits