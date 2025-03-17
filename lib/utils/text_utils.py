import re

#(?<!\S)              # Assert position is at start-of-string or preceded by whitespace
#(?!\S)                # Assert that the match is followed by whitespace or end-of-string
FAMILY_REGEX_PATTERN = """
                          [\*"\.\n\d+]*                # Optional leading asterisks or quotes
                          (?:
                            (?i:TRIBE|SERIES)   # Match either "Tribe" or "Series"
                            \s+                # Ensure at least one space
                            [IVXLCDM\.]+         # Match Roman numerals (I, V, X, L, C, D, M)
                            \s+                # Ensure at least one space before the family name
                            [A-Z]+((AC|OR)EAE|Æ|(AC|OR)E\.E\.|AE|(ac|or)eae|ORAE|orae|(OR|or)\.(E|e))\.?
                            |
                            [A-Z]+((AC|OR)EAE|Æ|(AC|OR)E\.E\.|AE|(ac|or)eae|ORAE|orae|(OR|or)\.(E|e))\.? # All-uppercase families ending with ACEAE (any number of letters before ACEAE)
                            |
                            [A-Za-z]+((AC|OR)EAE|Æ|(AC|OR)E\.E\.|AE|(ac|or)eae|ORAE|orae|(OR|or)\.(E|e))\.?
                            |
                            [A-Z][a-z]+(ac|or)eae\.?    # Normal mixed-case families ending with 'aceae' (e.g. Celastraceae)
                            |
                            (?=[A-Za-z]*[A-Z])   # Ensure at least one uppercase letter exists in the following synonym
                              (?:
                                [Cc][Oo][Mm][Pp][Oo][Ss][Ii][Tt][Aa][Ee]\.?       |   # Compositae
                                [Cc][Rr][Uu][Cc][Ii][Ff][Ee][Rr][Aa][Ee]\.?       |   # Cruciferae
                                [Gg][Rr][Aa][Mm][Ii][Nn][Ee][Aa][Ee]\.?           |   # Gramineae
                                [Gg][Uu][Tt][Tt][Ii][Ff][Ee][Rr][Aa][Ee]\.?       |   # Guttiferae
                                [Ll][Aa][Bb][Ii][Aa][Tt][Ee][Ee]\.?               |   # Labiatae
                                [Ll][Ee][Gg][Uu][Mm][Ii][Nn][Oo][Ss][Aa][Ee]\.?   |   # Leguminosae
                                [Pp][Aa][Ll][Mm][Aa][Ee]\.?                       |   # Palmae
                                [Uu][Mm][Bb][Ee][Ll][Ll][Ii][Ff][Ee][Rr][Aa][Ee]\.? |   # Umbelliferae
                                [Pp][Aa][Pp][Ii][Ll][Ii][Oo][Nn][Aa][Cc][Ee][Ee]\.?    # Papilionaceae
                              )
                          )
                          [\*"\.\n]*                # Optional trailing asterisks or quotes
                          

"""
FAMILY_REGEX_WITH_LOOKAHEAD = re.compile(rf"(?={FAMILY_REGEX_PATTERN})", re.VERBOSE)
FAMILY_REGEX = re.compile(rf"({FAMILY_REGEX_PATTERN})", re.VERBOSE)

def clean_text(text: str) -> str:

    """
    Perform cleaning on the input extracted text by removing or subsituting any noise
    
    Parameters:
        text (str): Input text to be cleaned
    
    Returns:
        text (str) -> the cleaned text
    """

    result = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.MULTILINE) # Remove any markdown (bold) on strings
    result = re.sub(r"```", "", result, flags=re.MULTILINE) # Remove any markdown
    result = re.sub(r"^Catalogue|catalogue$", "", result, flags=re.MULTILINE) # Remove Catalogue/catalogue
    result = re.sub(r"^[0-9]+$", "", result, flags=re.MULTILINE) # Remove page numbers
    result = re.sub(r"^John Lightfoot$", "", result, flags=re.MULTILINE) # Remove John Lightfoot name if on its own
    result = re.sub(r"\n{3,}", "", result)
    
    return result


def split_division(text: str, 
                   divisions: list[str]=["Dicotyledones", "Monocotyledones", "Pteridophyta", "Hepaticae", "Algae"]
                   ) -> list[tuple[str, str]]:
    """
    Splits the division titles from the extracted text and finds the divisions' text

    Parameters:
        text (str): Input extracted text
        divisions (list[str]): a list of all divisions in catalogue
    
    Returns:
        list[Tuple]:
            1) Division name
            2) Content of divisions
    """

    if divisions is None:
        return [("MAIN", text)]
    
    text = re.sub(rf"^.*?({divisions[0]})", r"\1", text, flags=re.S | re.I)
    division_str = "|".join(divisions)
    regex = re.compile(f"({division_str})", re.IGNORECASE)
    result = re.split(regex, text)

    remove_newline = lambda x: not(re.match(re.compile(r"^(\n)+$"), x))
    result = list(filter(None,result))
    result = list(filter(remove_newline, result))

    splits = list(zip(result[::2], result[1::2]))

    unique_splits = []
    division_hash = set()

    for division, division_text in splits:
        if division not in division_hash:
            division_hash.add(division)
            unique_splits.append((division, division_text))
        else:
            previous_d, previous_text = unique_splits.pop(-1)
            previous_text += division_text
            unique_splits.append((previous_d, previous_text))

    return unique_splits
        


def find_family(text: str) -> list[str]:
    """
    Finds the family names in the input text

    Args:
        text (str): Input text

    Returns:
        list[str]: a list of all family names in input text
    """
    # regex = re.compile("\n+(?=[A-Z ]+\n|.+?[aA][cC][eE][aA][eE])")

    regex = FAMILY_REGEX
    
    result = re.findall(regex, text)

    return list(filter(None,result))


def split_family(text: str, max_chunk_size: int=3000, is_indexed_species: bool=False) -> list[str]:
    """
    Split the input text into chunks for inference.
    This function checks if the big block needs to be seperated.
    If so, it passes the chunk splitting to split_into_smaller_chunks function

    Args:
        text (str): Big block of text from one family
        max_chunk_size (int, optional): The maximum size of a chunk before it needs to be split. Defaults to 3000.
        is_indexed_species (bool): if the species names are indexed with numbers or not

    Returns:
        list[str]: a list of all the split chunks for the family
    """


    # regex = re.compile("\n+(?=[A-Z ]+\n|.+?[aA][cC][eE][aA][eE])")

    regex = FAMILY_REGEX_WITH_LOOKAHEAD
    
    result = re.split(regex, text)
    
    final_list = []

    for family in result:
        
        if family is None or family == '' or family == "\n\n":
            continue
        elif len(family) > max_chunk_size:
            
            small_chunks = split_into_smaller_chunks(family, max_chunk_size, is_indexed_species)
            final_list.extend(small_chunks)
        else:
            final_list.append(family)
    
    return final_list


def split_into_smaller_chunks(large_block: str, max_chunk_size: int=3000, is_indexed_species: bool=False) -> list[str]:
    """
    Split the input text into smaller chunks.

    Args:
        text (str): Big block of text from one family
        max_chunk_size (int, optional): The maximum size of a chunk before it needs to be split. Defaults to 3000.
        is_indexed_species (bool): if the species names are indexed with numbers or not
    Returns:
        list[str]: a list of all the split chunks from the big block
    """

    # Split the large block into family name at the start and the rest
    family_regex = FAMILY_REGEX

    text_block = []

    try:
        family_name, text_block = list(
                                    filter(None, 
                                            re.split(family_regex, large_block)
                                            )
                                    )
    except:
        print("=====")
        print(repr(large_block))
        print("=====")
    #print(f"====={family_name}=====")

    #if len(family_name) < 5:
    #    print(repr(large_block))
    # Define static small chunks
    median_chunk = int(0.5 * max_chunk_size)
    small_chunks = [text_block[i:min(i+median_chunk, len(text_block))] for i in range(0, len(text_block), median_chunk)]
    # Setting cut_off index for next chunk to add to final chunks
    final_chunks = []
    cut_off = 0
    
    index_regex = "(?:^\d+\.\s+)" if is_indexed_species else "^"    
    species_name = re.compile(f"{index_regex}([A-Z][a-zA-Z-]+\s+[A-Za-z][a-zA-Z-]+)(?:,\s*[A-Z](?:\.[A-Z])*\.)")

    # Iterate through all the static chunks
    for ind, small_chunk in enumerate(small_chunks):
        # Define the regex for the species name    
        

        # Split the small chunk by line and then remove the lines that were added to the previous chunk using cut_off value
        small_chunk_splitted = small_chunk.split("\n")
        small_chunk_splitted = small_chunk_splitted[cut_off:]
        small_chunk = "\n".join(small_chunk_splitted)
        
        # Define the next chunk
        next_chunk =  None if ind+1 >= len(small_chunks) else small_chunks[ind+1]
        
        lines_to_add = []

        # If at end of list
        if next_chunk is None:
            # If the size of the small chunk is smaller than 5% of the max chunk size then just add it to the previous chunk
            if len(small_chunk) <= (max_chunk_size * 0.05):
                final_chunks[-1] += "\n" + small_chunk
            # If not add it to final chunks
            else:
                small_chunk = family_name + "\n\n" + small_chunk
                final_chunks.append(small_chunk)
            break
        else:
            # Split the next chunk by line
            next_chunk_splitted = next_chunk.split("\n")

            # For each line in next chunk
            for ind, line in enumerate(next_chunk_splitted):
                # check if the line matches the species name pattern
                if re.match(species_name, line) is None:
                    # if so add the line to the lines_to_add
                    lines_to_add.append(line)
                else:
                    print(re.match(species_name, line))
                    # if not break the loop and update cut_off
                    cut_off = ind
                    break
            print("Entire block added" if len(lines_to_add) == ind+1 else "part of block added")
        print("="*10)
        print(f"Small Chunk: {small_chunk}")
        print(f"Lines to add: {lines_to_add}")
        print("="*10)
        # Join all the lines together and append it to the small chunk
        small_chunk += "\n".join(lines_to_add)
        # Add the family name back
        small_chunk = family_name + "\n\n" + small_chunk
        # Add it to final chunks
        final_chunks.append(small_chunk)
    

    return final_chunks

def convertToTextBlocks(text: str, 
                        divisions: list[str]=["Dicotyledones", "Monocotyledones", "Pteridophyta", "Hepaticae", "Algae"], 
                        max_chunk_size: int=3000,
                        is_indexed_species: bool=False) -> dict:
    """
    Convert the input extracted text into a dictionary of hierarchy: Divisions -> Family Name -> Content chunks

    Args:
        text (str): Extracted text to be split into chunks
        divisions (list[str], optional): A list of division names. Defaults to ["Dicotyledones", "Monocotyledones", "Pteridophyta", "Hepaticae", "Algae"].
        max_chunk_size (int, optional): Maximum chunk size of each small chunk. Defaults to 3000.
        is_indexed_species (bool): If the text contains indexed species names 

    Returns:
        dict: text splitted into a dictionary of hierarchy: Divisions -> Family Name -> Content chunks
    """

    cleaned_text = clean_text(text)

    division_splits = split_division(cleaned_text, divisions)

    splits = {}

    for division, division_text in division_splits:
        check_none = lambda x: not(re.search(r"^[\s\n]*$", x))
        family_split = split_family(division_text, max_chunk_size, is_indexed_species)
        family_split = list(filter(check_none, family_split))
        if division in splits:
            splits[division].extend(family_split)
        else:
            splits[division] = family_split
    
    return splits
