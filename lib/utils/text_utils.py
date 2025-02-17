import re


FAMILY_REGEX = re.compile(r"""
                          (?=(?<!\S)              # Assert position is at start-of-string or preceded by whitespace
                          [\*"]*                # Optional leading asterisks or quotes
                          (?:
                            [A-Z]+ACEAE        # All-uppercase families ending with ACEAE (any number of letters before ACEAE)
                            |
                            [A-Z][a-z]+aceae    # Normal mixed-case families ending with 'aceae' (e.g. Celastraceae)
                            |
                            (?=[A-Za-z]*[A-Z])   # Ensure at least one uppercase letter exists in the following synonym
                              (?:
                                [Cc][Oo][Mm][Pp][Oo][Ss][Ii][Tt][Aa][Ee]       |   # Compositae
                                [Cc][Rr][Uu][Cc][Ii][Ff][Ee][Rr][Aa][Ee]       |   # Cruciferae
                                [Gg][Rr][Aa][Mm][Ii][Nn][Ee][Aa][Ee]           |   # Gramineae
                                [Gg][Uu][Tt][Tt][Ii][Ff][Ee][Rr][Aa][Ee]       |   # Guttiferae
                                [Ll][Aa][Bb][Ii][Aa][Tt][Ee][Ee]               |   # Labiatae
                                [Ll][Ee][Gg][Uu][Mm][Ii][Nn][Oo][Ss][Aa][Ee]   |   # Leguminosae
                                [Pp][Aa][Ll][Mm][Aa][Ee]                       |   # Palmae
                                [Uu][Mm][Bb][Ee][Ll][Ll][Ii][Ff][Ee][Rr][Aa][Ee] |   # Umbelliferae
                                [Pp][Aa][Pp][Ii][Ll][Ii][Oo][Nn][Aa][Cc][Ee][Ee]    # Papilionaceae
                              )
                          )
                          [\*"]*                # Optional trailing asterisks or quotes
                          (?!\S)                # Assert that the match is followed by whitespace or end-of-string
                        )
                        """, re.VERBOSE)

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
    
    division_str = "|".join(divisions)
    regex = re.compile(f"({division_str})", re.IGNORECASE)
    result = re.split(regex, text)

    remove_newline = lambda x: not(re.match(re.compile(r"^(\n)+$"), x))
    result = list(filter(None,result))
    result = list(filter(remove_newline, result))

    return list(zip(result[::2], result[1::2]))


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


def split_family(text: str, max_chunk_size: int=3000) -> list[str]:
    """
    Split the input text into chunks for inference.
    This function checks if the big block needs to be seperated.
    If so, it passes the chunk splitting to split_into_smaller_chunks function

    Args:
        text (str): Big block of text from one family
        max_chunk_size (int, optional): The maximum size of a chunk before it needs to be split. Defaults to 3000.

    Returns:
        list[str]: a list of all the split chunks for the family
    """


    # regex = re.compile("\n+(?=[A-Z ]+\n|.+?[aA][cC][eE][aA][eE])")

    regex = FAMILY_REGEX
    
    result = re.split(regex, text)
    
    final_list = []

    for family in result:
        
        if family is None or family == '':
            continue
        elif len(family) > max_chunk_size:
            
            small_chunks = split_into_smaller_chunks(family, max_chunk_size)
            final_list.extend(small_chunks)
        else:
            final_list.append(family)
    
    return final_list


def split_into_smaller_chunks(large_block: str, max_chunk_size: int=3000) -> list[str]:
    """
    Split the input text into smaller chunks.

    Args:
        text (str): Big block of text from one family
        max_chunk_size (int, optional): The maximum size of a chunk before it needs to be split. Defaults to 3000.

    Returns:
        list[str]: a list of all the split chunks from the big block
    """

    # Split the large block into family name at the start and the rest
    family_name, text_block = list(
                                filter(None, 
                                        re.split(r"^([A-Z]+)", large_block)
                                        )
                                )
    #print(f"====={family_name}=====")

    #if len(family_name) < 5:
    #    print(repr(large_block))
    # Define static small chunks
    small_chunks = [text_block[i:min(i+max_chunk_size, len(text_block))] for i in range(0, len(text_block), max_chunk_size)]

    # Setting cut_off index for next chunk to add to final chunks
    final_chunks = []
    cut_off = 0

    # Iterate through all the static chunks
    for ind, small_chunk in enumerate(small_chunks):
        # Define the regex for the species name    
        species_name = re.compile("^[A-Z]+ [a-z]+ \(?[A-Z]+\.\)?", re.IGNORECASE)

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
                    # if not break the loop and update cut_off
                    cut_off = ind
                    break
        
        # Join all the lines together and append it to the small chunk
        small_chunk += "\n".join(lines_to_add)
        # Add the family name back
        small_chunk = family_name + "\n\n" + small_chunk
        # Add it to final chunks
        final_chunks.append(small_chunk)
    

    return final_chunks


def convertToTextBlocks(text: str, 
                        divisions: list[str]=["Dicotyledones", "Monocotyledones", "Pteridophyta", "Hepaticae", "Algae"], 
                        max_chunk_size: int=3000) -> dict:
    """
    Convert the input extracted text into a dictionary of hierarchy: Divisions -> Family Name -> Content chunks

    Args:
        text (str): Extracted text to be split into chunks
        divisions (list[str], optional): A list of division names. Defaults to ["Dicotyledones", "Monocotyledones", "Pteridophyta", "Hepaticae", "Algae"].
        max_chunk_size (int, optional): Maximum chunk size of each small chunk. Defaults to 3000.

    Returns:
        dict: text splitted into a dictionary of hierarchy: Divisions -> Family Name -> Content chunks
    """

    cleaned_text = clean_text(text)

    division_splits = split_division(cleaned_text, divisions)

    splits = {}

    for division, division_text in division_splits:

        if division in splits:
            splits[division].extend(split_family(division_text, max_chunk_size))
        else:
            splits[division] = split_family(division_text, max_chunk_size)
    
    return splits
