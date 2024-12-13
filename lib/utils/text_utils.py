import re


def clean_text(text):

    result = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.MULTILINE) # Remove any markdown (bold) on strings
    result = re.sub(r"```", "", result, flags=re.MULTILINE) # Remove any markdown
    result = re.sub(r"^Catalogue|catalogue$", "", result, flags=re.MULTILINE) # Remove Catalogue/catalogue
    result = re.sub(r"^[0-9]+$", "", result, flags=re.MULTILINE) # Remove page numbers
    result = re.sub(r"^John Lightfoot$", "", result, flags=re.MULTILINE) # Remove John Lightfoot name if on its own
    result = re.sub(r"\n{3,}", "", result)
    
    return result

def split_division(text, divisions=["Dicotyledones", "Monocotyledones", "Pteridophyta", "Hepaticae", "Algae"]):

    if divisions is None:
        return [("MAIN", text)]
    
    division_str = "|".join(divisions)
    regex = re.compile(f"({division_str})", re.IGNORECASE)
    result = re.split(regex, text)

    remove_newline = lambda x: not(re.match(re.compile(r"^(\n)+$"), x))
    result = list(filter(None,result))
    result = list(filter(remove_newline, result))

    return list(zip(result[::2], result[1::2]))

def find_family(text):
    regex = re.compile("\n\n(?=[A-Z ]+\n|.+?[aA][cC][eE][aA][eE])")

    result = re.findall(regex, text)

    return list(filter(None,result))

def split_family(text, max_chunk_size=3000):

    regex = re.compile("\n\n(?=[A-Z ]+\n|.+?[aA][cC][eE][aA][eE])")

    result = re.split(regex, text)
    
    final_list = []

    for family in result:
        
        if family is None or family is '':
            continue
        elif len(family) > max_chunk_size:
            
            small_chunks = split_into_smaller_chunks(family, max_chunk_size)
            final_list.extend(small_chunks)
        else:
            final_list.append(family)
    
    return final_list

def split_into_smaller_chunks(large_block, max_chunk_size=3000):

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


def convertToTextBlocks(text, divisions=["Dicotyledones", "Monocotyledones", "Pteridophyta", "Hepaticae", "Algae"], max_chunk_size=3000):

    cleaned_text = clean_text(text)

    division_splits = split_division(cleaned_text, divisions)

    splits = {}

    for division, division_text in division_splits:

        if division in splits:
            splits[division].extend(split_family(division_text, max_chunk_size))
        else:
            splits[division] = split_family(division_text, max_chunk_size)
    
    return splits