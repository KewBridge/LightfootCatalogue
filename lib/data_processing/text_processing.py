import re
from typing import Optional, Iterator, Match
from lib.data_processing.chunker import SpeciesChunker


FAMILY_REGEX_PATTERN = """
                          [\*"\.]*                # Optional leading asterisks or quotes
                          (?:
                            (?i:TRIBE|SERIES)   # Match either "Tribe" or "Series"
                            \s+                # Ensure at least one space
                            [IVXLCDM\.]+         # Match Roman numerals (I, V, X, L, C, D, M)
                            \s+                # Ensure at least one space before the family name
                            ([A-Z\-]+(EE\.?|
                                    (AC|OR)EAE|
                                    Æ|
                                    (ACE|ORE|FER|NE)\.E\.?|
                                    AE|
                                    ORAE|
                                    (?i:OR\.E))\.?|
                                    \w+\.)?
                            |
                            [A-Z\-]+((AC|OR)EAE|Æ|(ACE|ORE|FER|NE)\.E\.?|AE|ORAE|OR\.E)\.? # All-uppercase families ending with ACEAE (any number of letters before ACEAE)
                            |
                            (COMPOSITAE
                            |CRUCIFERAE
                            |GRAMINEAE
                            |GUTTIFERAE
                            |LABIATAE
                            |LEGUMINOSAE
                            |PALMAE
                            |UMBELLIFERAE
                            |PAPILIONACEAE)\.? # All-uppercase families ending with ACEAE (any number of letters before ACEAE)
                          )
                          [\*"\.]*                # Optional trailing asterisks or quotes
                          

"""
FAMILY_REGEX_WITH_LOOKAHEAD = re.compile(rf"(?={FAMILY_REGEX_PATTERN})", re.VERBOSE)
FAMILY_REGEX = re.compile(rf"({FAMILY_REGEX_PATTERN})", re.VERBOSE)


class TextProcessor:

    def __init__(self):
        
        # Family Regex
        self.family_regex = re.compile(rf"""(?P<PAGENOSTART>^\d+)?\s*
                                       (?P<INDEX>[IVXLCDM\.]+\s)?(\s+|-)?
                                       (?P<FAMILY>{FAMILY_REGEX_PATTERN})\s*
                                       (?P<PAGENOEND>\d+$)?""", flags=re.VERBOSE)
        # Species Regex
        self.species_regex_pattern = """(?:\d+\.\s)?
                                        [A-Z][a-z\-]+
                                        (?:\s[a-z\-]+)?
                                        (?:\s(var\.|subsp\.|f\.)\s[a-z\-]+)?
                                        (?:\s[A-Z][a-z\-]+)?
                                        (?:\s\([\w\s]+\))?"""
        self.species_regex = re.compile(rf"(?P<SPECIES>{self.species_regex_pattern})", flags=re.VERBOSE)

        # Known non-species words
        self.not_species_text = set()

        self.species_chunker = SpeciesChunker()


    
    def __call__(self, text: str, divisions: list, max_chunk_size: int = 3000):
        
        # Pre-process input text to clean it
        text = self.preprocess_text(text, divisions[0])

        # split the structure by divisions
        div_struct = self.split_by_divisions(text, divisions)

        # Define a new structure
        struct = dict()

        for current_div, div_content in div_struct.items():
            current_div = current_div.strip()
            #print(f"==> Processing {current_div}")


            family_split = self.split_by_families(div_content)


            for i, text_chunk in enumerate(family_split):
                chunks = self.species_chunker.chunk_species(text_chunk["text"])
                text_chunk["species"] = self.species_chunker.group_into_major_chunks(chunks, max_chunk_size=max_chunk_size)

            struct[current_div] = family_split
        
        return struct
    
    def split_by_families(self, text: str):

        finds = re.finditer(self.family_regex, text)

        find_matches = [i for i in finds]

        text_chunks = []
        
        for idx, i in enumerate(find_matches):
            match = re.sub(r"[.\n\t,]*\s*([A-Z]+)\s*[.\n\t,]*", r"\1", i.group())
            start = i.end()
            end = find_matches[idx+1].start() if idx+1 < len(find_matches) else None
            text_chunk = text[start:end] if end else text[start:]
            text_chunks.append(dict(family=match, text=text_chunk))
        
        return text_chunks

    def _check_family(self, line: str) -> dict:
        """
        Check for all families in a line and organize their contents.
        
        Args:
            line (str): The line to check for families
        
        Returns:
            dict: A dictionary with family names as keys and their contents as values
        """
        if not line:
            return {}
        
        # Find all family matches
        family_matches = list(re.finditer(self.family_regex, line))
        
        # If no families found, return empty dict
        if not family_matches:
            return {}
        
        # Dictionary to store family information
        family_data = {}
        
        def clean_family_name (fname: str) -> str:
            fname = re.sub(r".*?([A-Za-z\s]+).*?", r"\1", fname)
            fname = fname.upper()
            return fname
        
        prev_family_name = None
        # Process each family match
        for idx, match in enumerate(family_matches):
            page_match = match.group("PAGENOSTART") or match.group("PAGENOEND")
            family_name = match.group('FAMILY').strip()
            family_name = clean_family_name(family_name)
            # Skip if this family was already processed (take the earliest match)
            if family_name in family_data:
                continue
            
            if page_match:
                family_name = prev_family_name
            # Determine the text segment for this family (until next family or end of line)
            start_pos = match.end()
            
            # Find the next family match position, or end of line if this is the last family
            end_pos = len(line)
            for next_match in family_matches[idx+1:]:
                next_family_name = next_match.group('FAMILY').strip()
                next_family_name = clean_family_name(next_family_name)
                # Only consider this a boundary if it's a different family
                if next_family_name != family_name:
                    end_pos = next_match.start()
                    break
            
            # Extract the content belonging to this family
            family_content = line[start_pos:end_pos].strip()
            
            # Store the family data
            if family_name in family_data:
                family_data[family_name]["content"].append(family_content)
            else:
                family_data[family_name] = {
                    #'match': match,
                    'content': [family_content],
                    'species': []  # To be filled if needed
                }
            
            prev_family_name = family_name

        return family_data
    def _find_all_species(self, line: str, family: Optional[str]=None) -> object:
        
        species_matches = re.finditer(self.species_regex, line)

        # filter out any matches that are 5 characters and above and have atleast 2 words in it
        species_matches = filter(lambda x: (len(x.group(0)) > 5 and len(x.group(0).split(" ")) >= 2), species_matches)

        regex_check = lambda x: True if re.match(r"\d+?\.?\s*\w{3,}\s\w{2}\.*", x.group(0)) else False
        species_matches = filter(regex_check, species_matches)
        
        species_matches = filter(lambda x: x.group(0) not in self.not_species_text, species_matches)

        check_against_gbif = lambda x: self._check_against_gbif(x, family)
        species_matches = filter(check_against_gbif, species_matches)

        return list(species_matches)

    def _check_against_gbif(self, species_match, family: str=None) -> bool:

        species_name = re.sub(r"^(\d+\.\s*)?", "", species_match.group(0))
        species_name = species_name.strip()
        
        if family:
            gbif_search = None #species.name_backbone(name=species_name, family=family, kingdom="plants", strict=False, verbose=True, limit=1)
        else:
            gbif_search = None #species.name_backbone(name=species_name, kingdom="plants", strict=False, verbose=True, limit=1)
        

        def check_gbif_dict(gbif_dict: dict) -> bool:
            
            if gbif_dict["matchType"] == "NONE":
                return False

            if (
                (gbif_dict["rank"].lower() in ["genus", "species"]) 
                and
                (gbif_dict["confidence"] >= 50)
                and
                (gbif_dict["status"].lower() == "ACCEPTED".lower())
                ):
                return True
            
            return False
        
        check_first_line = check_gbif_dict(gbif_search)

        if check_first_line:
            return True
        elif "alternatives" in gbif_search and len(gbif_search["alternatives"]) >= 1:
            # Only checking the first alternative
            check_alternative = check_gbif_dict(gbif_search["alternatives"][0])
            
            if check_alternative:
                return True
        
        self.not_species_text.add(species_match.group(0))
        return False
            


    def preprocess_text(self, text: str, first_division: str) -> str:
        """
        Preprocess the text for splitting into text blocks

        Args:
            text (str): Extracted text
            first_division (str): The first division in text

        Returns:
            str: Cleaned text
        """
        text = re.sub(rf"^.*?({re.escape(first_division)})", r"\1", text, flags=re.S | re.I)
         
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.MULTILINE) # Remove any markdown (bold) on string
        text = re.sub(r"\*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```", "", text, flags=re.MULTILINE) # Remove any markdown
        text = re.sub(r"^(Catalogue|catalogue)$", "", text, flags=re.MULTILINE) # Remove Catalogue/catalogue
        text = re.sub(f"^\d+\.?$", "", text, flags=re.MULTILINE)
        # Clean family ending
        text = re.sub(r"Æ", "AE", text, flags=re.MULTILINE)
        text = re.sub(r"œ", "ae", text, flags=re.MULTILINE)
        text = re.sub(r"ACE\.E\.?", "ACEAE", text, flags= re.MULTILINE) # This changes for all family level ones
        text = re.sub(r"ace\.e\.?", "aceae", text, flags= re.MULTILINE | re.I) # This changes for all others
        text = re.sub(r"OR\.E", "ORAE", text, flags= re.MULTILINE) # This changes for all family level ones
        text = re.sub(r"or\.e", "orae", text, flags= re.MULTILINE | re.I) # This changes for all others
        return text
    
    def _create_division_regex(self, divisions: Optional[list]=None) -> re.Pattern:
        """
        Generated the division regex

        Args:
            divisions (Optional[list], optional): List of divisions. Defaults to None.

        Returns:
            re.Pattern: Pattern for division regex
        """
        if not(divisions):
            return re.compile(f"(?:\d+\.?\s+)?([A-Z][a-z]+|[A-Z]+)\.?")
        
        division_str = "|".join(map(re.escape, divisions))
        return re.compile(f"(?:\d+\.?\s+)?({division_str})\.?", re.IGNORECASE)
    
    def split_by_divisions(self, text: str, divisions: list) -> dict:
        """
        Split the text by division and clean the output to get a structured hierarchy of divisions

        Args:
            text (str): extracted text
            divisions (list): List of divisions to split by

        Returns:
            dict: a structured hierarchy of divisions and their contents
        """

        # Generate div regexes
        # To split divisions
        div_regex = self._create_division_regex(divisions)
        # To check if a division
        div_check_regex = self._create_division_regex()

        # Intialise structure
        struct = {}

        #Split by divisions and clean
        div_split = re.split(div_regex, text)
        remove_newline = lambda x: not(re.match(re.compile(r"^(\n)+$"), x))
        div_split = list(filter(None,div_split))
        div_split = list(filter(remove_newline, div_split))

        # Pack into splits
        splits = list(zip(div_split[::2], div_split[1::2]))

        # Iterate through all divisions and Check if they match a divison, if not add it to previous divisions
        prev_div = None
        for div, content in splits:
            if re.match(div_check_regex, div):
                if div not in struct.keys():
                    struct[div] = content
                else:
                    struct[div] += content
                prev_div = div
            else:
                struct[prev_div] += div + content

        return struct

    def _clean_species_content(self, contents, family, max_chunk_size=3000):
        """
        Clean and chunk species content based on matches and size constraints.
        Ensures matched species text is preserved at the beginning of each chunk when splitting.
        
        Args:
            contents: List of content lines
            family: Family to search for species
            max_chunk_size: Maximum size of each content chunk
            
        Returns:
            List of chunked species content without duplicates
        """
        if not contents:
            return []
        
        species_list = []
        current_chunk = ""
        current_species_match = None  # Track the current species match text
        
        for idx, line in enumerate(contents):
            matches = self._find_all_species(line, family=family)
            
            # Case 1: No species matches in this line
            if not matches:
                if current_chunk:
                    # If adding this line would exceed max_chunk_size, split the chunk
                    if len(current_chunk + "\n" + line) > max_chunk_size:
                        # Only split if we know the current species match
                        if current_species_match:
                            species_list.append(current_chunk)
                            current_chunk = f"{current_species_match}\n{line}"
                        else:
                            species_list.append(current_chunk)
                            current_chunk = line
                    else:
                        current_chunk += "\n" + line
                else:
                    current_chunk = line
            
            # Case 2: Exactly one species match
            elif len(matches) == 1:
                match = matches[0]
                start_pos = match.start()
                species_match = match.group()  # Get the entire matched string
                
                # Species starts at beginning of line
                if start_pos == 0:
                    
                    current_species_match = species_match

                    if current_chunk: # and len(last_block+ "\n" + current_chunk) > max_chunk_size:
                        if species_list and len(species_list[-1] + "\n" + current_chunk) < max_chunk_size:
                                species_list[-1] += "\n" + current_chunk
                        else:
                            species_list.append(current_chunk)
                        
                        
                    current_chunk = line
                
                # Species appears later in the line
                else:
                    line_prefix = line[:start_pos]
                    line_species = line[start_pos:]
                    
                    # Handle the content before the species
                    if current_chunk:
                        if len(current_chunk + "\n" + line_prefix) > max_chunk_size:
                            # Split with current species if possible
                            if current_species_match:
                                species_list.append(current_chunk)
                                current_chunk = f"{current_species_match}\n{line_prefix}"
                            else:
                                species_list.append(current_chunk)
                                current_chunk = line_prefix
                        else:
                            current_chunk += "\n" + line_prefix
                            species_list.append(current_chunk)
                    else:
                        current_chunk = line_prefix
                        species_list.append(current_chunk)
                    
                    # Start new chunk with species part
                    current_species_match = species_match
                    current_chunk = line_species
            
            # Case 3: Multiple species matches
            else:
                for idx, match in enumerate(matches):
                    start_pos = match.start()
                    end_pos = matches[idx+1].start() if idx+1 < len(matches) else len(line)
                    species_match = match.group()  # Get the entire matched string
                    
                    # Handle the first match - might need to append prefix to previous chunk
                    if idx == 0:
                        line_prefix = line[:start_pos]
                        if line_prefix:
                            if current_chunk:
                                if len(current_chunk + "\n" + line_prefix) > max_chunk_size:
                                    # Split with current species if possible
                                    if current_species_match:
                                        species_list.append(current_chunk)
                                        current_chunk = f"{current_species_match}\n{line_prefix}"
                                    else:
                                        species_list.append(current_chunk)
                                        current_chunk = line_prefix
                                else:
                                    current_chunk += "\n" + line_prefix
                                    species_list.append(current_chunk)
                            else:
                                current_chunk = line_prefix
                                species_list.append(current_chunk)
                    
                    # Extract the current species segment
                    species_segment = line[start_pos:end_pos]
                    current_species_match = species_match
                    
                    # Check if this segment needs to be chunked further
                    if len(species_segment) > max_chunk_size:
                        # First chunk includes full species match
                        first_chunk = species_segment[:max_chunk_size]
                        species_list.append(first_chunk)
                        
                        # Remaining chunks start with the species match
                        remaining = species_segment[max_chunk_size:]
                        while remaining:
                            chunk_size = min(max_chunk_size - len(species_match) - 1, len(remaining))
                            next_chunk = f"{species_match}\n{remaining[:chunk_size]}"
                            species_list.append(next_chunk)
                            remaining = remaining[chunk_size:]
                        
                        current_chunk = ""
                    else:
                        if current_chunk:
                            species_list.append(current_chunk)
                        current_chunk = species_segment
                        
                        # If this is not the last match, add the chunk immediately
                        if idx + 1 < len(matches):
                            species_list.append(current_chunk)
                            current_chunk = ""
        
        # Don't forget the last chunk
        if current_chunk:
            species_list.append(current_chunk)
        
        species_list = list(dict.fromkeys(species_list))
        
        return species_list if species_list else contents
    
    def make_text_blocks(self, text_structure, max_chunk_size=3000, overlap_context=1000):

        text_blocks = []

        for div, div_content in text_structure.items():
            # for family, family_content in div_content["families"].items():

            #     if len("\n".join(family_content["species"])) <= max_chunk_size:
            #         text_blocks.append(dict(
            #             division=div,
            #             family=family,
            #             content="\n".join(family_content["species"])
            #         ))
            #         continue 
                
            #     text_block = ""
            #     overlap = ""
            #     for item in family_content["species"]:
                    
            #         if len(text_block) + len(item) <= max_chunk_size:
            #             text_block += item.strip() + "\n"
            #             if len(text_block) + len(item) >= max_chunk_size - overlap_context:
            #                 overlap += item.strip() + "\n"
            #         else:
            #             text_blocks.append(dict(
            #             division=div,
            #             family=family,
            #             content=text_block
            #             ))

            #             text_block = overlap
            #             overlap = ""

            for family_content in div_content:
                family = family_content["family"]
                contents = family_content["species"]

                for chunk in contents:
                    text_blocks.append(dict(
                        division=div,
                        family=family,
                        content=chunk
                    ))
            
        return text_blocks

