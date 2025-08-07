import re
from typing import Optional, Iterator, Match
from lib.data_processing.chunker import SpeciesChunker

# Logging
from lib.utils import get_logger
logger = get_logger(__name__)


# TODO: Improve regex / code for identifying family names

# TODO: Strictly split by family only
# THEN Under each family split by tribe
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
        # self.family_regex = re.compile(rf"""(?P<PAGENOSTART>^\d+)?\s*
        #                                (?P<INDEX>[IVXLCDM\.]+\s)?(\s+|-)?
        #                                (?P<FAMILY>{FAMILY_REGEX_PATTERN})\s*
        #                                (?P<PAGENOEND>\d+$)?""", flags=re.VERBOSE)
        legacy = [
            "COMPOSITAE", "GRAMINEAE", "LEGUMINOSAE", "PALMAE",
            "UMBELLIFERAE", "CRUCIFERAE", "LABIATAE", "GUTTIFERAE",
            "PAPILIONACEAE", "MIMOSACEAE", "CAESALPINIACEAE"
        ]
        legacy_alts = "|".join([re.escape(alt) for alt in legacy])

        self.family_regex = rf"\b([A-Z]+ACEAE|{legacy_alts})\b"
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


#TODO: update for family, tribe, series and species
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
# TODO:  update for family, tribe, series and species
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
        
        if text_chunks:
              return text_chunks
        else:
              return [{"family": "No family found", "text": text.strip()}]


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
        
        text = re.sub(r"^(Catalogue|catalogue)$", "", text, flags=re.MULTILINE) # Remove Catalogue/catalogue
        #text = re.sub(f"^\d+\.?$", "", text, flags=re.MULTILINE)
        # Clean family ending
        text = re.sub(r"Æ", "AE", text, flags=re.MULTILINE)
        text = re.sub(r"œ", "ae", text, flags=re.MULTILINE)

        text = re.sub(r"NE(\.|A)?E\.?", "NEAE", text, flags= re.MULTILINE) # This changes for all family level ones")
        text = re.sub(r"ACE(\.|A)?E\.?", "ACEAE", text, flags= re.MULTILINE) # This changes for all family level ones
        text = re.sub(r"ace(\.|a)?e\.?", "aceae", text, flags= re.MULTILINE) # This changes for all others
        text = re.sub(r"FLOR(\.|A)?E", "FLORAE", text, flags= re.MULTILINE) # This changes for all family level ones
        text = re.sub(r"flor(\.|a)?e", "florae", text, flags= re.MULTILINE) # This changes for all others
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

