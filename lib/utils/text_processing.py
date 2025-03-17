import re
from typing import Optional

from lib.utils.block_builder import create_text_blocks
#(?<!\S)              # Assert position is at start-of-string or preceded by whitespace
#(?!\S)                # Assert that the match is followed by whitespace or end-of-string
FAMILY_REGEX_PATTERN = """
                          [\*"\.\n]*                # Optional leading asterisks or quotes
                          (?:
                            (?i:TRIBE|SERIES)   # Match either "Tribe" or "Series"
                            \s+                # Ensure at least one space
                            [IVXLCDM\.]+         # Match Roman numerals (I, V, X, L, C, D, M)
                            \s+                # Ensure at least one space before the family name
                            \w+
                            |
                            [A-Z]+ACEAE\.?        # All-uppercase families ending with ACEAE (any number of letters before ACEAE)
                            |
                            [A-Za-z]+Æ\.?
                            |
                            [A-Z]+ACE\.E\.?
                            |
                            [A-Za-z]+ACE\.E\.?
                            |
                            [A-Z]+AE\.?
                            |
                            [A-Z][a-z]+aceae\.?    # Normal mixed-case families ending with 'aceae' (e.g. Celastraceae)
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



class TextProcessor:
    def __init__(self, family_regex=None):
        """Initialize the TextProcessing class."""
        self.family_regex = family_regex or FAMILY_REGEX
    
    def __call__(self, text, divisions, max_chunk_size=3000, return_blocks=True):
        """Parse text into a hierarchical structure based on divisions and families."""
        # Preprocess text
        text = self._preprocess_text(text, divisions[0])
        
        # Split by division and get a structure
        div_struct = self._split_by_divisions(text, divisions)
        
        
        # Initialize structure and patterns
        struct = {}

        #return div_struct
        for current_div, content in div_struct.items():
            current_div = current_div.strip()
            # Split the text with respect to paragraphs
            split_content = content.split("\n\n")
            current_family = None
            struct[current_div] = {"details":[], "families": {}}
            for line in split_content:
                line = line.strip()
                if not line:
                    continue

                # Process the line based on its content type
                current_family = self._process_line(
                    line, struct, 
                    current_div, current_family
                )
        
        if return_blocks:
            return create_text_blocks(struct, max_chunk_size)

        return struct
    
    def create_text_blocks(self, struct, max_block_size):
        """Create text blocks from hierarchical structure with size limitation."""
        blocks = []
        
        for division_name, division_data in struct.items():
            # Process division details
            self._process_division_details(blocks, division_name, division_data, max_block_size)
            
            # Process families
            self._process_families(blocks, division_name, division_data, max_block_size)
        
        return blocks
    

    def _preprocess_text(self, text: str, first_division: str) -> str:
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
    
    def _process_line(self, line, struct, current_div, current_family):
        """Process a single line and update the structure accordingly."""
        # Check for family match
        
        if re.match(self.family_regex, line) and current_div is not None:
            current_family = self._process_family_line(
                line, struct, current_div
            )
        # If not a division or family, it's a species or description
        elif current_div is not None:
            self._add_content_to_structure(
                line, struct, current_div, current_family
            )

        return current_family
    
    def _split_by_divisions(self, text: str, divisions: list) -> dict:
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
    
    # def _process_family_in_division(self, item, struct, division_name):
    #     """Process a family that appears in the same line as a division."""
    #     family_matches = list(filter(None, re.split(self.family_regex, item)))
    #     family_name = family_matches[0].strip()
        
    #     if family_name not in struct[division_name]["families"]:
    #         struct[division_name]["families"][family_name] = {
    #             "details": family_matches[1:], 
    #             "species": []
    #         }
    #     else:
    #         struct[division_name]["families"][family_name]["details"].extend(family_matches[1:])
        
    #     return family_name
    
    def _process_family_line(self, line, struct, current_div):
        """Process a line that contains a family."""
        family_matches = list(filter(None, re.split(self.family_regex, line)))
        family_name = family_matches[0].strip()
        
        if family_name not in struct[current_div]["families"]:
            struct[current_div]["families"][family_name] = {
                "species": family_matches[1:]
            }
        else:
            struct[current_div]["families"][family_name]["species"].extend(family_matches[1:])
        
        return family_name
    
    def _add_content_to_structure(self, line, struct, current_div, current_family):
        """Add content to the appropriate part of the structure."""
        if current_family is not None:
            # Add to current family as a species
            struct[current_div]["families"][current_family]["species"].append(line)
        else:
            # Add to division details
            if "details" not in struct[current_div]:
                struct[current_div]["details"] = []
            struct[current_div]["details"].append(line)
    
    # Private methods for text block creation
    def _process_division_details(self, blocks, division_name, division_data, max_block_size):
        """Process and split division details into blocks."""
        if not division_data["details"]:
            return
            
        division_details = "\n".join(division_data["details"])
        self._create_content_blocks(
            blocks, 
            "division_details", 
            division_details, 
            max_block_size,
            {"division": division_name}
        )
    
    def _process_families(self, blocks, division_name, division_data, max_block_size):
        """Process families within a division."""
        for family_name, family_data in division_data["families"].items():
            # Process family details
            # if family_data["details"]:
            #     family_details = "\n".join(family_data["details"])
            #     self._create_content_blocks(
            #         blocks, 
            #         "family_details", 
            #         family_details, 
            #         max_block_size,
            #         {"division": division_name, "family": family_name}
            #     )
            
            # Process species
            self._process_species(blocks, division_name, family_name, family_data, max_block_size)
    
    def _process_species(self, blocks, division_name, family_name, family_data, max_block_size):
        """Process species within a family."""
        if not family_data["species"]:
            return
            
        species_list = family_data["species"]
        context = {"division": division_name, "family": family_name}
        
        # Simple case: all species fit in one block
        species_text = "\n".join(species_list)
        if len(species_text) <= max_block_size:
            blocks.append({
                "type": "species",
                **context,
                "content": species_text
            })
            return
        
        # Complex case: need to split species across multiple blocks
        self._create_species_blocks(blocks, species_list, max_block_size, context)
    
    def _create_species_blocks(self, blocks, species_list, max_block_size, context):
        """Create blocks from species list with intelligent splitting."""
        current_text = ""
        
        for species in species_list:
            # Check if adding this species would exceed the block size
            potential_text = current_text + ("\n" + species if current_text else species)
            
            if len(potential_text) > max_block_size and current_text:
                # Current block is full, add it to blocks
                blocks.append({
                    "type": "species",
                    **context,
                    "content": current_text
                })
                current_text = species
            else:
                # Add to current block
                current_text = potential_text
        
        # Add the last block if there's anything left
        if current_text:
            blocks.append({
                "type": "species",
                **context,
                "content": current_text
            })
    
    def _create_content_blocks(self, blocks, block_type, content, max_block_size, context):
        """Create blocks from any content with size limitation."""
        if len(content) <= max_block_size:
            blocks.append({
                "type": block_type,
                **context,
                "content": content
            })
        else:
            # For text that needs character-by-character splitting
            self._split_content_by_size(blocks, block_type, content, max_block_size, context)
    
    def _split_content_by_size(self, blocks, block_type, content, max_block_size, context):
        """Split content into blocks of maximum size."""
        # Try to split at newlines first for more natural breaks
        lines = content.split('\n')
        current_block = ""
        
        for line in lines:
            if len(current_block + line + '\n') > max_block_size and current_block:
                # This line would make the block too big, store current block
                blocks.append({
                    "type": block_type,
                    **context,
                    "content": current_block.rstrip()
                })
                current_block = line + '\n'
            else:
                current_block += line + '\n'
        
        # Add the last block if there's anything left
        if current_block:
            blocks.append({
                "type": block_type,
                **context,
                "content": current_block.rstrip()
            })