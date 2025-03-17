

def create_text_blocks(struct, max_block_size):
    """Create text blocks from hierarchical structure with size limitation."""
    blocks = []
    
    for division_name, division_data in struct.items():
        # Process division details
        process_division_details(blocks, division_name, division_data, max_block_size)
        
        # Process families
        process_families(blocks, division_name, division_data, max_block_size)
    
    return blocks

def process_division_details(blocks, division_name, division_data, max_block_size):
    """Process and split division details into blocks."""
    if not division_data["details"]:
        return
        
    division_details = "\n".join(division_data["details"])
    create_content_blocks(
        blocks, 
        "division_details", 
        division_details, 
        max_block_size,
        {"division": division_name}
    )

def process_families(blocks, division_name, division_data, max_block_size):
    """Process families within a division."""
    for family_name, family_data in division_data["families"].items():
        # Process family details
        # if family_data["details"]:
        #     family_details = "\n".join(family_data["details"])
        #     create_content_blocks(
        #         blocks, 
        #         "family_details", 
        #         family_details, 
        #         max_block_size,
        #         {"division": division_name, "family": family_name}
        #     )
        
        # Process species
        process_species(blocks, division_name, family_name, family_data, max_block_size)

def process_species(blocks, division_name, family_name, family_data, max_block_size):
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
    create_species_blocks(blocks, species_list, max_block_size, context)

def create_species_blocks(blocks, species_list, max_block_size, context):
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

def create_content_blocks(blocks, block_type, content, max_block_size, context):
    """Create blocks from any content with size limitation."""
    if len(content) <= max_block_size:
        blocks.append({
            "type": block_type,
            **context,
            "content": content
        })
    else:
        # For text that needs character-by-character splitting
        split_content_by_size(blocks, block_type, content, max_block_size, context)

def split_content_by_size(blocks, block_type, content, max_block_size, context):
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