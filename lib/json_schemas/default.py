from typing import Optional
from pydantic import BaseModel, Field, root_validator

class Folder(BaseModel):
    description: str = Field(..., description="A description of the folder contents. For description fields that contain citations or quotes, ensure proper escaping of all contained quotation marks.") 

    class Config:
        extra="allow"

    @root_validator(pre=True)
    def merge_extra_keys(cls, values):
        # Get the current description value
        desc = values.get("description", "")
        extra_parts = []
        # Iterate over a list of keys to safely remove items
        for key in list(values.keys()):
            if key != "description":
                # Pop the extra key so it won't be in the final model
                extra_value = values.pop(key)
                extra_parts.append(f"{key}: {extra_value}")
        if extra_parts:
            desc = f"{desc} " + " ".join(extra_parts)
        values["description"] = desc.strip()
        return values

class Species(BaseModel):
    speciesName: str = Field(..., description="The full scientific name as written in the text. Keep the index number infront of the family name.N/A if not available.")
    folders: list[Folder] = Field(default_factory=list, description="A list of each folder under species.")

class BotanicalCatalogue(BaseModel):
    familyName: str = Field(..., description="The scientific family name in uppercase. If not available, use \"N/A\". Might include Tribe or Series with roman indexes.")
    species: list[Species] = Field(default_factory=list, description="List of all species and their contents")
    
