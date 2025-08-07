from typing import Optional
from pydantic import BaseModel, Field, root_validator

class FolderAndSheets(BaseModel):
    description: str = Field(..., description="A description of the folder or sheet contents. For description fields that contain citations or quotes, ENSURE proper escaping of all contained quotation marks.") 

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
    species_name: str = Field(..., description="The full SCIENTIFIC SPECIES NAME as written in the text. Keep the index number infront of the family name. If not available, use N/A.")
    number_of_folders: int = Field(..., description="The number of folders under this species. If not available, use 0.")
    number_of_sheets: int = Field(..., description="The number of sheets under this species. If not available, use 0.")
    folders_and_sheets: list[FolderAndSheets] = Field(default_factory=list, description="A list of each folder and sheet under species.")

class BotanicalCatalogue(BaseModel):
    family_name: str = Field(..., description="The SCIENTIFIC FAMILY NAME in uppercase. If not available, use \"N/A\".")
    species: list[Species] = Field(default_factory=list, description="List of all species and their contents")
    
