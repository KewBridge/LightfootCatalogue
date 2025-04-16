from typing import Optional
from pydantic import BaseModel, Field, model_validator

class Folder(BaseModel):
    description: str = Field(..., description="A description of the folder contents. For description fields that contain citations or quotes, ensure proper escaping of all contained quotation marks.") 

    class Config:
        extra="allow"

    @model_validator(mode="before")
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

class FamilyContent(BaseModel):
    familyName: str = Field(..., description="The scientific family name in uppercase. If not available, use \"N/A\". Might include Tribe or Series with roman indexes.")
    species: list[Species] = Field(default_factory=list, description="List of all species and their contents")
    
class BotanicalCatalogue(BaseModel):
    #previousContent: str = Field(..., description="The contentnt before the first species and after the first family name. This is usually text that is carried over from the previous prompt input.N\A if not available.")
    familyContents : list[FamilyContent] = Field(default_factory=list, description="A list of all family contents. Each family content contains the family name and a list of species.")

    @model_validator(mode="before")
    def merge_extra_keys(cls, values):
        
        if not "previousContent" in values:
            values["previousContent"] = "N\A"
        return values