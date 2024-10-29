

# Using Qwen2-VL-7B model
MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# A list of image extensions to approve
IMAGE_EXT = ["jpeg", "png", "jpg"]
IGNORE_FILE = [".ipynb_checkpoints"]
# Path to the save directory of the json files
SAVE_PATH = "./parsed_json"
CROPPED_DIR_NAME = "cropped"

PROMPT = """The image consists of a catalogue of plant collection names from the Lightfoot collection. It follows the hierarchy of Taxon name, species name and the collection inside the folder consisting of species name, where it was collected, who collected it (citation) and other meta data.

I want you to extract the information from the catalogue and prepare a structured json output. If a text does not have taxon name or species name either at the start or at the very end, please mark that down as extra_COUNTER.

Example:

{"CAPRIFOLIACEAE" : {"Linnaea borealis L.": [ {"folder": 1, "content": "Linnea borealis [JL]. i. Cites Linn. Sp. Pl. 631; Bauh. Pin. 93"} ] }
}

Example (extra data at the start or end of image without taxon):

{"extra_data_1":{"content": "Folder 2. Campanula hederacea [G]. i. "Devon & Cornwal" [JL]"},
"CAPRIFOLIACEAE" : {"Linnaea borealis L.": [ {"folder": 1, "content": "Linnea borealis [JL]. i. Cites Linn. Sp. Pl. 631; Bauh. Pin. 93"} ] }
}

Your output should be a structured json format as shown above in the examples and not the example itself."""

DEFAULT_PROMPT = "Process this image in two steps: Step 1: Turn the text in this picture into markdown. Indicate italics. \
            Indent lines which discuss folder contents as unordered lists Step 2: Convert the markdown text that you created in step 1 into JSON. \
            Use the heading texts as keys, and the folder details as nested numbered lists\
            Your output should consist of the markdown text, then a fenced JSON code block"

get_conversation = lambda prompt: [
            {
                "role": "system",
                "content": "You are an expert in horticulture, plant taxonomy and botanical catalogues"
            },
            {
                "role": "system",
                "content": '''
                Context:
                The following images provided are snapshots of botanical catalogue pages from botanical books. The pages contain information on species, where they were collected and in which folder they were stored. The information is organised in order of plant taxonomy with the highest classification being family name. Typically the page contain information seperated in one to two columns. Family names are typically Capitalized and species names can be found italicized. The contents of the folders are written under each species and sometimes found at the start of the page (mostly due to continuation from the previous page).'''
            },
            {
                "role": "system",
                "content": """
                Instructions:
                1) Parse the image for the unstructed text
                2) Refactor the unstructed text from step 1 into a json structed outlined below
                3) JSON key values should be left empty or set to N/A if the corresponding information is not found
                4) Duplicate dictionary keys are not allowed
                5) Ensure all JSON keys are in camel case
                6) Ensure all JSON key-value pairs strictly follow the format and data types specified in the template below.
                7) Ensure output JSON is valid JSON format. It should not have trailing commas or unqouted keys
                8) Only return the JSON structure as a string
                """
            },
            {
                "role": "system",
                "content": '''
                Rules:
                metadata: Metadata for each page containing the division if available and page number of given image.
                division: Division of the current page, defined by a bold and larger font size. If it is not available set to N/A. Do not try to fill this in using any other information if it is not available.
                page: Page number as denoted on the top right part of the image, typically depicted by 1 to 4 digits. If not available set to N/A.
                contents: A list of dictionaries containing the familyName and species information for all families and species under them.
                familyName: The scientific name of the family of the species, typically found capitalized. If not available do not try to fill in and set as N/A. Family name must be Captialized.
                species: A list of dictionaries containing all speciesName and folders.
                speciesName: The full scientific name of the species as noted in the image. Do not change format or correct grammer.
                folders: A list of dictionaries containing the description and citations for each folder under the noted species.
                description: The description under the folder noting where the plant was found and collected from.
                citations: A list of all collectors and transcriptors. Typically found as capitalized initials in square brackets. The number of characters for each citation typically range between 1 to 4 characters.
                previous: A list of dictionaries containing the description and citations for each folder under a non-noted species. This inforamation is generally found at the start of the page and is a textual continuation from a previous page.
                '''
            },
            {
                "role": "system",
                "content": '''
                Template/Structure:
                
                {
                    "metadata":{ 
                        "division":"", 
                        "page":"" 
                    },
                    "contents":[ 
                        {
                            "familyName":"", 
                            "species":[ 
                                {"speciesName":"", 
                                 "folders":[ 
                                     {"description":"", 
                                      "citations":["",""] 
                                     } 
                                 ]
                                }
                            ]
                        }
                    ],
                    "previous":[ 
                        {"description":"", 
                         "citations":["",""] 
                        }
                    ]
                }
                '''
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

CONVERSATION = get_conversation(PROMPT)

# Variable Maximum number of unique tokens (words) the model can remember/allow at any one time
# This affects how much the model returns
# Since we are asking for both the markdown and JSON code, we need a high value
MAX_NEW_TOKENS = 5000
# Define batch size (greater the size, the more memory needed)
# With a 80GB RAM, a batch size of 3 takes around 55-60GB (incl model)
BATCH_SIZE = 2
