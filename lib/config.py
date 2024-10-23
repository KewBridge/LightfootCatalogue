

# Using Qwen2-VL-7B model
MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# A list of image extensions to approve
IMAGE_EXT = ["jpeg", "png", "jpg"]
IGNORE_FILE = [".ipynb_checkpoints"]
# Path to the save directory of the json files
SAVE_PATH = "./parsed_json"
CROPPED_DIR_NAME = "cropped"

PROMPT = "Process this image in two steps: Step 1: Turn the text in this picture into markdown. Indicate italics. \
            Indent lines which discuss folder contents as unordered lists Step 2: Convert the markdown text that you created in step 1 into JSON. \
            Use the heading texts as keys, and the folder details as nested numbered lists\
            Your output should consist of the markdown text, then a fenced JSON code block"

DEFAULT_PROMPT = "Process this image in two steps: Step 1: Turn the text in this picture into markdown. Indicate italics. \
            Indent lines which discuss folder contents as unordered lists Step 2: Convert the markdown text that you created in step 1 into JSON. \
            Use the heading texts as keys, and the folder details as nested numbered lists\
            Your output should consist of the markdown text, then a fenced JSON code block"

get_conversation = lambda prompt: [
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
