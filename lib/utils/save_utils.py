import pandas as pd
import os
import json

from typing import Union

from json_repair import repair_json
from lib.json_schemas import get_catalogue

# Logging
from lib.utils import get_logger
logger = get_logger(__name__)
#==================================
# Saving JSON
#==================================
def verify_json(text: str, clean: bool = False, out: bool = False, schema: str = "default") -> bool:
    """
    Verifies if the input text is of JSON format

    Parameters:
        text (str) : Initial input text to be verified for JSON format
    Return:
        Verification : True or False depending on if the input text is json format or not
    """

    verified = False
    message = None

    # Initiate a repair using json-repair
    #message = repair_json(text, ensure_ascii=False, return_objects=True)
    catalogue_schema = get_catalogue(schema)
    try:
        text = repair_json(text, ensure_ascii=True, return_objects=False) if clean else text
        text_obj = json.loads(text)
        json_loaded = catalogue_schema(**text_obj)
        verified = True
        message = json_loaded.dict()
    except Exception as e:
        logger.debug(f">>> Non JSON format found in input text")
        logger.debug(f">>> Error: \n {e}")
        message = e

    if out:
        return verified, message
    return verified


def save_json(json_file: dict, file_name: str = "sample", save_path: str = "./outputs/") -> None:
    """

    Save JSON file

    Args:
        json_file (dict): JSON object to be dumped into json file
        file_name (str, optional): Name of file . Defaults to "sample".
        save_path (str, optional): save path of file. Defaults to "./outputs/".
    """
    
    # Check if file name is valid
    if not(file_name.endswith(".json")):
        file_name += ".json"
    
    # Load file path (path in which the json is saved)
    file_path = os.path.join(save_path, file_name)

    logger.debug(f"Dumping extracted text into {file_path}")
    with open(file_path, "w", encoding="utf-8") as save_file:
            json.dump(json_file, save_file, indent=4)



#==================================
# Saving CSV
#==================================


def save_csv_from_json(json_file: Union[str, dict], 
                       file_name: str = "sample", 
                       save_path: str = "./outputs") -> None:
    """
    unwrap the JSON file into a CSV file and save the CSV file.

    Args:
        json_file (Union[str, dict]): Either a json object or path to a JSON file
        file_name (str, optional): File name to save the CSV file under. Defaults to "sample".
        save_path (str, optional): Save path for csv file. Defaults to "./outputs".
    """

    # Load the json data from either file or provided dictionary
    if isinstance(json_file, str):
        if os.path.isfile(json_file):
             with open(json_file, "r", encoding="utf-8") as open_file:
                  json_data = json.load(open_file)
        else:
             raise ValueError(f"{json_file} is not a valid path")
    elif isinstance(json_file, dict):
        json_data = json_file
    else:
        raise ValueError(f"{json_file} is not a valid JSON file path or object")

    converter = JSONTOCSVCONVERTER(output_path=save_path)
    div_data = converter.add_division(json_data)
    flattened_data = converter.flatten_all(div_data)
    normalised_df = pd.json_normalize(flattened_data)

    if not(file_name.endswith(".csv")):
        file_name += ".csv"

    normalised_df.to_csv(os.path.join(save_path, file_name), index=True)


class JSONTOCSVCONVERTER:

    def __init__(self, output_path: str = "./outputs"):
        self.output_path = output_path

    def flatten_dict(self, key, value, parent_record):
        """
        Flatten a single key/value pair from a dict.
        Handles nested dicts, lists, and primitive values.
        """
        next_batch = []

        for child_record in self.flatten(value, {}):
            new_parent_record = parent_record.copy()
            for child_key, child_value in child_record.items():
                column_name = child_key if child_key else key #f"{key}.{child_key}"
                new_parent_record[column_name] = child_value
            next_batch.append(new_parent_record)

        return next_batch
    
    def flatten_list(self, key, value, parent_record):

        next_batch = []

        if all(not isinstance(elem, (dict, list)) for elem in value):
            # join primitives
            new_parent_record = parent_record.copy()
            new_parent_record[key] = ",".join(map(str, value))
            next_batch.append(new_parent_record)
        else:
            # explode objects/lists
            for elem in value:
                for child_record in self.flatten(elem, {}):
                    new_parent_record = parent_record.copy()
                    for child_key, child_value in child_record.items():
                        column_name = child_key if child_key else key #f"{key}.{child_key}"
                        new_parent_record[column_name] = child_value
                    next_batch.append(new_parent_record)
        
        return next_batch

    def flatten(self, item, record=None):
        if record is None:
            record = {}

        # If it's a dict, walk each key/value
        if isinstance(item, dict):
            records = [record]
            for key, value in item.items():
                next_batch = []
                for record in records:
                    if isinstance(value, dict):
                        # flatten nested dict under 'key'
                        next_batch.extend(self.flatten_dict(key, value, record))

                    elif isinstance(value, list):
                        # decide: list of primitives vs list of dicts
                        next_batch.extend(self.flatten_list(key, value, record))
                    else:
                        # primitive value
                        new_record = record.copy()
                        new_record[key] = value
                        next_batch.append(new_record)

                records = next_batch
            return records
        
    def add_division(self, data):
        
        new_data = []
        for key, value in data.items():
            new_data.append({"division": key, "data": value})
        
        return new_data

    def flatten_all(self, data):

        flattened_data = []

        for item in data:
            flattened_data.extend(self.flatten(item))

        return flattened_data