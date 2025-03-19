import pandas as pd
import os
import json
import logging
from typing import Union

from json_repair import repair_json
from lib.json_schemas import get_catalogue

logger = logging.getLogger(__name__)

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
        json_loaded = catalogue_schema.model_validate_json(text)
        json_loaded = json.loads(json_loaded.model_dump_json(indent=2))
        verified = True
        message = json_loaded
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

def check_for_lists_and_dicts(dataframe: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check if there are any lists and dicts in the Dataframe

    Args:
        dataframe (pd.DataFrame): Pandas Dataframe

    Returns:
        tuple[bool, list[str]]: 
            1) If there are any lists or not. Boolean answer
            2) List of all columns with lists and dicts
    """
     
    list_of_columns = []

    for col in dataframe.columns:
        try:
            non_null_values = dataframe[col].dropna()

            if non_null_values.apply(lambda x: isinstance(x, (list,dict))).any():
                list_of_columns.append(col)
        except:
            logger.debug(f"{col} not found in datafrom")
            logger.debug(dataframe.head(2))

    return len(list_of_columns) > 0, list_of_columns


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

    # Collate the data into a pandas normalisable format
    collapse_dict_lists = lambda x: x[0] if isinstance(x, list) else x
    unnormalised = dict(
            data = [
                dict(division=div_name, items=div_value) for div_name, div_value in json_data.items()
            ]
        )      
    
    # Normalise with pandas
    normalised_df = pd.json_normalize(unnormalised)
    
    max_iter = 10
    # Define an infinite loop
    while max_iter > 0:

        # Check for any dicts or lists in each column         
        has_lists_and_dicts, list_of_columns = check_for_lists_and_dicts(normalised_df)

        # If no lists or dicts as values then break the loop
        if not(has_lists_and_dicts):
            break

        # Iterate through each column
        for col in list_of_columns:
            
            sample_value = normalised_df[col].dropna().iloc[0]
            # If it is a list then use pandas explode function to seperate them into multiple fields
            if isinstance(sample_value, list):
                normalised_df = normalised_df.explode(col, ignore_index=True)
            #If not and it is a dict, then perform json_normalisation provided by pandas
            elif isinstance(sample_value, dict):
                # To ensure no lists are misplaced in this column
                mask = normalised_df[col].apply(lambda x: isinstance(x, list))
                normalised_df.loc[mask, col] = normalised_df.loc[mask, col].explode(ignore_index=True)
                dict_df = pd.json_normalize(normalised_df[col])#.add_prefix(f"{col}-")
                normalised_df = normalised_df.drop(columns=[col]).reset_index(drop=True)
                normalised_df = pd.concat([normalised_df, dict_df], axis=1)
        
        max_iter -= 1
    # Save as csv
    save_csv(normalised_df, file_name, save_path)


def save_csv(csv_file: pd.DataFrame, file_name: str = "sample", save_path: str = "./outputs") -> None:
    """
    Save csv file

    Args:
        csv_file (object): Pandas Dataframe to be save as csv file
        file_name (str, optional): File name to save the CSV file under. Defaults to "sample".
        save_path (str, optional): Save path for csv file. Defaults to "./outputs".
    """
    # Check if file name is valid
    if not(file_name.endswith(".csv")):
        file_name += ".csv"
    
    # Load file path (path in which the json is saved)
    file_path = os.path.join(save_path, file_name)

    # Save to CSV
    logger.debug(f"Dumping extracted text into {file_path}")
    csv_file.to_csv(file_path, encoding="utf-8", index=False)
