import pandas as pd
import os
import json


#==================================
# Saving JSON
#==================================

def save_json(json_file: dict, file_name: str = "sample", save_path: str = "./outputs/"):
    
    # Check if file name is valid
    if not(file_name.endswith(".json")):
        file_name += ".json"
    
    # Load file path (path in which the json is saved)
    file_path = os.path.join(save_path, file_name)

    print(f"Dumping extracted text into {file_path}")
    with open(file_path, "w", encoding="utf-8") as save_file:
            json.dump(json_file, save_file, indent=4)



#==================================
# Saving CSV
#==================================

def check_for_lists_and_dicts(dataframe: object):
     
    list_of_columns = []

    for col in dataframe.columns:
        sample_item = dataframe[col][0]

        if isinstance(sample_item, list) or isinstance(sample_item, dict):
             list_of_columns.append(col)

    return len(list_of_columns) > 0, list_of_columns

def save_csv_from_json(json_file: str | dict, file_name: str = "sample", save_path: str = "./outputs"):

    # Load the json data from either file or provided dictionary
    if isinstance(json_file, str):
        if os.path.isfile(os.path.join(save_path, json_file)):
             with open(json_file, "r", encoding="utf-8") as open_file:
                  json_data = json.load(open_file)
        else:
             raise ValueError(f"{json_file} is not a valid path")
    elif isinstance(json_file, dict):
        json_data = json_file
    else:
        raise ValueError(f"{json_file} is not a valid JSON file path or object")

    # Collate the data into a pandas normalisable format
    unnormalised = dict(
            data = [
                dict(division=div_name, items=div_value) for div_name, div_value in json_data.items()
            ]
        )          
    
    # Normalise with pandas
    normalised_df = pd.json_normalize(unnormalised)
    
    # Define an infinite loop
    while True:

        # Check for any dicts or lists in each column         
        has_lists_and_dicts, list_of_columns = check_for_lists_and_dicts(normalised_df)

        # If no lists or dicts as values then break the loop
        if not(has_lists_and_dicts):
            break

        # Iterate through each column
        for col in list_of_columns:
            
            # If it is a list then use pandas explode function to seperate them into multiple fields
            if isinstance(normalised_df[col][0], list):
                normalised_df = normalised_df.explode(col, ignore_index=True)
            #If not and it is a dict, then perform json_normalisation provided by pandas
            else:
                dict_df = pd.json_normalize(normalised_df[col])
                normalised_df = normalised_df.drop(columns=[col]).reset_index(drop=True)
                normalised_df = pd.concat([normalised_df, dict_df], axis=1)
    
    # Save as csv
    save_csv(normalised_df, file_name, save_path)




def save_csv(csv_file: object, file_name: str = "sample", save_path: str = "./outputs"):
    # Check if file name is valid
    if not(file_name.endswith(".csv")):
        file_name += ".csv"
    
    # Load file path (path in which the json is saved)
    file_path = os.path.join(save_path, file_name)

    # Save to CSV
    csv_file.to_csv(file_path, encoding="utf-8", index=False)