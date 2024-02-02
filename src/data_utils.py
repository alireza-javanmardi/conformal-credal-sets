import json
import os
import pandas as pd


def load_raw(source):
    """returns list of dictionaries

    Args:
        source (str): "snli" or "mni_m"

    Returns:
        list: list of dictionaries, each dictionary contains info of an instance
    """
    with open(os.path.join("data","chaosNLI","chaosNLI_v1.0","chaosNLI_"+source+".jsonl"), "rb") as f:
        data = f.readlines()
        return [json.loads(datum) for datum in data]
    


def df_creator(raw_data):
    """get a list of dictionaries and return a dataframe

    Args:
        raw_data (list): list of dictionaries, each dictionary contains info of an instance

    Returns:
        dataframe: dataframe containing all instances
    """

    main_df = pd.DataFrame(raw_data)
    premise_hypothesis_df = pd.DataFrame(main_df["example"].values.tolist())
    merged_df = premise_hypothesis_df.merge(main_df, on="uid")
    return merged_df