import re
import pandas as pd
from pandas import DataFrame

def extractTableFromVinaOutput(input_string):
    # Regular expression to match the lines with mode and affinity
    pattern = r"^\s*(\d+)\s+(-?\d+\.\d+)"
    
    # Extract mode and affinity from input string
    data = []
    for line in input_string.split('\n'):
        match = re.match(pattern, line)
        if match:
            mode, affinity = match.groups()
            data.append([int(mode), float(affinity)])

    # Create a pandas DataFrame from the extracted data
    table = pd.DataFrame(data, columns=['mode', 'affinity'])

    if len(table) == 0:
        raise RuntimeError

    return table
