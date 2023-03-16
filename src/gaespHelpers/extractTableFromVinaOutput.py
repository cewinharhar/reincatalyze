from pandas import DataFrame

def extractTableFromVinaOutput(output : str):
    """
    extracts the mode & affinit table from the vina output
    """
    # Split the output into lines
    lines = output.split('\n')
    
    # Find the line where the table starts
    table_start = None
    for i, line in enumerate(lines):
        if line.startswith('mode |'):
            table_start = i
            break
    
    # If the table start line was not found, return None
    if table_start is None:
        return None
    
    # Find the line where the table ends
    table_end = None
    for idx, findEnd in enumerate(lines[table_start:]):
        if findEnd == "Writing output ... done.":
            table_end = table_start + idx
    
    # If the table end line was not found, return None
    if table_end is None:
        return None
    
    # Extract the table lines
    table_lines = lines[table_start:table_end]
    
    # Parse the table lines and return a list of dictionaries
    col_names = table_lines[0].split('|')[:-1]
    col_names = [col.strip() for col in col_names]

    # create an empty list to hold the data
    data = []

    # loop through the remaining lines and extract the data
    for line in table_lines[3:]:
        mode, aff, rmsd_lb, rmsd_ub = line.split()
        data.append([int(mode), float(aff)])

    # create a dataframe from the data and column names
    df = DataFrame(data, columns=col_names)

    return df