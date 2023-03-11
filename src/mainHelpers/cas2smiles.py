import requests

def cas2smiles(casList: list):
    """
    Curls smiles from cas from cactus.nci.nih.gov
    """
    if not isinstance(casList, list):
        casList = [casList]

    smilesList = []
    nrOfCas = len(casList)

    print(f"Number of molecules: {nrOfCas}")
    print("Starting:")

    for idx, molec in enumerate(casList):
        print(f"    {idx+1}/{nrOfCas}", end = "\r")
        try:
            tmp = (
                "http://cactus.nci.nih.gov/chemical/structure/"+molec+ "/smiles"
            )

            res = requests.get(tmp)
            if not res.status_code == 200:
                print(f"cas: {molec} did not work, continuing")

            smilesList.append(res.text)
        except Exception as err:
            print(err)

    print(f"Scraping successfull: \n {len(smilesList)}/{nrOfCas} stored")
    return smilesList
