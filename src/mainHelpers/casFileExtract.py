def casFileExtract(subProdDict : dict):
    # Extract cas from storage
    subCas = [
        subProdDict["substrate"][key_][1] for key_ in subProdDict["substrate"].keys()
    ]
    subName = [
        subProdDict["substrate"][key_][0] for key_ in subProdDict["substrate"].keys()
    ]

    prodCas = [subProdDict["product"][key_][1] for key_ in subProdDict["product"].keys()]
    prodName = [subProdDict["product"][key_][0] for key_ in subProdDict["product"].keys()] 

    return subCas, subName, prodCas, prodName

