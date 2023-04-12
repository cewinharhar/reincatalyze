def getMutationsList(wildtype_sequence, mutant_sequence):
    """
    Generate a mutation list from wildtype and mutant sequences.

    :param wildtype_sequence: String containing the wildtype amino acid sequence
    :param mutant_sequence: String containing the mutant amino acid sequence
    :return: List of tuples with residue index, original residue, and target residue
             (e.g. [(5, 'A', 'T'), (10, 'L', 'M')])
    """

    if len(wildtype_sequence) != len(mutant_sequence):
        raise ValueError("Wildtype and mutant sequences must have the same length.")

    mutation_list = []

    for idx, (wt_aa, mt_aa) in enumerate(zip(wildtype_sequence, mutant_sequence), start=1):
        if wt_aa != mt_aa:
            mutation_list.append((idx, wt_aa, mt_aa))

    return mutation_list