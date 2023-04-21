class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
aaMap = {
    'A': 'ALA',  # Alanine
    'C': 'CYS',  # Cysteine
    'D': 'ASP',  # Aspartic Acid
    'E': 'GLU',  # Glutamic Acid
    'F': 'PHE',  # Phenylalanine
    'G': 'GLY',  # Glycine
    'H': 'HIS',  # Histidine
    'I': 'ILE',  # Isoleucine
    'K': 'LYS',  # Lysine
    'L': 'LEU',  # Leucine
    'M': 'MET',  # Methionine
    'N': 'ASN',  # Asparagine
    'P': 'PRO',  # Proline
    'Q': 'GLN',  # Glutamine
    'R': 'ARG',  # Arginine
    'S': 'SER',  # Serine
    'T': 'THR',  # Threonine
    'V': 'VAL',  # Valine
    'W': 'TRP',  # Tryptophan
    'Y': 'TYR',  # Tyrosine
}
aaMap = dotdict(aaMap)  