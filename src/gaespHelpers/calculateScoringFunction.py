from Bio.PDB import PDBParser, Superimposer, PDBIO

def calculate_difference(pdb_file_1, pdb_file_2, residue_ids_file1, residue_ids_file2, residue_names_file1, residue_names_file2, output_file):
    # Initialize the PDB Parser
    parser = PDBParser()

    # Load the two structures
    struct_1 = parser.get_structure('struct_1', pdb_file_1)
    struct_2 = parser.get_structure('struct_2', pdb_file_2)

    # Get the atoms for the specified residues (by number) in each structure
    #[res for res in struct_1.get_residues() if res.id[1] in residue_ids_file1]

    atoms_struct_1_by_id = [atom for res in struct_1.get_residues() for atom in res.get_atoms() if res.id[1] in residue_ids_file1]
    atoms_struct_2_by_id = [atom for res in struct_2.get_residues() for atom in res.get_atoms() if res.id[1] in residue_ids_file2]

    # Get the atoms for the specified residues (by name) in each structure
    #[res.get_resname().strip(" ") for res in struct_1.get_residues()]

    atoms_struct_1_by_name = [atom for res in struct_1.get_residues() for atom in res.get_atoms() if res.get_resname().strip(" ") in residue_names_file1]
    atoms_struct_2_by_name = [atom for res in struct_2.get_residues() for atom in res.get_atoms() if res.get_resname().strip(" ") in residue_names_file2]

    # Combine the two lists of atoms
    atoms_struct_1 = atoms_struct_1_by_id + atoms_struct_1_by_name
    atoms_struct_2 = atoms_struct_2_by_id + atoms_struct_2_by_name

    # Make sure the residues exist in both structures
    if not atoms_struct_1 or not atoms_struct_2:
        raise ValueError('One of the specified residues does not exist in the structures')

    # Ensure the number of selected atoms is equal in both structures
    assert len(atoms_struct_1) == len(atoms_struct_2), "Mismatch in number of selected atoms"

    # Align the structures based on the selected atoms
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms_struct_1, atoms_struct_2)
    super_imposer.apply(struct_1.get_atoms())

    # Calculate the sum of the distances between the corresponding atoms in the two structures
    sum_difference = sum(atom_1 - atom_2 for atom_1, atom_2 in zip(atoms_struct_1, atoms_struct_2))

    # Save the superimposed structure to a PDB file
    if output_file: 
        io = PDBIO()
        io.set_structure(struct_1)
        io.save(output_file)    

    return sum_difference



if __name__ == "__main__":

    pdb_file_1 = 'data/raw/ortho12_FE_oxo.pdb'
    pdb_file_2 = 'data/raw/aKGD_FE_oxo.pdb'

    residue_ids_file1 = [210, 268]  # Insert your residue IDs here
    residue_ids_file2 = [167, 225]  # Insert your residue IDs here

    residue_names_file1 = ['FE2','FE', 'AKG']  # Insert your residue names here
    residue_names_file2 = ['FE2','FE', 'AKG']  # Insert your residue names here

    outputFile = "data/raw/superImposed.pdb"

    difference = calculate_difference(pdb_file_1, 
                                      pdb_file_2, 
                                      residue_ids_file1=residue_ids_file1,
                                      residue_ids_file2=residue_ids_file2,
                                      residue_names_file1=residue_names_file1,
                                      residue_names_file2=residue_names_file2,
                                      output_file=outputFile)
    print('Difference: ', difference)
    
