def renameAtomsFromPDB(pdb_filename, pdb_output):
    with open(pdb_filename, 'r') as infile, open(pdb_output, 'w') as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                atom_serial = int(line[6:11])
                if atom_name == 'C':
                    new_atom_name = 'C' + str(atom_serial)
                    line = line[:12] + new_atom_name.ljust(4) + line[16:]
                if atom_name == 'O':
                    new_atom_name = 'O' + str(atom_serial)
                    line = line[:12] + new_atom_name.ljust(4) + line[16:]
                if atom_name == 'N':
                    new_atom_name = 'O' + str(atom_serial)
                    line = line[:12] + new_atom_name.ljust(4) + line[16:]
                outfile.write(line)