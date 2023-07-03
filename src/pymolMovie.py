import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import os
from os.path import join as pj
from typing import List
import subprocess

from pymol import cmd

def pymolMovie(csvPath, pdbPath):
    # Load the data
    df = pd.read_csv(csvPath)
    df = df.iloc[:250,:]

    # Load your structure into pymol
    cmd.load(pdbPath)

    # Create the movie
    total_frames = df['generation'].nunique()

    nrow = len(df)

    # Initialize a dict to keep track of current generation and frame
    current_generation = {'generation': None, 'frame': 1}

    for index, row in df.iterrows():
        print(f"({index}/{nrow})", end = "\r")
        if row['generation'] != current_generation['generation']:
            # Create a new state for a new generation
            current_generation['generation'] = row['generation']
            cmd.create(f'state_{current_generation["frame"]}', 'all', 1, current_generation['frame'])
            cmd.frame(current_generation['frame'])
            cmd.hide('spheres', 'all')
            cmd.color('white', 'all')  # Reset color to default white
            current_generation['frame'] += 1

        # Select the residue to mutate
        selection = f'resi {row["mutationResidue"]}'
        
        # Color the selected residue
        _ = cmd.color('red', selection)

        # Now make it more visually distinct by showing it as a sphere
        #_ = cmd.show('spheres', selection)

    # Set the total number of frames for the movie
#    cmd.mset(f'1 x{total_frames}')
    cmd.mset(f'1 x{nrow}')

    # Rewind to the first frame
    cmd.frame(1)

    cmd.mpng('data/wasteBin/movie_frames.png')

    command = """ffmpeg -r 10 -i /home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/movie_frames%04d.png -c:v libx264 -vf "fps=25,format=yuv420p" /home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/movie.mp4"""
    ps = subprocess.Popen([command],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate()    
