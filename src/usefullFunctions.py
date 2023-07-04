import os
import subprocess
import platform

def rename_files(path, filename, operating_system):
    # Walking through all folders and subfolders
    for dirpath, dirnames, filenames in os.walk(path):
        # Checking if the file exists
        if filename in filenames:
            # Creating the full path for the file
            full_path = os.path.join(dirpath, filename)
            # Getting the name of the current directory
            dirname = os.path.basename(dirpath)
            # Splitting the directory name by underscores
            parts = dirname.split('_')
            # Constructing the new filename based on the desired parts
            new_filename = parts[0][:9] + '_' + parts[7] + '_' + parts[14] + '_' + parts[15] + "".join("_"+str(i) for i in parts[16:]) +'.png'
            # Creating the full path for the new filename
            new_path = os.path.join(dirpath, new_filename)
            
            # Running the rename command in cmd
            try:
                if operating_system == 'Windows':
                    subprocess.run(f'ren "{full_path}" "{new_path}"', shell=True)
                elif operating_system == 'Linux':
                    subprocess.run(f'mv "{full_path}" "{new_path}"', shell=True)
                else:
                    print(f'Unsupported operating system: {operating_system}')
            except Exception as e:
                print("Error occurred:", e)

# Usage:
os_name = platform.system()
rename_files(
    path= r"C:\Users\kevin\OneDrive - ZHAW\KEVIN STUFF\ZHAW\_PYTHON_R\_GITHUB\reincatalyze\log\residora\target_alphaCarbon\2023_July_noSkipAA", 
    filename='2023_July_noSkipAA_G-Reincatalyze_resultOverview_withGrid.png',
    operating_system=os_name)
