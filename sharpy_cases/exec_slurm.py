# Launching Script (Thanks to Guillaume Bogopolosky)

import argparse
import os
from os import system
from subprocess import run

#Manually define list of config files
base_folder = '/scratch/dmsm/m.colombo/cases/data/HALE_50_2st2m_L/'

files=os.listdir(base_folder)
# Add only wanted elements
conf_files = []
for file in files:
    if file.endswith('.sharpy') :  #and not(file.startswith('HALE_'))
        conf_files.append(base_folder+file)

print(conf_files)

for i, config in enumerate(conf_files):
    print(f'Loading config file: {config}')
    opts = f"-J HALE_{i} launch_sharpy.ssh {config} "
    # run(["echo", opts], shell=True, capture_output=True, text=True)
    system(f"sbatch {opts}")
