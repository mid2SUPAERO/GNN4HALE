# Launching Script (Thanks to Guillaume Bogopolosky)

import argparse
import os
from os import system
from subprocess import run
import shutil




directory='HALE_50_2st2m_L'
unitloads=0
if unitloads:
    filename_tr=directory+'c_unitloads_tr.pickle'
    filename_va=directory+'c_unitloads_va.pickle'
else:
    filename_tr=directory+'c_loads_tr.pickle'
    filename_va=directory+'c_loads_va.pickle'


opts = f"-J read_data launch_read_data.ssh {directory} {filename_tr} {filename_va}  {str(unitloads)} "
# run(["echo", opts], shell=True, capture_output=True, text=True)
system(f"sbatch {opts}")
