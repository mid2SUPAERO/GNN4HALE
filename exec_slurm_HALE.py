# Launching Script (Thanks to Guillaume Bogopolosky)

import argparse
import os
from os import system
from subprocess import run
import shutil

conf_file = 'HALE_50_2st2m'


filename_tr='HALE_50_2st2m_loads_tr.pickle'
filename_va='HALE_50_2st2m_loads_va.pickle'
directory='/scratch/dmsm/m.colombo/cases/data/HALE_50_2st2m'
study_name='loads_ltl_dsh'

os.makedirs(os.path.join(directory,study_name))
shutil.copyfile(os.path.join(directory,filename_tr), os.path.join(directory,study_name,filename_tr))
shutil.copyfile(os.path.join(directory,filename_va), os.path.join(directory,study_name,filename_va))


parameter1_list=[1,2,4]
parameter2_list=[1,3,5]

for parameter1 in parameter1_list:
    for parameter2 in parameter2_list:
        opts = f"-J train_hale launch_train_HALE.ssh {os.path.join(directory,study_name)} {filename_tr} {filename_va}  {str(parameter1)} {str(parameter2)}"
        # run(["echo", opts], shell=True, capture_output=True, text=True)
        system(f"sbatch {opts}")
