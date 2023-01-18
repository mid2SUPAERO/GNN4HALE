import os
import sys
import argparse

sys.path.append('/scratch/dmsm/m.colombo/sharpy') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python37.zip') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python3.7') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python3.7/lib-dynload') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python3.7/site-packages') # go to parent dir

import sharpy.sharpy_main as sharpy_main

# CLI argument parser
parser = argparse.ArgumentParser(description="Sharpy job launcher")

parser.add_argument("-f", "--filename", type=str, default="",
                    help="config file to launch")
args = parser.parse_args()



sharpy_main.main(['',args.filename])

