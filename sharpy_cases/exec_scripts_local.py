import os
import sharpy.sharpy_main as sharpy_main

case_name_root='debug'

route = '/home/dmsm/m.colombo/Documents/sharpy/data/'+case_name_root+'/'
cases=[]

for file in sorted(os.listdir(route)):
    if file.endswith(".sharpy") & file.startswith("HALE_") :
        print(file)
        cases.append(sharpy_main.main(['',route+file]))

