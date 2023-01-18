#script to read impulse cases
#3D output (not only wing)
#

import sys
import os
import numpy as np
import sharpy
from sharpy.utils.h5utils import readh5
import pandas as pd
from IPython.display import Image
import pickle
import random

sys.path.append('/scratch/dmsm/m.colombo/sharpy') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python37.zip') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python3.7') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python3.7/lib-dynload') # go to parent dir
sys.path.append('/home/dmsm/m.colombo/.conda/envs/sharpy_env/lib/python3.7/site-packages') # go to parent dir


import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython import display
import re
import argparse

def ismember(a, b):
    boolarray = np.array([i in b for i in a])
    u = np.unique(a[boolarray])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,boolarray)])
    return boolarray, index



def createpickle(filename,filelist,plotstuff=0,bzeroloads=1):
    cases=list()
    list_tse=list()
    bfirst=1
    
    nodes_list=list() 
    edges_list=list() 
    global_list=list()
    selglobal_list=list() 
    
    for file in filelist:  #list(filter(lambda file : file.endswith(".sharpy")  & file.startswith("HALE_light_"),os.listdir(curdir)))
     
        
        if (os.path.exists(file)):    
            print(file)
            data = readh5(file)
            cases.append(data)

            #nodes: position, velocities,accelerations and loads
            timelen = int(len(data.data.structure.timestep_info))


            #number of variables position (3D)  + (displacements, twists)*2( = x, dx/dt)  
            #eventually +  1 dim extra for lumped mass info,              
            nvar_nodes = 3+6*2 

            #only nodes at extreme of elements are taken so the total number is the number of nodes - 
            n_tot_nodes=data.data.structure.timestep_info[0].pos.shape[0]
            n_nodes = n_tot_nodes-data.data.structure.timestep_info[0].postproc_cell['loads'].shape[0]

            # 4 dim extra for stiffness and mass info
            nvar_edges_in=4
            idx_loads=[0,2, 4];
            nvar_edges_out=len(idx_loads)#data.data.structure.timestep_info[0].postproc_cell['loads'].shape[1]
            n_edges = data.data.structure.timestep_info[0].postproc_cell['loads'].shape[0]

            #%indeces of global longitudinal motions
            idx_glob=[0, 2, 6, 8, 10, 12, 14 ,16 ,18 ,20, 21] #x z xd zd thetad Uz        
            nvar_globals = 2+10*2 #hard coded number of independent variables

            

            #%indeces of global longitudinal motions
            nodes = np.ndarray(shape=(n_nodes,nvar_nodes,timelen), dtype=float, order='F')
            edges_in = np.ndarray(shape=(n_edges,nvar_edges_in,timelen), dtype=float, order='F')
            edges_out = np.ndarray(shape=(n_edges,nvar_edges_out,timelen), dtype=float, order='F')
            globs = np.ndarray(shape=(nvar_globals,timelen), dtype=float, order='F')

            idx_nodes = np.setdiff1d(np.arange(data.data.structure.timestep_info[0].pos.shape[0]), data.data.structure.connectivities[:, 2])

           
            new_nodes=np.arange(n_nodes)

            #i renumber connection since i'm skipping middle node. this will be for senders/receivers
            ba, idx = ismember(data.data.structure.connectivities[:, 0], idx_nodes)
            conn0=new_nodes[idx]
            ba, idx = ismember( data.data.structure.connectivities[:, 1], idx_nodes)
            conn1=new_nodes[idx]

            senders=np.hstack([conn1])
            receivers=np.hstack([conn0])

            n_time=0
            id_var=0


            idx_keep=list()  #this is to avoid too many points with nothign happening
            for t in range(timelen):
                #loads = data.data.structure.timestep_info[t].postproc_cell['loads']

                pos0 = data.data.structure.timestep_info[0].pos[idx_nodes,:]                                       
                pos00 = np.vstack([np.zeros((1,6)),np.reshape(data.data.structure.timestep_info[0].q[:-10],(n_tot_nodes-1,6)) ])[idx_nodes,:]
                pos =  np.vstack([np.zeros((1,6)),np.reshape(data.data.structure.timestep_info[t].q[:-10],(n_tot_nodes-1,6)) ])[idx_nodes,:]-pos00
                pos_dot =  np.vstack([np.zeros((1,6)),np.reshape(data.data.structure.timestep_info[t].dqdt[:-10],(n_tot_nodes-1,6)) ])[idx_nodes,:]


                #lumped_mass = np.zeros([data.data.structure.timestep_info[t].pos.shape[0],1])
                #lumped_mass[data.data.structure.lumped_mass_nodes,:]=data.data.structure.lumped_mass

                #loads are zeroed compared to simulation start. only increment above 1g will be considered            
                loads=data.data.structure.timestep_info[t].postproc_cell['loads']-bzeroloads*data.data.structure.timestep_info[0].postproc_cell['loads']


                #HARDCODED SIGN CHANGE TO HAVE SYMMETRIC LOADS L/R
                #loads[8:16,:]=np.matmul(loads[8:16,:],np.diag(np.array([1, -1 ,1 ,1 ,1 ,-1])))
                #unitloads = np.vstack([loads[0:7,:]-loads[1:8,:], loads[7,:], loads[8:15,:]-loads[9:16,:], loads[15,:], loads[16:,:]] )               


                #I had a look at the mass and stiffness database.  there are three element types.
                #however there are only two linear indipendent values
                mass1 =  np.expand_dims(data.data.structure.mass_db[ data.data.structure.elem_mass, 0, 0],axis=1)
                mass2 = np.expand_dims(data.data.structure.mass_db[data.data.structure.elem_mass, 5, 5],axis=1)
                #there are 3 stiffness elements, but only linearly indipendent values either
                stif1 = np.expand_dims(data.data.structure.stiffness_db[ data.data.structure.elem_stiffness,0 , 0],axis=1)
                stif2 = np.expand_dims(data.data.structure.stiffness_db[data.data.structure.elem_stiffness, 5, 5],axis=1)

                pos_glob = data.data.structure.timestep_info[t].q[-10:]-data.data.structure.timestep_info[0].q[-10:]
                vel_glob = data.data.structure.timestep_info[t].dqdt[-10:]


                idx_keep.append(t)


                drive, path = os.path.splitdrive(file)
                path, filepart = os.path.split(path)
                filepart = os.path.splitext(filepart)[0]
                filepart = filepart.split('.')[0]

                match = re.match(r"([a-zA-Z_.]+)([0-9]+)(_T)([0-9]+)", filepart, re.I)
                if match:
                    items = match.groups()                
                else:
                    print(filepart)
                duration=float(items[3])/10 / data.data.settings['DynamicCoupled']['dt']

               
                #i need to extract from name time as well
                if (t>8) & (t<duration+8) & filepart.endswith('ail'):                           
                    Fail=float(items[1])
                    #print(Fail)
                else:
                    Fail=0
                    
                if (t>8) & (t<duration+8)  & filepart.endswith('elev'):       
                    #print(file+items[1])
                    Felev=float(items[1])
                    #print(Felev)
                else:
                    Felev=0    

                nodes[:,:,t]=np.hstack( [pos0,pos,pos_dot]) #pos_ddot
                edges_in[:,:,t]=np.hstack( [mass1, mass2, stif1, stif2])  
                edges_out[:,:,t]=np.hstack( [loads[:,idx_loads]])  
                globs[:,t]=np.hstack( [pos_glob,vel_glob,Fail,Felev])

            if bfirst:
                #idx_keep.append(min(idx_keep)-1)
                nodes_all = nodes[:,:,idx_keep]
                edges_in_all = edges_in[:,:,idx_keep]
                edges_out_all = edges_out[:,:,idx_keep]
                #globals_all = globs[:,idx_keep]
                selglobs = globs[idx_glob,:]
                globals_all = selglobs[:,idx_keep]
                bfirst=0
            else:
                #idx_keep.append(min(idx_keep)-1)
                nodes_all = np.dstack([nodes_all, nodes[:,:,idx_keep]])
                edges_in_all = np.dstack([edges_in_all, edges_in[:,:,idx_keep]])
                edges_out_all = np.dstack([edges_out_all, edges_out[:,:,idx_keep]])
                #globals_all = np.hstack([globals_all, globs[:,idx_keep] ])
                selglobs = globs[idx_glob,:]
                globals_all = np.hstack([globals_all, selglobs[:,idx_keep] ])


            nodes_list.append(nodes[:,:,idx_keep])
            edges_list.append(edges_out[:,:,idx_keep])
            global_list.append(globs[:,idx_keep])
            selglobal_list.append(selglobs[:,idx_keep])


            list_tse.append(globals_all.shape[1])   


    with open(filename , 'wb') as handle:
        pickle.dump([nodes_all,edges_in_all,edges_out_all,globals_all,senders,receivers,list_tse], handle, protocol=pickle.HIGHEST_PROTOCOL)


    if plotstuff:
        fig,ax = plt.subplots(4,1)
        
        for i in range(4):        
            if i==0:
                el=np.transpose(nodes_all,(2,0,1))
                el=el.reshape(([el.shape[1]*el.shape[0],el.shape[2]]))
            elif i==1:
                el=np.transpose(edges_in_all,(2,0,1))
                el=el.reshape(([el.shape[1]*el.shape[0],el.shape[2]]))
            elif i==2:
                el=np.transpose(edges_out_all,(2,0,1))
                el=el.reshape(([el.shape[1]*el.shape[0],el.shape[2]]))
            else:
                el=globals_all.T
                

            el=np.abs(el)

            dfnda=pd.DataFrame(el)
            dfvals= pd.DataFrame([dfnda.max(), dfnda.min(),dfnda.quantile(q=0.25),dfnda.quantile(q=0.75),dfnda.median()])
            dfvals.index=['whislo','whishi','q1','q3','med']
            labels=list(dfvals.columns)
            bxp_stats=dfvals.apply(lambda x: {'med':x.med,'q1':x.q1,'q3':x.q3,'whislo':x.whislo,'whishi':x.whishi})
            for index,item in enumerate(bxp_stats):
                item.update({'label':labels[index]})

            ax[i].set_yscale('log')
            ax[i].bxp(bxp_stats, showfliers=False)

        plt.show()
            
    return nodes_list,edges_list,global_list,selglobal_list





# CLI argument parser
parser = argparse.ArgumentParser(description="read data launcher")
parser.add_argument("-d", "--directory", type=str, default="", help="subfodler of input folder")
parser.add_argument("-t", "--training_factor", type=float, default="0.8", help="ratio data for training / validation")
args = parser.parse_args()



training_factor=args.training_factor

allcases  = list()
pathlist=  os.listdir(args.directory+'/output/')     

for path in pathlist:
    allcases.append(args.directory +'/output/'+path+'/savedata/'+path+'.data.h5')

random.shuffle(allcases)
#print(allcases)




nodes_list,edges_list,global_list,selglobal_list = createpickle(args.directory+'/',+'_tr.pickle',allcases[0:int(len(allcases)*training_factor)])     
nodes_list,edges_list,global_list,selglobal_list = createpickle(args.directory+'_va.pickle',allcases[int(len(allcases)*training_factor):])     
  