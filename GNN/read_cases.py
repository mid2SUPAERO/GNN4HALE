import os
import numpy as np
import sharpy
from sharpy.utils.h5utils import readh5
import pandas as pd
def ismember(a, b):
    boolarray = np.array([i in b for i in a])
    u = np.unique(a[boolarray])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,boolarray)])
    return boolarray, index


curdir= os.getcwd()
for file in os.listdir(curdir):
    if file.endswith(".sharpy"):
        h5file=file.replace(".sharpy",".data.h5")
        path=file.replace(".sharpy","")
        data = readh5(curdir+'/output/'+path+'/savedata/'+h5file)

        #nodes: position, velocities,accelerations and loads
        timelen = len(data.data.structure.timestep_info)
        # 1 dim extra for lumped mass info
        nvar_nodes = data.data.structure.timestep_info[0].pos.shape[1] +data.data.structure.timestep_info[0].pos_dot.shape[1] + data.data.structure.timestep_info[0].pos_ddot.shape[1]+1
        n_nodes = data.data.structure.timestep_info[0].pos.shape[0]-data.data.structure.timestep_info[0].postproc_cell['loads'].shape[0]

        # 4 dim extra for stiffness and mass info
        nvar_edges=data.data.structure.timestep_info[0].postproc_cell['loads'].shape[1]+4
        n_edges = data.data.structure.timestep_info[0].postproc_cell['loads'].shape[0]

        nvar_globals = data.data.structure.timestep_info[0].for_pos.shape[0]+data.data.structure.timestep_info[0].for_vel.shape[0]



        nodes = np.ndarray(shape=(n_nodes,nvar_nodes,timelen), dtype=float, order='F')
        edges = np.ndarray(shape=(n_edges,nvar_edges,timelen), dtype=float, order='F')
        globals = np.ndarray(shape=(nvar_globals,timelen), dtype=float, order='F')

        #i keep only nodes that are borders of the elements skip middle node
        idx_nodes = np.setdiff1d(np.arange(data.data.structure.timestep_info[0].pos.shape[0]), data.data.structure.connectivities[:, 2])
        new_nodes=np.arange(n_nodes)

        #i renumber connection since i'm skipping middle node. this will be for senders/receivers
        ba, idx = ismember(data.data.structure.connectivities[:, 0], idx_nodes)
        conn0=new_nodes[idx]
        ba, idx = ismember( data.data.structure.connectivities[:, 1], idx_nodes)
        conn1=new_nodes[idx]

        senders=np.hstack([conn0,conn1])
        receivers=np.hstack([conn1,conn0])

        n_time=0
        id_var=0
        for t in range(timelen):
            #loads = data.data.structure.timestep_info[t].postproc_cell['loads']
            pos = data.data.structure.timestep_info[t].pos[idx_nodes,:]
            pos_dot = data.data.structure.timestep_info[t].pos_dot[idx_nodes,:]
            pos_ddot = data.data.structure.timestep_info[t].pos_ddot[idx_nodes,:]

            lumped_mass = np.zeros([data.data.structure.timestep_info[t].pos.shape[0],1])
            lumped_mass[data.data.structure.lumped_mass_nodes,:]=data.data.structure.lumped_mass

            loads=data.data.structure.timestep_info[t].postproc_cell['loads']

            #I had a look at the mass and stiffness database.  there are three element types.
            #however there are only two linear indipendent values
            mass1 = data.data.structure.mass_db[ data.data.structure.elem_mass, 0, 0]
            mass2 = data.data.structure.mass_db[data.data.structure.elem_mass, 5, 5]
            #there are 3 stiffness elements, but only linearly indipendent values either
            stif1 = data.data.structure.stiffness_db[ data.data.structure.elem_stiffness,0 , 0]
            stif2 = data.data.structure.stiffness_db[data.data.structure.elem_stiffness, 5, 5]

            pos_glob = data.data.structure.timestep_info[t].for_pos
            vel_glob = data.data.structure.timestep_info[t].for_vel

            nodes[:,:,t]=np.hstack( [pos,pos_dot,pos_ddot,lumped_mass])
            edges[:,:,t]=np.hstack( [loads, mass1, mass2, stif1, stif2])
            globals[:,t]=np.hstack( [pos_glob,vel_glob])

        if not('nodes_all' in locals()):
            nodes_all = nodes
            edges_all = edges
            globals_all = globals
        else:
            nodes_all = np.dstack([nodes_all, nodes])
            edges_all = np.dstack([edges_all, edges])
            globals_all = np.vstack([globals_all, globals])
