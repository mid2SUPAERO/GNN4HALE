import os
import numpy as np
import sharpy
from sharpy.utils.h5utils import readh5
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import pickle
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython import display


def ismember(a, b):
    boolarray = np.array([i in b for i in a])
    u = np.unique(a[boolarray])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i, t in zip(a, boolarray)])
    return boolarray, index


def createpickle(filename, filelist, plotstuff=0):
    cases = list()
    list_tse = list()
    bfirst = 1
    for file in filelist:  # list(filter(lambda file : file.endswith(".sharpy")  & file.startswith("HALE_light_"),os.listdir(curdir)))
        print(file)
        h5file = file.replace(".sharpy", ".data.h5")
        path = file.replace(".sharpy", "")
        if os.path.exists(route + '/output/' + path + '/savedata/' + h5file):
            data = readh5(route + '/output/' + path + '/savedata/' + h5file)
            cases.append(data)

            # nodes: position, velocities,accelerations and loads
            timelen = len(data.data.structure.timestep_info)

            # number of variables position (3D)  + (displacements, twists)*2( = x, dx/dt)
            # eventually +  1 dim extra for lumped mass info,
            nvar_nodes = 3 + 6 * 2 + 1

            # only nodes at extreme of elements are taken so the total number is the number of nodes -
            n_tot_nodes = data.data.structure.timestep_info[0].pos.shape[0]
            n_nodes = n_tot_nodes - data.data.structure.timestep_info[0].postproc_cell['loads'].shape[0]

            # 4 dim extra for stiffness and mass info
            nvar_edges_in = 4
            nvar_edges_out = data.data.structure.timestep_info[0].postproc_cell['loads'].shape[1]
            n_edges = data.data.structure.timestep_info[0].postproc_cell['loads'].shape[0]

            # %indeces of global longitudinal motions
            # idx_glob=[0, 2, 6, 8, 10, 12] #x z xd zd thetad Uz
            nvar_globals = 1 + 10 * 2

          
            idx_nodes = np.setdiff1d(np.arange(data.data.structure.timestep_info[0].pos.shape[0]),
                                    data.data.structure.connectivities[:, 2])

            #i_loads_sel = [2,3,4]  # only longi loads

            new_nodes = np.arange(n_nodes)

            # i renumber connection since i'm skipping middle node. this will be for senders/receivers
            ba, idx = ismember(data.data.structure.connectivities[:, 0], idx_nodes)
            conn0 = new_nodes[idx]
            ba, idx = ismember(data.data.structure.connectivities[:, 1], idx_nodes)
            conn1 = new_nodes[idx]

            #i further downselect only one wing nodes (x0=0, y>0)
            idx_sel_nodes = (data.data.structure.timestep_info[0].pos[idx_nodes, 0]<0.1) & (data.data.structure.timestep_info[0].pos[idx_nodes, 1]>-0.1)
            idx_sel_loads=idx_sel_nodes[conn1]&idx_sel_nodes[conn0]

            n_nodes=sum(idx_sel_nodes)
            n_edges=sum(idx_sel_loads)

            senders = np.hstack([conn1[idx_sel_loads]])
            receivers = np.hstack([conn0[idx_sel_loads]])

          # %indeces of global longitudinal motions
            nodes = np.ndarray(shape=(n_nodes, nvar_nodes, timelen), dtype=float, order='F')
            edges_in = np.ndarray(shape=(n_edges, nvar_edges_in, timelen), dtype=float, order='F')
            edges_out = np.ndarray(shape=(n_edges, nvar_edges_out, timelen), dtype=float, order='F')
            globs = np.ndarray(shape=(nvar_globals, timelen), dtype=float, order='F')
            

            
            n_time = 0
            id_var = 0

            Ude = float(data.data.settings['StepUvlm']['velocity_field_input']['gust_parameters']['gust_intensity'])
            S = float(data.data.settings['StepUvlm']['velocity_field_input']['gust_parameters']['gust_length'])
            offset = float(data.data.settings['StepUvlm']['velocity_field_input']['offset'])

            idx_keep = list()  # this is to avoid too many points with nothign happening
            for t in range(timelen):
                # loads = data.data.structure.timestep_info[t].postproc_cell['loads']

                pos0 = data.data.structure.timestep_info[0].pos[idx_nodes[idx_sel_nodes], :]
                pos00 = np.vstack(
                    [np.zeros((1, 6)), np.reshape(data.data.structure.timestep_info[0].q[:-10], (n_tot_nodes - 1, 6))])[
                    idx_nodes[idx_sel_nodes], :]
                pos = np.vstack(
                    [np.zeros((1, 6)), np.reshape(data.data.structure.timestep_info[t].q[:-10], (n_tot_nodes - 1, 6))])[
                    idx_nodes[idx_sel_nodes], :] - pos00
                pos_dot = np.vstack(
                    [np.zeros((1, 6)), np.reshape(data.data.structure.timestep_info[t].dqdt[:-10], (n_tot_nodes - 1, 6))])[
                        idx_nodes[idx_sel_nodes], :]

                lumped_mass = np.zeros([data.data.structure.timestep_info[t].pos.shape[0],1])
                lumped_mass[data.data.structure.lumped_mass_nodes,:]=data.data.structure.lumped_mass
                lumped_mass=lumped_mass[idx_nodes[idx_sel_nodes]]
                # loads are zeroed compared to simulation start. only increment above 1g will be considered
                loads = data.data.structure.timestep_info[t].postproc_cell['loads'] - \
                        data.data.structure.timestep_info[0].postproc_cell['loads']
                loads= loads[idx_sel_loads,:]
                # HARDCODED SIGN CHANGE TO HAVE SYMMETRIC LOADS L/R
                # loads[8:16,:]=np.matmul(loads[8:16,:],np.diag(np.array([1, -1 ,1 ,1 ,1 ,-1])))
                loads = np.vstack([loads[0:3,:]-loads[1:4,:], loads[3,:]] )

                # I had a look at the mass and stiffness database.  there are three element types.
                # however there are only two linear indipendent values
                mass1 = np.expand_dims(data.data.structure.mass_db[data.data.structure.elem_mass, 0, 0], axis=1)
                mass2 = np.expand_dims(data.data.structure.mass_db[data.data.structure.elem_mass, 5, 5], axis=1)
                # there are 3 stiffness elements, but only linearly indipendent values either
                stif1 = np.expand_dims(data.data.structure.stiffness_db[data.data.structure.elem_stiffness, 0, 0], axis=1)
                stif2 = np.expand_dims(data.data.structure.stiffness_db[data.data.structure.elem_stiffness, 5, 5], axis=1)

                mass1=mass1[idx_sel_loads]
                mass2=mass2[idx_sel_loads]
                stif1=stif1[idx_sel_loads]
                stif2=stif2[idx_sel_loads]

                pos_glob = data.data.structure.timestep_info[t].q[-10:]
                vel_glob = data.data.structure.timestep_info[t].dqdt[-10:]

                if -pos_glob[0] - offset < S and -pos_glob[0] - offset >= 0:
                    U = Ude / 2 * (1 - np.cos(2 * np.pi * (-pos_glob[0] - offset) / S))
                    idx_keep.append(t)
                else:
                    U = 0
                    if -pos_glob[0] - offset < 4 * S and -pos_glob[0] - offset >= S:
                        idx_keep.append(t)

                nodes[:, :, t] = np.hstack([pos0, pos, pos_dot, lumped_mass])  # pos_ddot
                edges_in[:, :, t] = np.hstack([mass1, mass2, stif1, stif2])
                edges_out[:, :, t] = np.hstack([loads])
                globs[:, t] = np.hstack([pos_glob, vel_glob, U])

            if bfirst:
                # idx_keep.append(min(idx_keep)-1)
                nodes_all = nodes[:, :, idx_keep]
                edges_in_all = edges_in[:, :, idx_keep]
                edges_out_all = edges_out[:, :, idx_keep]
                globals_all = globs[:, idx_keep]
                # selglobs = globs[idx_glob,:]
                # globals_all = selglobs[:,idx_keep]
                bfirst = 0
            else:
                # idx_keep.append(min(idx_keep)-1)
                nodes_all = np.dstack([nodes_all, nodes[:, :, idx_keep]])
                edges_in_all = np.dstack([edges_in_all, edges_in[:, :, idx_keep]])
                edges_out_all = np.dstack([edges_out_all, edges_out[:, :, idx_keep]])
                globals_all = np.hstack([globals_all, globs[:, idx_keep]])
                # selglobs = globs[idx_glob,:]
                # globals_all = np.hstack([globals_all, selglobs[:,idx_keep] ])
            list_tse.append(globals_all.shape[1])

        # reduction of the structure to a subpart only
    # since flexible flight mechanics is seemingly dominated by wing I only take wing
    with open(filename , 'wb') as handle:
        pickle.dump([nodes_all,edges_in_all,edges_out_all,globals_all,senders,receivers,list_tse], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # HARDCODED NODE SELECTION AND SENDERS/RECEIVERS

    '''
    idx_nodes_selection=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]


    idx1, index1 =ismember(senders,idx_nodes_selection)
    idx2, index2 =ismember(receivers,idx_nodes_selection)

    idx_edges_selection = np.logical_and(idx1,idx2)

    nodes_all=nodes_all[idx_nodes_selection,:,:]                      
    edges_in_all=edges_in_all[idx_edges_selection,:,:]                      
    edges_out_all=edges_out_all[idx_edges_selection,:,:]     

    senders=    [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    receivers=  [ 0, 1, 2, 3, 4, 5, 6, 7, 0,  9, 10, 11, 12, 13, 14, 15]
    '''

    # to be reviewed
    # new_nodes_idx=np.arange(len(idx_nodes_selection))
    # senders= list(new_nodes_idx[index1[idx1]])
    # receivers= list(new_nodes_idx[index2[idx2]])

training_factor = 0.8
case_name_root='HALE_50_2elem'
route = '/scratch/dmsm/m.colombo/cases/data/'+case_name_root+'/'
filelist = os.listdir(route)

allcases = list(filter(lambda file: file.endswith(".sharpy"), filelist))
print(allcases)
#createpickle(route+'test.pickle', allcases[0:2], plotstuff=1)
createpickle(route+'HALE_50_2_tr.pickle', allcases[0:int(len(allcases) * training_factor)], plotstuff=1)
createpickle(route+'HALE_50_2_va.pickle', allcases[int(len(allcases) * training_factor):], plotstuff=1)

