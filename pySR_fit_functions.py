import numpy as np
import pickle

import pysr
#pysr.install()

from pysr import PySRRegressor

with open('/home/dmsm/m.colombo/Documents/graphnets/GNN/HALE_ver4-pySR_data.pickle', 'rb') as handle:
    encoder_collected_globals = pickle.load(handle)    
    encoder_updated_globals = pickle.load(handle)
    core_collected_edges = pickle.load(handle)
    core_updated_edges = pickle.load(handle)
    core_collected_nodes = pickle.load(handle)
    core_updated_nodes = pickle.load(handle)
    core_collected_globals = pickle.load(handle)
    core_updated_globals = pickle.load(handle)
    


symbolic_encoder_globals_models=[]
for ydim in range(encoder_updated_globals.shape[1]):
    model = PySRRegressor(
    procs=4,
    populations=8,
    ncyclesperiteration=500, 
    niterations=1000,  
    timeout_in_seconds=60 * 2 ,
    maxsize=30,
    maxdepth=8,
    model_selection="best",  # Result is mix of simplicity+accuracy    
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["exp","ceil","floor","abs"],
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    update=False,            
    )

    
    y=encoder_updated_globals[:,ydim]
    X=encoder_collected_globals
    model.fit(X, y)
    symbolic_encoder_globals_models.append(model.get_best())

symbolic_core_globals_models=[]
for ydim in range(core_updated_globals.shape[1]):
    model = PySRRegressor(
    procs=4,
    populations=8,
    ncyclesperiteration=500, 
    niterations=1000,  
    timeout_in_seconds=60 * 2 ,
    maxsize=30,
    maxdepth=8,
    model_selection="best",  # Result is mix of simplicity+accuracy    
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["exp","ceil","floor","abs"],
    select_k_features=5,       	    
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    update=False,            
    )

    
    y=core_updated_globals[:,ydim]
    X=core_collected_globals
    model.fit(X, y)
    symbolic_core_globals_models.append(model.get_best())

symbolic_core_nodes_models=[]
for ydim in range(core_updated_nodes.shape[1]):
    model = PySRRegressor(
    procs=4,
    populations=8,
    ncyclesperiteration=500, 
    niterations=1000,  
    timeout_in_seconds=60 * 2 ,
    maxsize=30,
    maxdepth=8,
    model_selection="best",  # Result is mix of simplicity+accuracy    
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["exp","ceil","floor","abs"],
    select_k_features=5,       	    
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    update=False,            
    )

    
    y=core_updated_nodes[:,ydim]
    X=core_collected_nodes
    model.fit(X, y)
    symbolic_core_nodes_models.append(model.get_best())

symbolic_core_edges_models=[]
for ydim in range(core_updated_edges.shape[1]):
    model = PySRRegressor(
    procs=4,
    populations=8,
    ncyclesperiteration=500, 
    niterations=1000,  
    timeout_in_seconds=60 * 2 ,
    maxsize=30,
    maxdepth=8,
    model_selection="best",  # Result is mix of simplicity+accuracy    
    binary_operators=["*", "+", "-", "/"],
    #unary_operators=["square", "cube", "exp","ceil","floor","abs"],
    unary_operators=["exp","ceil","floor","abs"],
    select_k_features=7,       	    
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    update=False,            
    )

    
    y=core_updated_edges[:,ydim]
    X=core_collected_edges
    model.fit(X, y)
    symbolic_core_edges_models.append(model.get_best())


print(symbolic_core_globals_models)
with open('/home/dmsm/m.colombo/Documents/graphnets/GNN/symbolic_regression_results.pk', 'wb') as sr_model_file:
    pickle.dump([symbolic_encoder_globals_models,symbolic_core_globals_models,symbolic_core_nodes_models,symbolic_core_edges_models], sr_model_file)