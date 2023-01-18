# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model architectures for the demos in TensorFlow 2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from graph_nets import modules
import modules
from graph_nets import utils_tf
from six.moves import range
import sonnet as snt
import tensorflow as tf
import numpy as np
from sympy import lambdify
import sympy  as sy

NUM_LAYERS = 2 # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.
LATENT_SIZE_NALU = 4   # Hard-code latent layer sizes for demos.
AXISNORM=-1

class NAC(snt.Module):
    """Neural accumulator cell.

    The implementation is based on https://arxiv.org/abs/1808.00508.

    Attributes:
        input_shape: Number describing the shape of the input tensor.
        num_outputs: Integer describing the number of outputs.

    """
    def __init__(self, num_outputs, name=None):
        super(NAC, self).__init__(name=name)
        self.num_outputs = num_outputs

    @snt.once
    def _initialize(self, x):        
        input_size = x.shape[1]
        #shape = [int(input_shape[-1]), self.num_outputs]
        shape = [input_size, self.num_outputs]
        self.W_hat = tf.Variable("W_hat", shape=shape)  #initializer=tf.initializers.GlorotUniform()
        self.M_hat = tf.Variable("M_hat", shape=shape) #initializer=tf.initializers.GlorotUniform()

    def __call__(self, x):
        self._initialize(x)
        W = tf.nn.tanh(self.W_hat) * tf.nn.sigmoid(self.M_hat)
        return tf.matmul(x, tf.cast(W, 'float64'))

class NALU(snt.Module):
    """Neural arithmetic logic unit.

    The implementation is based on https://arxiv.org/abs/1808.00508.

    Attributes:
        input_shape: Number describing the shape of the input tensor.
        num_outputs: Integer describing the number of outputs.

    """
    def __init__(self, num_outputs, name='nalu'):
        super(NALU, self).__init__(name=name)
        self.num_outputs = num_outputs
      
    @snt.once
    def _initialize(self, x):
        input_size = x.shape[1]
        #shape = [int(input_shape[-1]), self.num_outputs]
        shape = [input_size, self.num_outputs]      
        self.eps = 1e-5
        
        r=np.asarray([-1, 1])          
        num_units=input_size
        grid=np.meshgrid(*[r]*num_units)   
        grid=tf.cast(tf.constant(np.array(grid).T.reshape(-1,num_units).T),'float64')
        shape=[2**num_units, self.num_outputs]        
        self.grid = grid      
       
        self.W_hat = tf.Variable(tf.random.truncated_normal(shape,stddev=0.2))              
        self.M_hat = tf.Variable(tf.random.truncated_normal(shape,stddev=0.2) ) #initializer=tf.initializers.GlorotUniform()
        self.G = tf.Variable(tf.random.truncated_normal(shape,stddev=0.2) ) #initializer=tf.initializers.GlorotUniform()

    def __call__(self, x):
        self._initialize(x)
        
        x=tf.matmul(x,self.grid)
        x=tf.nn.relu(x)
        # NAC cell

        W = tf.math.tanh( self.W_hat) * tf.math.sigmoid( self. M_hat)
        a = tf.matmul(x, tf.cast(W, 'float64') )

        # NALU
        m = tf.math.exp(tf.matmul(tf.math.log(tf.math.abs(x) + self.eps), tf.cast(W, 'float64')))
        g = tf.math.sigmoid(tf.matmul(x, tf.cast(self.G, 'float64') ))
        out = g * a + (1 - g) * m
        return out

class NumpyLambda(snt.Module):
    """
    array of numpy lambda functions
    """
    def __init__(self, num_outputs, name='NumpyLambda',list_lambda=[] ):
        super(NumpyLambda, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.list_lambda = list_lambda
        
    @snt.once
    def _initialize(self, x):
        input_size = x.shape[1]
        #shape = [int(input_shape[-1]), self.num_outputs]
        shape = [input_size, self.num_outputs]      
       
        '''
        x = sy.IndexedBase('x')
        i = sy.symbols('i', cls=Idx)
        s = sy.Product(x[i], (i, 1, input_size))
        '''

        self.list_var=['x'+str(i) for i in range(0,input_size)]

        lambda_matrix=[]
        for fun in self.list_lambda:
          lambda_matrix.append(lambdify(self.list_var,fun.equation,"numpy"))
        self.lambda_matrix=lambda_matrix

        
    def __call__(self, x):
        self._initialize(x)

        kwargs={}
        
        for el,iel in zip(self.list_var,range(x.shape[1])):
          kwargs.update({el:x[:,iel]})


        outnp=np.ndarray((x.shape[0], len(self.lambda_matrix)))
        for (fun,ifun) in zip(self.lambda_matrix,range(len(self.lambda_matrix))):

          outnp[:,ifun]=fun(**kwargs)

        out=tf.convert_to_tensor(outnp, dtype=x.dtype)
        return out


def make_seq_model(num_units=LATENT_SIZE,num_layers=NUM_LAYERS,num_outputs=None,addNormLayer=True):
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  #i use this initializer only for 
  #initializer = tf.initializers.HeNormal()  
  list_layer=[num_units]*num_layers
  list_act=[tf.nn.elu]*num_layers
  list_act[0]=tf.nn.tanh
  list_act[-1]=None

  activate_final=True
  if num_outputs!=None:
     list_layer.append(num_outputs)
     activate_final=False
        
 
  listlayers=list()

  for num_lay,actfun in zip(list_layer,list_act):
      listlayers.append(snt.Linear(num_lay))
      if not(actfun==None):
        listlayers.append(actfun)
  if addNormLayer == True:    
        listlayers.append( snt.LayerNorm(axis=AXISNORM, create_offset=True, create_scale=True))

  model = snt.Sequential(listlayers)               
       
  return model




def make_mlp_model(num_units=LATENT_SIZE,num_layers=NUM_LAYERS,num_outputs=None,addNormLayer=True):
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  #i use this initializer only for 
  #initializer = tf.initializers.HeNormal()  
  list_layer=[num_units]*num_layers
  activate_final=True
  if num_outputs!=None:
     list_layer.append(num_outputs)
     activate_final=False
        
  if addNormLayer == False:
     model = snt.Sequential([              
         snt.nets.MLP(list_layer , activate_final=activate_final), #, w_init=initializer,b_init=initializer                   
     ])         
  else: 
     model = snt.Sequential([              
         #still not so sure about what axis to apply normalization and if before/After 
         snt.nets.MLP(list_layer, activate_final=activate_final),        
         snt.LayerNorm(axis=AXISNORM, create_offset=True, create_scale=True),
     ])
        
       
  return model


def make_linear_model(num_outputs=None ):
    """Instantiates a new MLP, followed by LayerNorm.

    The parameters of each new MLP are not shared with others generated by
    this function.

    Returns:
      A Sonnet module which contains the MLP and LayerNorm.
    """
  

    model =snt.Linear(output_size=num_outputs)


    return model




class MLPGraphIndependent(snt.Module):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self,
               num_units=LATENT_SIZE,
               num_layers=NUM_LAYERS,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               addNormLayer=True,
               name="MLPGraphIndependent"):
        
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = modules.GraphIndependent(edge_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                              num_layers=num_layers,
                                                                              num_outputs=edge_output_size,
                                                                              addNormLayer=addNormLayer),
                                             node_model_fn=lambda:  make_mlp_model(num_units=num_units,
                                                                               num_layers=num_layers,
                                                                               num_outputs=node_output_size,
                                                                               addNormLayer=addNormLayer),
                                             global_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                                num_layers=num_layers,
                                                                                num_outputs=global_output_size,
                                                                                addNormLayer=addNormLayer))

  def __call__(self, inputs):
    return self._network(inputs)


class MLPCustomEncoder(snt.Module):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self,
               num_units=LATENT_SIZE,
               num_layers=NUM_LAYERS,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               addNormLayer=True,
               name="MLPCustomEncoder"):

    super(MLPCustomEncoder, self).__init__(name=name)

    if edge_output_size!=None:
        edge_model_fn= lambda: make_mlp_model(num_units=num_units,
                                              num_layers=num_layers,
                                              num_outputs=edge_output_size,
                                              addNormLayer=addNormLayer)
    else:
        if addNormLayer==True:
            edge_model_fn=  lambda: snt.Sequential([snt.LayerNorm(axis=AXISNORM, create_offset=True, create_scale=True)])
        else:
            edge_model_fn=None

    if node_output_size!=None:
        node_model_fn= lambda: make_mlp_model(num_units=num_units,
                                              num_layers=num_layers,
                                              num_outputs=node_output_size,
                                              addNormLayer=addNormLayer)
    else:
        if addNormLayer==True:
            node_model_fn=  lambda: snt.Sequential([snt.LayerNorm(axis=AXISNORM, create_offset=True, create_scale=True)])
        else:
            node_model_fn=None

    if global_output_size!=None:
        global_model_fn= lambda: make_mlp_model(num_units=num_units,
                                              num_layers=num_layers,
                                              num_outputs=global_output_size,
                                              addNormLayer=addNormLayer)
    else:
        if addNormLayer==True:
            global_model_fn= lambda: snt.Sequential([snt.LayerNorm(axis=AXISNORM, create_offset=True, create_scale=True)])
        else:
            global_model_fn=None

    self._network = modules.GraphIndependent(edge_model_fn=edge_model_fn,
                                            node_model_fn=node_model_fn,
                                            global_model_fn=global_model_fn)

  def __call__(self, inputs):
    return self._network(inputs)



class MLPGraphNetwork(snt.Module):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self,
               num_units=LATENT_SIZE,
               num_layers=NUM_LAYERS,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               addNormLayer=True,
               name="MLPGraphNetwork"):
        
        
  
    super(MLPGraphNetwork, self).__init__(name=name)
    self._network = modules.GraphNetwork(edge_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                              num_layers=num_layers,
                                                                              num_outputs=edge_output_size,
                                                                              addNormLayer=addNormLayer),
                                         node_model_fn=lambda:  make_mlp_model(num_units=num_units,
                                                                               num_layers=num_layers,
                                                                               num_outputs=node_output_size,
                                                                               addNormLayer=addNormLayer),
                                         global_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                                num_layers=num_layers,
                                                                                num_outputs=global_output_size,
                                                                                addNormLayer=addNormLayer))
    
    
  def __call__(self, inputs):
    return self._network(inputs)


class MLPFixedGraphNetwork(snt.Module):
    """GraphNetwork with MLP edge, node, and global models for graphs sharing the same structure"""

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 edgetoglobal_size=1,
                 nodetoglobal_size=1,
                 addNormLayer=True,
                 name="MLPFixedGraphNetwork"):
        super(MLPFixedGraphNetwork, self).__init__(name=name)
        self._network = modules.FixedGraphNetwork(edge_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                                  num_layers=num_layers,
                                                                                  num_outputs=edge_output_size,
                                                                                  addNormLayer=addNormLayer),
                                             node_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                                  num_layers=num_layers,
                                                                                  num_outputs=node_output_size,
                                                                                  addNormLayer=addNormLayer),
                                             global_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                                    num_layers=num_layers,
                                                                                    num_outputs=global_output_size,
                                                                                    addNormLayer=addNormLayer),
                                             edgetoglobalreducer= snt.Linear(output_size=edgetoglobal_size),
                                             nodetoglobalreducer= snt.Linear(output_size=nodetoglobal_size))

    def __call__(self, inputs):
        return self._network(inputs)

class SeqFixedGraphNetwork(snt.Module):
    """GraphNetwork with custom Sequential edge, node, and global models for graphs sharing the same structure"""

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 edgetoglobal_size=1,
                 nodetoglobal_size=1,
                 addNormLayer=True,
                 name="SeqFixedGraphNetwork"):
        super(SeqFixedGraphNetwork, self).__init__(name=name)
        self._network = modules.FixedGraphNetwork(edge_model_fn=lambda: make_seq_model(num_units=num_units,
                                                                                  num_layers=num_layers,
                                                                                  num_outputs=edge_output_size,
                                                                                  addNormLayer=addNormLayer),
                                             node_model_fn=lambda: make_seq_model(num_units=num_units,
                                                                                  num_layers=num_layers,
                                                                                  num_outputs=node_output_size,
                                                                                  addNormLayer=addNormLayer),
                                             global_model_fn=lambda: make_seq_model(num_units=num_units,
                                                                                    num_layers=num_layers,
                                                                                    num_outputs=global_output_size,
                                                                                    addNormLayer=addNormLayer),
                                             edgetoglobalreducer= snt.Linear(output_size=edgetoglobal_size),
                                             nodetoglobalreducer= snt.Linear(output_size=nodetoglobal_size))

    def __call__(self, inputs):
        return self._network(inputs)

class MLPFixedEncoderGraphNetwork(snt.Module):
    """Encodes Node to some new globals """

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 global_output_size=None,
                 nodetoglobal_size=1,
                 addNormLayer=True,
                 name="MLPFixedEncoderGraphNetwork"):
        super(MLPFixedEncoderGraphNetwork, self).__init__(name=name)
        self._network = modules.FixedGEncoderGraphNetwork(global_model_fn=lambda: make_mlp_model(num_units=num_units,
                                                                                    num_layers=num_layers,
                                                                                    num_outputs=global_output_size,
                                                                                    addNormLayer=addNormLayer),
                                             edgetoglobalreducer= None,
                                             nodetoglobalreducer= snt.Linear(output_size=nodetoglobal_size))

    def __call__(self, inputs):
        return self._network(inputs)

class MLPSimpleEncoderGraphNetwork(snt.Module):
    """Encodes Node to some new globals """

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 global_output_size=None,
                 nodetoglobal_size=1,
                 addNormLayer=True,
                 name="MLPSimpleEncoderGraphNetwork"):
        super(MLPSimpleEncoderGraphNetwork, self).__init__(name=name)
        self._network = modules.FixedGEncoderGraphNetwork(global_model_fn=lambda: make_linear_model(num_outputs=global_output_size),
                                             edgetoglobalreducer= None,
                                             nodetoglobalreducer= snt.Linear(output_size=nodetoglobal_size))

    def __call__(self, inputs):
        return self._network(inputs)



class SeqFixedEncoderGraphNetwork(snt.Module):
    """Linear aggregator, customized sequential update function """

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 global_output_size=None,
                 nodetoglobal_size=1,
                 addNormLayer=True,
                 name="SeqFixedEncoderGraphNetwork"):
        super(SeqFixedEncoderGraphNetwork, self).__init__(name=name)
        self._network = modules.FixedGEncoderGraphNetwork(global_model_fn=lambda: make_seq_model(num_units=num_units,
                                                                                    num_layers=num_layers,
                                                                                    num_outputs=global_output_size,
                                                                                    addNormLayer=addNormLayer),
                                             edgetoglobalreducer= None,
                                             nodetoglobalreducer= snt.Linear(output_size=nodetoglobal_size))

    def __call__(self, inputs):
        return self._network(inputs)

class EncodeProcessDecode(snt.Module):
  """Full encode-process-decode model.

  The model we explore includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
  - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each message-passing
    step.

                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

  def __init__(self,
               num_units=LATENT_SIZE,
               num_layers=NUM_LAYERS,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent(num_units=num_units,num_layers=num_layers)
    self._core = MLPGraphNetwork(num_units=num_units,num_layers=num_layers) # MLPGraphNetwork()
    self._decoder = MLPGraphIndependent(num_units=num_units,num_layers=num_layers)
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    self._output_transform = modules.GraphIndependent(
        edge_fn, node_fn, global_fn)

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
    #  output_ops.append(self._output_transform(decoded_op))
    #return output_ops
    return self._output_transform(decoded_op)



class Process(snt.Module):
  """Simple-process-extract model.
  - A "Core" graph net, which performs 1 round of processing (message-passing) and then outputs a linear combination of Latent       space

                                       
                                           
                  *------*     *------*     *---------*  
                  |      |     |      |     |         |  
  Input      ---->|Encode|---->| Core |---->| Linear  |---> Output(t+1)
                  |      |     |      |     | Layer   |
                  *------*     *------*     *---------* 
  
  
  
  """

  def __init__(self,
               num_units=LATENT_SIZE,
               num_layers=NUM_LAYERS,
               edge_core_size=None,
               node_core_size=None,
               global_core_size=None,               
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="Process"):
    super(Process, self).__init__(name=name)
    '''
    self._core = NALUGraphNetwork(num_units=num_units,
                             num_layers=num_layers,
                             edge_output_size=edge_core_size,
                             node_output_size=node_core_size,
                             global_output_size=global_core_size) # MLPGraphNetwork()
    '''
    self._core = MLPGraphNetwork(num_units=num_units,
                                 num_layers=num_layers,
                                 edge_output_size=edge_core_size,
                                 node_output_size=node_core_size,
                                 global_output_size=global_core_size,
                                 addNormLayer=False) # MLPGraphNetwork()
    
    self._encoder = MLPCustomEncoder(num_units=num_units,
                                 num_layers=num_layers,
                                 edge_output_size=None,
                                 node_output_size=None,
                                 global_output_size=None,
                                 addNormLayer=False) # MLPGraphNetwork()
    
    
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    
    self._output_transform = modules.GraphIndependent(
        edge_fn, node_fn, global_fn)

  def __call__(self, input_op, num_processing_steps):
    
    latent0=self._encoder(input_op)      
    latent = self._core(latent0)      
    return self._output_transform(latent)


class FixedEncodeProcessDecode(snt.Module):
    """Simple-process-extract model.
    - A "Core" graph net, which performs 1 round of processing (message-passing) and then outputs a linear combination of Latent       space



                    *------*     *------*     *---------*
                    |      |     |      |     |         |
    Input      ---->|Encode|---->| Core |---->| Linear  |---> Output(t+1)
                    |      |     |      |     | Layer   |
                    *------*     *------*     *---------*



    """

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 edge_core_size=None,
                 node_core_size=None,
                 global_core_size=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 edgetoglobal_size=None,
                 nodetoglobal_size=None,
                 name="FixedEncodeProcessDecode"):
        super(FixedEncodeProcessDecode, self).__init__(name=name)
        '''
        self._core = NALUGraphNetwork(num_units=num_units,
                                 num_layers=num_layers,
                                 edge_output_size=edge_core_size,
                                 node_output_size=node_core_size,
                                 global_output_size=global_core_size) # MLPGraphNetwork()
        '''
        self._core = MLPFixedGraphNetwork(num_units=num_units,
                                     num_layers=num_layers,
                                     edge_output_size=edge_core_size,
                                     node_output_size=node_core_size,
                                     global_output_size=global_core_size+nodetoglobal_size,
                                     edgetoglobal_size=edgetoglobal_size,
                                     nodetoglobal_size=nodetoglobal_size,
                                     addNormLayer=False)  # MLPGraphNetwork()

        self._encoder = MLPFixedEncoderGraphNetwork(num_units=num_units,
                                     num_layers=num_layers,
                                     global_output_size=global_core_size+nodetoglobal_size,
                                     nodetoglobal_size=nodetoglobal_size,
                                     addNormLayer=False)  # MLPGraphNetwork()


        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")

        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, name="node_output")

        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")

        self._output_transform = modules.GraphIndependent(
            edge_fn, node_fn, global_fn)

    def __call__(self, input_op, num_processing_steps):

        latent0 = self._encoder(input_op)
        latent = self._core(latent0)
        return self._output_transform(latent)


class SimpleEncodeProcessDecode(snt.Module):
    """Simple-process-extract model.
    - A "Core" graph net, which performs 1 round of processing (message-passing) and then outputs a linear combination of Latent       space



                    *------*     *------*     *---------*
                    |      |     |      |     |         |
    Input      ---->|Encode|---->| Core |---->| Linear  |---> Output(t+1)
                    |      |     |      |     | Layer   |
                    *------*     *------*     *---------*



    """

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 edge_core_size=None,
                 node_core_size=None,
                 global_core_size=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 edgetoglobal_size=None,
                 nodetoglobal_size=None,
                 name="SimpleEncodeProcessDecode"):
        super(SimpleEncodeProcessDecode, self).__init__(name=name)
       
        self._core = MLPFixedGraphNetwork(num_units=num_units,
                                     num_layers=num_layers,
                                     edge_output_size=edge_core_size,
                                     node_output_size=node_core_size,
                                     global_output_size=global_core_size,
                                     edgetoglobal_size=edgetoglobal_size,
                                     nodetoglobal_size=0,
                                     addNormLayer=False)  # MLPGraphNetwork()

        self._encoder = MLPFixedEncoderGraphNetwork(num_units=num_units,
                                     num_layers=num_layers,
                                     global_output_size=global_core_size+nodetoglobal_size,
                                     nodetoglobal_size=nodetoglobal_size,
                                     addNormLayer=False)  # MLPGraphNetwork()


        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")

        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, name="node_output")

        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")

        self._output_transform = modules.GraphIndependent(
            edge_fn, node_fn, global_fn)

    def __call__(self, input_op, num_processing_steps):

        latent0 = self._encoder(input_op)
        latent = self._core(latent0)
        return self._output_transform(latent)



class SeqEncodeProcessDecode(snt.Module):
    """Encode Process decode Structure with no loops using customizred sequential model and linear functions for global aggregations
    - 



                    *------*     *------*     *---------*
                    |      |     |      |     |         |
    Input      ---->|Encode|---->| Core |---->| Linear  |---> Output(t+1)
                    |      |     |      |     | Layer   |
                    *------*     *------*     *---------*



    """

    def __init__(self,
                 num_units=LATENT_SIZE,
                 num_layers=NUM_LAYERS,
                 edge_core_size=None,
                 node_core_size=None,
                 global_core_size=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 edgetoglobal_size=None,
                 nodetoglobal_size=None,
                 name="SeqEncodeProcessDecode"):
        super(SeqEncodeProcessDecode, self).__init__(name=name)
       
        self._core = SeqFixedGraphNetwork(num_units=num_units,
                                     num_layers=num_layers,
                                     edge_output_size=edge_core_size,
                                     node_output_size=node_core_size,
                                     global_output_size=global_core_size+nodetoglobal_size,
                                     edgetoglobal_size=edgetoglobal_size,
                                     nodetoglobal_size=nodetoglobal_size,
                                     addNormLayer=False)  # MLPGraphNetwork()

        self._encoder = SeqFixedEncoderGraphNetwork(num_units=num_units,
                                     num_layers=num_layers,
                                     global_output_size=global_core_size+nodetoglobal_size,
                                     nodetoglobal_size=nodetoglobal_size,
                                     addNormLayer=False)  # MLPGraphNetwork()


        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")

        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, name="node_output")

        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")

        self._output_transform = modules.GraphIndependent(
            edge_fn, node_fn, global_fn)

    def __call__(self, input_op, num_processing_steps):

        latent0 = self._encoder(input_op)
        latent = self._core(latent0)
        return self._output_transform(latent)










class ProcessLoop(snt.Module):
  """Simple-process-extract model.
  - A "Core" graph net, which performs n round of processing (message-passing) and then outputs a linear combination of Latent       space

                        Latent(t+1)               
                            ^                
                  *------*  |  *---------*  
                  |      |  |  |         |  
  Input      ---->| Core |---->| Linear  |---> Output(t+1)
  Latent(t+1)---->|      |     | Layer   |
                  *------*     *---------* 
  
  
  
  """

  def __init__(self,
               num_units=LATENT_SIZE,
               num_layers=NUM_LAYERS,
               edge_core_size=None,
               node_core_size=None,
               global_core_size=None,               
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="ProcessLoop"):
    super(ProcessLoop, self).__init__(name=name)
    
    self._core = NALUGraphNetwork(num_units=num_units,
                                 num_layers=num_layers,
                                 edge_output_size=edge_core_size,
                                 node_output_size=node_core_size,
                                 global_output_size=global_core_size) # MLPGraphNetwork()
    
    '''
    self._core = MLPGraphNetwork(num_units=num_units,
                                 num_layers=num_layers,
                                 edge_output_size=edge_core_size,
                                 node_output_size=node_core_size,
                                 global_output_size=global_core_size,
                                 addNormLayer=False) # MLPGraphNetwork()
    '''
    self._encoder = MLPGraphNetwork(num_units=num_units,
                                 num_layers=num_layers,
                                 edge_output_size=edge_core_size,
                                 node_output_size=None,
                                 global_output_size=None,
                                 addNormLayer=False) # MLPGraphNetwork()
    
    
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    
    self._output_transform = modules.GraphIndependent(
        edge_fn, node_fn, global_fn)

  def __call__(self, input_op, num_processing_steps):
    
    latent0=self._encoder(input_op)      
    latent = latent0

    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)      
    return self._output_transform(latent)



class Identity(snt.Module):
  """Simple-process-extract model.
  - A "toy" graph net, to check things

            
  
  """

  def __init__(self,
               num_units=LATENT_SIZE,
               num_layers=NUM_LAYERS,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="Identity"):
    super(Identity, self).__init__(name=name)
    '''
    self._core = NALUGraphNetwork(num_units=num_units,
                             num_layers=num_layers,
                             edge_output_size=edge_core_size,
                             node_output_size=node_core_size,
                             global_output_size=global_core_size) # MLPGraphNetwork()
    '''

    
    self._encoder = MLPCustomEncoder(num_units=num_units,
                                 num_layers=num_layers,
                                 edge_output_size=None,
                                 node_output_size=None,
                                 global_output_size=None,
                                 addNormLayer=True) # MLPGraphNetwork()
    
    
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size,with_bias=True, name="edge_output")
    
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size,with_bias=True, name="node_output")
    
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size,with_bias=True, name="global_output")
    
    self._output_transform = modules.GraphIndependent(
        edge_fn, node_fn, global_fn)

  def __call__(self, input_op, num_processing_steps):
    
    latent0=self._encoder(input_op)   
    return self._output_transform(latent0)
