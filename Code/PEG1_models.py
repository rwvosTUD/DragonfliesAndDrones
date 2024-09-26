import torch
from torch.utils.data import DataLoader

import numpy as np
import warnings
import random
import os

##
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override

from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer


from ray.rllib.algorithms.ppo import PPO, PPOConfig

#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.sample_batch import SampleBatch # REMOVE?

from ray.rllib.utils.spaces import space_utils

##
import pytorch_lightning as pl

import ncps.torch as ncpstorch
import ncps.wirings as ncpswirings
import ncps



def seed_everything(seed):
    # TODO SEMINAR
    ''' 
    https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    
    seed; numpy, torch, tensorflow, ray, scipy, pandas, 
    ''' 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    pl.seed_everything(seed, workers = True)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


#%% UTILITY FUNCTIONS

class BCdataset(torch.utils.data.Dataset):
    
    def __init__(self, x, y, tensors = True):
        
        if not tensors:
            dtype = torch.float
            x = torch.as_tensor(x, dtype = dtype) 
            y = torch.as_tensor(y, dtype = dtype) 
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)
        return 
    
    def setup_dataloader(self, batch_size = 512, shuffle = True):
        
        dataloader = DataLoader(self, 
                                batch_size=batch_size, shuffle=True,
                          # num workers etc is possible here,
                          ) 
        return dataloader
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        x_s = self.x[idx]
        y_s = self.y[idx]
        return x_s, y_s





#%% SUPER MODEL


class CustomTorchSuperModel(torch.nn.Module):
    
    def __init__(self, RNN_type = False):
        super().__init__()
            
        self.RNN_type = RNN_type
        
        return 
        
    def predict(self, inputs: torch.tensor, rnn_state: list = [], seq_lens: list = [], 
                np_out = True, unsquash = True,
                ):
        '''
        

        Parameters
        ----------
        inputs : torch.tensor
            Inputs of dimension (T,F), even if not RNN model. Inputs are
            expected to originate from env.get_next_input to ensure correct format.
        rnn_state : List
            DESCRIPTION.
        seq_lens : List
            DESCRIPTION.
        np_out : Boolean, optional
            Whether to output in numpy array format. The default is True.
            Involves detaching and converting data to CPU.

        Returns
        -------
        action_mu: np.array/torch.tensor with mean action of policy distribution

        '''
        ## evaluation
        self.eval()
        assert not self.training, 'Training'
        
        act_dim = self.num_outputs//2
        ##
        with torch.no_grad(): # dont know if this works at this outer level, TODO PREDICT_ACTION FUNCTION( with e.g. value_out = False)?
            # action_dist contains action_(mu, logstd)
            # see source https://stackoverflow.com/questions/69496570/neural-network-outputs-in-rllib-ppo-algorithm
            if self.RNN_type:
                '''
                # recurrent network type, i.e. RecurrentNetwork
                if bool(rnn_state): # if empty list
                    rnn_state = self.get_initial_state() # get state
                
                action_dist, rnn_state = self.forward_rnn(inputs, 
                                                          rnn_state, [])
                action_dist = action_dist[-1,:] # (T,a_dim*2) -> (a_dim*2,); mu & logstd
                '''
                # recurrent network type in TorchModelV2 wrapper
                input_dict ={'prev_n_obs': inputs}
                action_dist, rnn_state = self.forward(input_dict, 
                                                          rnn_state, [])
            else:
                # non-recurrent network type
                input_dict = {'obs_flat': inputs[-1,:]} # (F,) notice T dim is cutoff by [-1,] which ensures we always use most recent T
                action_dist, _ = self.forward(input_dict, [], []) # forward expects obs_flat i.e. flattened tensor of inputs
                # action_dist dim (a_dim*2,); mu & logstd

        
        ## finalize
        self.train()
        
        action_mu = action_dist[:act_dim].detach() # (a_dim*2,) -> (a_dim,), select mu only i.e. no exploration
        if np_out: 
            action_mu = action_mu.numpy() # (a_dim,) detach for grad graph, all is on CPU!
            # note that if not numpy out we have unsquased 
            if unsquash:
                # [-1,1] inputs back to obs space
                action_mu = space_utils.unsquash_action(action_mu, self.action_space)
                # benchmarks dont need this as theyre not defined in this squased space
        return action_mu
    
    
#%% BENCHMARK MODELS (pre-defined parameters!)

class BenchmarkTorch_StaticGainPN(TorchModelV2, CustomTorchSuperModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, ray = True):
        CustomTorchSuperModel.__init__(self)
        self.network = torch.nn.Linear(1,1, bias = False) # to get device
        if not ray:
            return 
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
    
        self.num_outputs = num_outputs
        obs_size  = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs = obs_size
        ##
        self.custom_model_config = model_config['custom_model_config']
        self.SAdim_s = int(self.custom_model_config['SA']['Dimension'])
        self.SAdim_a = int(self.custom_model_config['SA']['DOF'])
        self.SAcode_num = int(str(self.SAdim_s) + str(self.SAdim_a))
        
        self.gain_setting = self.custom_model_config.get('gain',1.)
        ##
        if self.SAcode_num in [21,32,33,34]:
            if self.SAcode_num == 21:
                input_indices = [[1]]
            elif self.SAcode_num == 32:
                input_indices = [[1,3]]
            elif self.SAcode_num == 33:
                input_indices = [[1,3,5]]
            elif self.SAcode_num == 34:
                input_indices = [[1,3,5]]
            #gain_matrix = np.eye(self.SAdim_a)[np.newaxis,:]*self.gain_setting 
        else:
            raise Exception('dim & dof requested not yet supported')
        
        
        self.input_indices = torch.as_tensor(input_indices, dtype = torch.int32) # (1,F*) features to be selected (across all adims!)
        ''' 
        currently the indices should represent the following 
        action = gain_vector * d (=vector element product, not dot product)
        with d:
            d = |v| *v_ego/|v_ego| x omega
            and omega = (r x v)/|r|^2
        
        ''' 
        
        self.gain_matrix = torch.diag((self.network.weight.new(min(self.SAdim_a,3))\
                                       .zero_() + 1.)*self.gain_setting)\
            .unsqueeze(dim = 0) # (1,F*,A)
        # this ensures its on the same device
        
        #self.gain_matrix = torch.as_tensor(gain_matrix, dtype = torch.float32) # (1,F*,A)
        
        ##
        self._value_out = None # CHANGE TO MATRIX BASED ON N DIM
        return 
    
    def init_alternative(env_config: dict, input_indices: list = None, gain_matrix: list = []):
        '''
        CHECK PREVIOUS VERSIONS, INTENT/LOGIC IS FUNDAMENTALL DIFFERENT FOR THIS ONE
        
        also see the forward function over these, it uses diagonal
        and multiple feature sets based on action dimensions =(1,A,F*)
        
        However I dont believe that would ever be futureproof, 
        in case some dims need more features (i.e. inconsistent) (1,A,F*)
        '''
        raise Exception('check input_indices logic, I do not follow')
        
        return 
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"].float() #(N,F)
        
        nodim = False
        if x.dim() == 1: 
            nodim = True
            x = x.unsqueeze(dim = 0) # (1,F)
            gain_matrix = self.gain_matrix
            self._value_out = self.network.weight.new(1).zero_() 
        else:
            N = int(x.shape[0])
            #gain_matrix = torch.broadcast_to(self.gain_matrix, (N,*tuple(self.gain_matrix.shape[1:]))) # (1,A,F) -> (N,A,F)
            gain_matrix = self.gain_matrix.expand(N,*tuple(self.gain_matrix.shape[1:]))
            self._value_out = self.network.weight.new(N).zero_() 
        ##
        inputs = x[:,self.input_indices]  #(N,F) -> (N,1,F*); selected features (across all adims!)
        
        #gain_matrix = gain_matrix.to(inputs.get_device())
        action_mu =  torch.bmm(inputs, gain_matrix).squeeze(dim = 1) 
        # (N,1,F*)@(N,F*,A) -> (N,1,A) -> (N,A),squeeze
        if self.SAcode_num == 34:
            action_mu = torch.cat([action_mu[:,[0]].clone().zero_(), action_mu], dim = -1)
        
        action_logstd = (action_mu.clone().zero_()-99.)
        action_out = torch.cat([action_mu, action_logstd], dim = -1) #(N,2*A)
        # notice forcing of no exploration
        if nodim:
            action_out = action_out.squeeze(dim = 0) # cutoff batch dim
            
        return action_out, []

    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        return self._value_out # (N,) 
    
    
class BenchmarkTorch_StaticGainPNnError(TorchModelV2, CustomTorchSuperModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, ray = True):
        CustomTorchSuperModel.__init__(self)
        self.network = torch.nn.Linear(1,1, bias = False) # to get device
        if not ray:
            return 
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
    
        self.num_outputs = num_outputs
        obs_size  = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs = obs_size
        ##
        self.custom_model_config = model_config['custom_model_config']
        self.SAdim_s = int(self.custom_model_config['SA']['Dimension'])
        self.SAdim_a = int(self.custom_model_config['SA']['DOF'])
        self.SAcode_num = int(str(self.SAdim_s) + str(self.SAdim_a))
        
        self.gain_dlos_setting = self.custom_model_config.get('gain_dlos',1.)
        self.gain_los_setting = self.custom_model_config.get('gain_los',-1.)
        
        ##
        if self.SAcode_num in [21,32,33,34]:
            if self.SAcode_num == 21:
                input_los_indices = [[0]]
                input_dlos_indices = [[1]]
            elif self.SAcode_num == 32:
                input_los_indices = [[0,2]]
                input_dlos_indices = [[1,3]]
            elif self.SAcode_num == 33:
                input_los_indices = [[0,2,4]]
                input_dlos_indices = [[1,3,5]]
            elif self.SAcode_num == 34:
                input_los_indices = [[0,2,4]]
                input_dlos_indices = [[1,3,5]]
            #gain_matrix = np.eye(self.SAdim_a)[np.newaxis,:]*self.gain_setting 
        else:
            raise Exception('dim & dof requested not yet supported')
        
        
        self.input_los_indices = torch.as_tensor(input_los_indices, dtype = torch.int32) # (1,F*) features to be selected (across all adims!)
        self.input_dlos_indices = torch.as_tensor(input_dlos_indices, dtype = torch.int32) # (1,F*) features to be selected (across all adims!)
        ''' 
        currently the indices should represent the following 
        action = gain_vector * d (=vector element product, not dot product)
        with d:
            d = |v| *v_ego/|v_ego| x omega
            and omega = (r x v)/|r|^2
        
        ''' 
        
        self.gain_dlos_matrix = torch.diag((self.network.weight.new(min(self.SAdim_a,3))\
                                       .zero_() + 1.)*self.gain_dlos_setting)\
            .unsqueeze(dim = 0) # (1,F*,A)
            
        self.gain_los_matrix = torch.diag((self.network.weight.new(min(self.SAdim_a,3))\
                                       .zero_() + 1.)*self.gain_los_setting)\
            .unsqueeze(dim = 0) # (1,F*,A)
            
        # this ensures its on the same device
        
        #self.gain_matrix = torch.as_tensor(gain_matrix, dtype = torch.float32) # (1,F*,A)
        
        ##
        self._value_out = None # CHANGE TO MATRIX BASED ON N DIM
        return 
    
    def init_alternative(env_config: dict, input_indices: list = None, gain_matrix: list = []):
        '''
        CHECK PREVIOUS VERSIONS, INTENT/LOGIC IS FUNDAMENTALL DIFFERENT FOR THIS ONE
        
        also see the forward function over these, it uses diagonal
        and multiple feature sets based on action dimensions =(1,A,F*)
        
        However I dont believe that would ever be futureproof, 
        in case some dims need more features (i.e. inconsistent) (1,A,F*)
        '''
        raise Exception('check input_indices logic, I do not follow')
        
        return 
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"].float() #(N,F)
        
        nodim = False
        if x.dim() == 1: 
            nodim = True
            x = x.unsqueeze(dim = 0) # (1,F)
            gain_los_matrix = self.gain_los_matrix
            gain_dlos_matrix = self.gain_dlos_matrix
            self._value_out = self.network.weight.new(1).zero_() 
        else:
            N = int(x.shape[0])
            #gain_matrix = torch.broadcast_to(self.gain_matrix, (N,*tuple(self.gain_matrix.shape[1:]))) # (1,A,F) -> (N,A,F)
            gain_los_matrix = self.gain_los_matrix.expand(N,*tuple(self.gain_los_matrix.shape[1:]))
            gain_dlos_matrix = self.gain_dlos_matrix.expand(N,*tuple(self.gain_dlos_matrix.shape[1:]))
            
            self._value_out = self.network.weight.new(N).zero_() 
        ##
        inputs_los = x[:,self.input_los_indices]  #(N,F) -> (N,1,F*); selected features (across all adims!)
        inputs_dlos = x[:,self.input_dlos_indices]  #(N,F) -> (N,1,F*); selected features (across all adims!)
        
        #gain_matrix = gain_matrix.to(inputs.get_device())
        action_mu_los =  torch.bmm(inputs_los, gain_los_matrix).squeeze(dim = 1) 
        action_mu_dlos =  torch.bmm(inputs_dlos, gain_dlos_matrix).squeeze(dim = 1) 
        action_mu = action_mu_los + action_mu_dlos
        # (N,1,F*)@(N,F*,A) -> (N,1,A) -> (N,A),squeeze
        if self.SAcode_num == 34:
            action_mu = torch.cat([action_mu[:,[0]].clone().zero_(), action_mu], dim = -1)
        
        action_logstd = (action_mu.clone().zero_()-99.)
        action_out = torch.cat([action_mu, action_logstd], dim = -1) #(N,2*A)
        # notice forcing of no exploration
        if nodim:
            action_out = action_out.squeeze(dim = 0) # cutoff batch dim
            
        return action_out, []

    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        return self._value_out # (N,) 
    
#%% DUMMY MODELS (no parameters)

class CustomTorch_DummyNullModel(TorchModelV2, CustomTorchSuperModel):
    '''
    MODEL THAT RETURNS NO COMPUTRED ACTION NOR VALUE ESTIMATE.
    It returns zero on the correct device and according to the desired dimensions.
    
    '''
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        CustomTorchSuperModel.__init__(self)
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        

        self.num_outputs = num_outputs
        obs_size  = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs = obs_size
        #self.custom_model_config = model_config['custom_model_config']
        
        self.network = torch.nn.Linear(1,1, bias = False) # to get device
        self._action_out = None
        self._value_out = None
        return 

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"].float()
        
        a_dim = self.num_outputs//2
        
        if x.dim() > 1:
            # if multiple samples return batch, otherwise return only (act_dim*2,)
            N = int(x.shape[0])

            self._dummy_out = self.network.weight.new(N, self.num_outputs)\
                .zero_()
            #self._dummy_out[:,a_dim:] = -99. 
            # center std around mean with p = 1 certainty
            self._value_out = self._dummy_out[:,0] # cutoff last dim
        else:
            self._dummy_out = self.network.weight.new(self.num_outputs)\
                .zero_() 
            #self._dummy_out[a_dim:] = -99.
            # center std around mean with p = 1 certainty
            self._value_out = self._dummy_out[[0]] # retain a single dim
        ''' 
        # TODO EXPLORATION/RANDOM ACTION HERE?
        # todo get gaussian distribution
        '''
        return self._dummy_out, [] # (N, act_dim*2)

    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        return self._value_out # (N,) 

#%% FEEDFORWARD (NON-RECURRENT) MODELS


def setup_sequential_FNN(num_inputs, units, activation_fn = 'tanh', layernorm = False):
    hi, network = num_inputs, []      
    # intermed layers
    for ho in units:
        network.append(
            SlimFC(
                in_size=hi,
                out_size=ho,
                initializer=normc_initializer(1.0), # weights ~N(0,1)
                activation_fn=activation_fn, # only lowercase! tanh, no relu
                # note that bias is set to zero in this function
            ))
        if layernorm:
            network.append(torch.nn.LayerNorm(ho))
        hi = ho
    # output layer
    out_dim = hi
    network = torch.nn.Sequential(*network)
    return network, out_dim


    
class CustomTorch_FNN2branches(TorchModelV2, CustomTorchSuperModel):
    ''' 
    SOURCE CODE:
    - fully connected network https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
        - notice that initialization std is 0.01 for output layers & biases always at 0!
        - slimFC & initializations https://github.com/ray-project/ray/blob/master/rllib/models/torch/misc.py#L256
    
    The following custom network class defines a replication of Ray RLlib's native FCnet.
    Additional notation is provided to improve tractability and insights.
    
    BEST PRACTICES: based on the source code links provided, the following 
    best practices are identified;
   	- PARALLEL FLOWS: Use seperate actor and critic setup, i.e. parallel 
           (non-shared) I/O flow, in case the agents already operate on 
           system states. The use of shared layers makes sense in case the 
           inputs are not explicitly defined as/related to the system state. 
           - An example would be an RGB image input, where the actual state 
               of the system is the location of agents within this case. 
               In such a case a shared network might extract the system 
               state first, after which parallel branches exist
   	- ACTIVATION & STABILITY: RL tasks are already hard enough in case of
           environment stability so the use of tanh activations should ensure 
           centering of flows. Additionally, one might consider some type of 
           normalization e.g. batch/layer. NO ACTIVATIONS ON OUTPUT!
   	- INITIALIZATION: FCnet/FCslim shows intermediary layers initialized 
           with N(0,1) distribution and biases set to zero. For output layers 
           the bias is zero as well and the std is set to 0.01; I think this 
           is to enforce the agent initially does not do anything too weird 
           (i.e. its even close to no input 0) and then it gradually picks 
           up behaviour as it goes
       	- NOTE; this latter point might be stabler across trials, 
               however I myself observe that it often takes 20000 steps before 
               the agent learns anything, so another type of 
               intialization might be fine.
               
    TODO;
    - EVALUATION MODE MIGHT NOT CALL .EVAL() OR TORCH.NO_GRAD; im not confident it does
        -> in TF they use this flag is_training boolean for models is spotted for 
        batch normalization 
        (https://docs.ray.io/en/latest/rllib/rllib-models.html)
    '''


    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self)

        self.num_outputs = num_outputs
        obs_size  = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs = obs_size
        self.custom_model_config = model_config['custom_model_config']
        
        
        ## shared network
        self.network_shared = lambda x: x  

        ## action network
        self.action_units = [4,4]
        
        self.network_action, hi = \
            setup_sequential_FNN(self.num_inputs,self.action_units)
        # append output layer
        self.network_action.add_module('out', SlimFC(
            in_size=hi,
            out_size=self.num_outputs,
            initializer=normc_initializer(0.01), # notice std!
            activation_fn=None, # no activation
            # note that bias is set to zero in this function
        ))
        
        ## value network
        self.value_units = [4,4] 
        
        self.network_value, hi = \
            setup_sequential_FNN(self.num_inputs,self.value_units)
        # append output layer
        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        ## finalize
        self._value_out = None
        self._shared_out = None
        #print(f'\n=== NAME: {name} ===')
        #print(self)
        return 

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"].float()
        
        self._shared_out = self.network_shared(x)
        
        action_out = self.network_action(self._shared_out)
        return action_out, [] # (N, act_dim*2) , []

    @override(TorchModelV2)
    def value_function(self):
        '''
        in ray native examples you can see that value is only 
        computed through value function to speed up inference 
        after training is over!
        see link above
        '''
        self._value_out = self.network_value(self._shared_out) # (N,1)
        if self._value_out.dim() == 2:
            self._value_out = self._value_out.squeeze(dim = 1) # (N,)
        return self._value_out

    '''
    # https://github.com/ray-project/ray/blob/master/rllib/examples/models/custom_loss_model.py
    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        
        dLoS = self._dLoS  # (N,s_dim)
        dLoS_hat = self._dLoS_hat # (N,s_dim)
        # cache these from obs during forward pass
        # NOTE ensure that forward still outputs actions i.e. 
        
        auxiliary_loss = (lambda_dot-lambda_dot_hat).pow(2.).mean(dim = 0).sum()
        
        
        ## cache individual losses for tracking
        self.policy_loss = policy_loss
        self.auxiliary_loss = auxiliary_loss
        
        ## combine and return for optimization routine
        total_loss = policy_loss + auxiliary_loss
    
        return total_loss
    
    def metrics(self):
        # TODO, HOW TO ASSSESS THESE IN TENSORBOARD? M
        # MAYBE SEE; https://github.com/ray-project/ray/blob/master/rllib/examples/custom_model_loss_and_metrics.py
        # OR ACCESS/USE A CALLBACK
        return {
            "policy_loss": self.policy_loss,
            "auxiliary_loss": self.auxiliary_loss,
        }
    '''

class CustomTorch_FNN3branches(TorchModelV2, CustomTorchSuperModel):
    ''' 
    see CustomTorch_FNN2branches for main documentation
    '''


    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self)

        self.num_outputs = num_outputs
        obs_size  = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs = obs_size
        self.custom_model_config = model_config['custom_model_config']
        
        
        ## shared network
        self.network_shared = lambda x: x  

        ## action network
        self.action_mu_units = [4,4]
        self.action_logstd_units = [4,4]
        
        self.network_action_mu, hi = \
            setup_sequential_FNN(self.num_inputs,self.action_mu_units)
        # append output layer
        self.network_action_mu.add_module('out', SlimFC(
            in_size=hi,
            out_size=self.num_outputs // 2,
            initializer=normc_initializer(0.01), # notice std!
            activation_fn=None, # no activation
            # note that bias is set to zero in this function
        ))
        
        self.network_action_logstd, hi = \
            setup_sequential_FNN(self.num_inputs,self.action_logstd_units)
        # append output layer
        self.network_action_logstd.add_module('out', SlimFC(
            in_size=hi,
            out_size=self.num_outputs // 2,
            initializer=normc_initializer(0.01), # notice std!
            activation_fn=None, # no activation
            # note that bias is set to zero in this function
        ))
        
        ## value network
        self.value_units = [4,4] 
        
        self.network_value, hi = \
            setup_sequential_FNN(self.num_inputs,self.value_units)
        # append output layer
        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        ## finalize
        self._value_out = None
        self._shared_out = None
        #print(f'\n=== NAME: {name} ===')
        #print(self)
        return 

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        #TRAINING_FLAG = input_dict["is_training"]
        x = input_dict["obs_flat"].float()
        
        self._shared_out = self.network_shared(x)
        
        action_mu = self.network_action_mu(self._shared_out)
        action_logstd = self.network_action_logstd(self._shared_out)
        
        action_out = torch.cat([action_mu, action_logstd], dim = -1)
        return action_out, [] # (N, act_dim*2) , []

    @override(TorchModelV2)
    def value_function(self):
        '''
        in ray native examples you can see that value is only 
        computed through value function to speed up inference 
        after training is over!
        see link above
        '''
        self._value_out = self.network_value(self._shared_out) # (N,1)
        if self._value_out.dim() == 2:
            self._value_out = self._value_out.squeeze(dim = 1) # (N,)
        return self._value_out

#%% CUSTOM AL MODEL
class CustomTorch_FNN2b_PNpinn(TorchModelV2, CustomTorchSuperModel):
    ''' 
    see CustomTorch_FNN2branches for main documentation
    '''

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self)

        self.num_outputs = num_outputs
        self.AL_DIM = 0 # 2
        self.VALUE_DIM = 0 # 5
        
        self.num_outputs_AL = self.num_outputs + self.AL_DIM # adjust outputs based on auxiliary objectives
        
        obs_size  = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs = obs_size
        self.num_inputs -= self.VALUE_DIM
        self.num_inputs -= self.AL_DIM # adjust inputs based on provided targets
        self.custom_model_config = model_config['custom_model_config']
        
        
        ## shared network
        self.network_shared = lambda x: x  

        ## action network
        self.action_units = [4,4]
        
        self.network_action, hi = \
            setup_sequential_FNN(self.num_inputs,self.action_units)
        # append output layer
        self.network_action.add_module('out', SlimFC(
            in_size=hi,
            out_size=self.num_outputs_AL,
            initializer=normc_initializer(0.01), # notice std!
            activation_fn=None, # no activation
            # note that bias is set to zero in this function
        ))
        self.TANH_SCALER = torch.nn.Tanh()
        self.TANH_SCALER2 = torch.nn.Tanh()
        ## value network
        self.value_units = [4,4] 
        
        self.network_value, hi = \
            setup_sequential_FNN(self.num_inputs+self.VALUE_DIM,
                                 self.value_units, 
                                 #activation_fn='relu'
                                 )
        # append output layer
        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        ## finalize
        self._value_out = None
        self._shared_out = None
        self._dLoS_target = None
        #print(f'\n=== NAME: {name} ===')
        #print(self)
        return 

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        #TRAINING_FLAG = input_dict["is_training"]
        x = input_dict["obs_flat"].float()
        ## prep inputs
        # IMPORTANT be sure to adjust inputs based on provided targets
        self._dLoS_target = x[...,-self.AL_DIM:]
        
        #self._t = x[:,[-3]] # (N,1) time label, used for autodifferentation
        x = x[...,:-self.AL_DIM] # cutoff targets
        
        ##
        x_v = x.clone()
        x = x[..., :-self.VALUE_DIM].clone()
        
        ##
        self._shared_out = self.network_shared(x)
        
        action_out = self.network_action(self._shared_out)
        
        self._value_out = self.network_value(x_v)
        ## prep outputs
        # IMPORTANT be sure to adjust outputs based on provided targets
        '''
        TODO START WITH PREDICTING DLOS DIRECTLY, THEN IF THAT WORKS
        TRY PREDICTING LOS AND USE THE AUTO DIFFERENTIATON TO GET DLOSS
        
        TO CONSIDER;
        - maybe lambdadot should be split from the computation graph wrt to the action
            - (A) because the graph becomes cluttered wrt to responsibility i.e.
                dLoS will be optimized wrt to policy and AL
            - (B) std is now ambigious, it relates to the product of dLoS*gain,
                instead of either
            - MAYBE BOTH STD & MU SHOULD BE WRT TO (DETACHED) DLOS to maintain
                consistency in scale
                - problem with that is that the sign switch of dLoS can cause
                    very weird exploration
        
        '''
        self._dLoS_pred = action_out[...,-self.AL_DIM:] # (N,2), dLoS predictions
        self._action_out = action_out[...,:self.num_outputs] # (N,adim), mu & std for the gain
        ## ^DOING IT THIS WAY IS ESSSENTIALLY SAYING; FIND DLOS AND WE EXPECT TO BE USED, BUT DONT DO SO EXPLICITLY
        
        #self._gains = action_out[:,:(self.num_outputs//2)]
        #action_mu = self._gains*self._dLoS_pred # (N,2), gains*dLoS
        #'''
        action_mu, action_std = torch.chunk(self._action_out, chunks =2, dim = -1)
        #action_mu = self.TANH_SCALER(action_mu)*torch.pi
        #action_std = self.TANH_SCALER2(action_std)
        self.action_out = torch.cat([action_mu, action_std], dim = -1)
        #'''
        return self._action_out, [] # (N, act_dim*2) , []

    @override(TorchModelV2)
    def value_function(self):
        '''
        in ray native examples you can see that value is only 
        computed through value function to speed up inference 
        after training is over!
        see link above
        '''
        
        #self._value_out = self.network_value(self._shared_out) # (N,1)
        if self._value_out.dim() == 2:
            self._value_out = self._value_out.squeeze(dim = 1) # (N,)
        
        return self._value_out


    # https://github.com/ray-project/ray/blob/master/rllib/examples/models/custom_loss_model.py
    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """
        NOTE THIS IS THE ADJUSTMENT OF THE POLICY LOSS, NOT THAT
        OF THE CRITIC!
        
        Calculates a custom loss on top of the given policy_loss(es).

        Args:
           policy_loss (List[TensorType]): The list of already calculated
               policy losses (as many as there are optimizers).
               -> thus one, example output:
                   [tensor(2.7049, device='cuda:0', grad_fn=<AddBackward0>)]
           loss_inputs: Struct of np.ndarrays holding the
               entire train batch. NOT TRUE, STRUCT OF TENSORS!
               -> SampleBatch(128: ['obs', 'new_obs', 'actions', 'rewards', 'terminateds', 
                                    'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 
                                    't', 'vf_preds', 'action_dist_inputs', 'action_logp',
                                    'values_bootstrapped', 'advantages', 'value_targets'])
               - for notes on these inputs see Loss_inputs Ray AI discussion (loss inputs) see onenote
               
        Returns:
           List[TensorType]: The altered list of policy losses. In case the
               custom loss should have its own optimizer, make sure the
               returned list is one larger than the incoming policy_loss list.
               In case you simply want to mix in the custom loss into the
               already calculated policy losses, return a list of altered
               policy losses (as done in this example below).
               
        TODO;
        - consider the case where a seperate optimizer makes sense?
            - e.g. if you want different LR rate for the different loss components?
        """
                
        #assert False, f'actions: {actions[[0],...]} action_dist: {action_dist_inputs[[0],...]}'
        #assert False, f'Ad: {advantages.shape} action_dist: {action_dist_inputs.shape} action_logp: {action_logp.shape}'
        #assert False, f'INPUTS {loss_inputs}'
        
        self.DO_SIL, self.SIL_coef = False, 1. #1/100.
        self.DO_AL, self.AL_coef = True, 1.
        
        ## Policy loss
        '''
        TO CONSIDER;  (also see notes above)
        the scale of policy loss with regard to additional ones (SIL/AL)
        is important, but also the scale is important with regard to
        value or entropy if they are connected by shared optimizer/parametrs
        -> I dont know if this connection exists
        '''
        
        ## Self-imitation loss
        SIL_loss = self._action_out.new(1,).zero_()
        if self.DO_SIL:
            '''
            Self-imitation learning (form of behavioural cloning)
            in this case we only fixate on the policy mean, which we
            push more towards the
            
            TO CONSIDER;
            - this does not stand inline with PPO's idea, which ensures 
            the policy does diverge too much from the old. Hence you might
            get unstable behaviour if you run this.
            - if you do use it consider the scale of the loss; the MSE
            structure and the advantage func can inflate this quite a bit
            where you might lose PPO entirely
            
            IDEA; maybe give trajectory_view of last_info, and do SIL for missed
                or correct interception
            '''
            actions = loss_inputs.get('actions') # sampled actions from the distribution
            advantages = loss_inputs.get('advantages') # advantages for the sampled actions
            advantages_clip = advantages.clamp(min = 0.)
            #action_dist_inputs = loss_inputs.get('action_dist_inputs')
            #action_logp = loss_inputs.get('action_logp')
            action_mu, action_logstd = torch.chunk(self._action_out, chunks = 2, dim = 1)
            SIL_loss = ((action_mu-actions).pow(2.).sum(dim = 1)*advantages_clip).sum() # SIL for mean only ~= BC
            SIL_loss *= self.SIL_coef
            

        ## Auxiliary Loss
        auxiliary_loss = self._action_out.new(1,).zero_()
        if self.DO_AL:
            '''
            auxiliary learning to learn specific variables
            from observations (e.g. acceleration/dLOS) which are
            considered important for robust solution.
            
            - Generally this should not interfere with PPO, as long
            as you think about the scale of the losses
            
            TODO;
            - try to access from (intermediary) model layers how
                good the learnt properties (e.g. dLoS) are
            - consider if we also want to learn velocity/acceleration norm
                of the adversary
            '''
            #dLoS_target = self._dLoS_target  # (N,s_dim)
            #dLoS_pred = self._dLoS_pred # (N,s_dim)
            # cache these from obs during forward pass
            # NOTE ensure that forward still outputs actions i.e. 
            
            ##
            #self._dLoS_pred = torch.autograd.grad(self._LoS, self._t) 
            ##
            #auxiliary_loss_LoS = (self._LoS_pred-self._LoS_target).pow(2.).mean(dim = 0).sum()
            auxiliary_loss_dLoS = (self._dLoS_pred-self._dLoS_target).pow(2.).mean(dim = 0).sum()
            
            auxiliary_loss = auxiliary_loss_dLoS # + auxiliary_loss_LoS 
            auxiliary_loss *= self.AL_coef
        
        ## cache individual losses for tracking
        self.policy_loss = np.mean([loss.item() for loss in policy_loss])
        self.auxiliary_loss = auxiliary_loss.item()
        self.SIL_loss = SIL_loss.item()
        
        ## combine and return for optimization routine
        total_loss = [Ploss + auxiliary_loss + SIL_loss for Ploss in policy_loss]
    
        return total_loss
    
    def metrics(self):
        # TODO, HOW TO ASSSESS THESE IN TENSORBOARD? M
        # MAYBE SEE; https://github.com/ray-project/ray/blob/master/rllib/examples/custom_model_loss_and_metrics.py
        # OR ACCESS/USE A CALLBACK
        return {
            "policy_loss": self.policy_loss,
            "auxiliary_loss": self.auxiliary_loss,
            "SIL_loss":self.SIL_loss,
            #auxiliary_loss_LoS": self.auxiliary_loss_LoS,
            #auxiliary_loss_dLoS": self.auxiliary_loss_dLoS,
        }

    
#%% RECURRENT MODELS

class CustomTorchLSTMModel_MAIN(RecurrentNetwork, CustomTorchSuperModel):
    '''
    Custom Pytorch RNN model according to Ray RLlib descriptions. 
    based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
    alternatively the source code for 
        (A) LSTM PPO default at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        (B) Recurrent super class, also at same link
        (C) add_time_dimension at https://github.com/ray-project/ray/blob/master/rllib/policy/rnn_sequencing.py
        (D) action dist: https://github.com/ray-project/ray/blob/49b72eec162d0f0275ea1f1213616e1647adf360/rllib/models/torch/\torch_action_dist.py#L179
        (E) algo.evaluate(): algo.evaluate() https://github.com/ray-project/ray/blob/master/rllib/algorithms/algorithm.py#L908
        (F) inference github exampl; https://github.com/ray-project/ray/blob/master/rllib/examples/inference_and_serving/policy_inference_after_training_with_lstm.py
            - this seems improper use! notice recurrent use of state, no reset; tbh this confirms for me that
                training is also wrong! REMARK ON TRAINING; no this is not neccesarily true, maybe batch setup is correct
                and thus the T dim as well for RNN
    NOTE ON RAY's RNN:
        Two functions are important 
            - forward: this function prepares inputs to the model by appending time-dim & caching previous
                observations (& even actions). This is where the max_seq_len is implemented using (C).
                - works on obs_flat which is (N,T*s) shape
            - forward_rnn: this is your conventional forwardpass with the model layers now.
                - works on preprocessed 'inputs' which is (N,T,s) shape. DO NOT ACCESS obs_flat in this func
            
    NOTE ON ESTIMATION OF RNN:
        Looking at source code (A), you see that prev observations (& even actions) are cached. 
        The following sentence is important in comment in (B)
            'max_seq_len != input_dict.max_seq_len != seq_lens.max()'
            (-> 'setting  != input shape != environment steps related)
        they mention padding is conducted in case to short and I assume they also cutoff
        in case to long. Note that 'obs_flat' contains all previous inputs which
        is reshaped into the correct (1,T,s) before estimation using (C).
        
        TODO: Currently I do not know whether estimation focussed only on T=-1,
        or on all of them, the reason for this concern is because in
        the forward passes we cannot imply rnn_out = rnn_out[:,[-1],:]
        -> this likely has to do with the value function and the overall system 
        tracking howmany timesteps entered and left (raising error on inconsistency)
    
    NOTE ON INFERENCE: following transformations occur,
    (N=1 expected for compute_single_action())
        - obs: (N,s) -> (N,1,s) # time minor
        - rnn state: (N,h_dim) -> (1,N,h_dim) # always time major!
            -> out: action (a,) & next rnn state [(h_dim,),(h_dim,)].
    Idea is that RNN was properly trained wrt to time (i.e. never saw T> max_seq_length),
    thus continuous re-use of state is possible. However, reusing the state continuously 
    (>max_seq_length) instead of resetting seems erroneous as now T -> inf theoretically?
    
    EVALUATE function of an algorithm (not policy) does not seem to cache the required amount
    of observations for proper inference of a RNN model. As observed through
    printing of RNN model's inputs during .evaluate() call. I have not
    confirmed this with the evaluate source code at E.


    MODEL OUTPUT
    for continuous action spaces the action model's output dim is 2*action_space_size
    i.e. (N,out_dim), this used by action dim as torch.chunk(out, 2, axis = 1)
        i.e. so first means, then std's (Source code at (D))

    SOME FORUMS POSTS RELATED TO RNN;
        - https://discuss.ray.io/t/yet-another-question-on-rnn-sequencing/4549/6
        - for inference example (& issue) see link F above
        
    TODO
    - for inference example (& issue) see link F above
    - potentially ray rllibs rnn_tokenizer for attention might be the key
        to ensure consistent T dimensions
        - https://github.com/ray-project/ray/blob/master/rllib/examples/custom_recurrent_rnn_tokenizer.py
    - Trajectory view API https://docs.ray.io/en/latest/rllib/rllib-sample-collection.html <- quit clear as well as next!!
        - examples;
            - https://discuss.ray.io/t/trajectory-view-api-example/8873
        - SOURCE CODE https://github.com/ray-project/ray/blob/master/rllib/examples/models/trajectory_view_utilizing_models.py
        - SAMPLE BATCH; 
            - https://docs.ray.io/en/latest/_modules/ray/rllib/policy/sample_batch.html
            - https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.policy.sample_batch.SampleBatch.html 
            - STACK HELP https://stackoverflow.com/questions/65772439/rllib-offlinedata-preparation-for-sac
            - DOCS FOR BUILDER; https://github.com/ray-project/ray/blob/master/rllib/evaluation/sample_batch_builder.py
            - RAY DOCS; https://docs.ray.io/en/latest/rllib/rllib-sample-collection.html
        - might be replaced by 'RLmodules' or something?
    - define seq lens? https://discuss.ray.io/t/custom-lstm-model-how-to-define-the-seq-len/5657/3
    '''
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
        # alternatively the source code for super at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self,RNN_type=True)
        assert not self.time_major, 'time major seems to be true!'

        #self.MODE_INFERENCE = False # deprecated
        
        obs_size = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs  = obs_size
        self.num_outputs = num_outputs # inferred as well
        self.config_model = model_config['custom_model_config']
        
        self.lstm_units = 8 #self.config_model['lstm_units']
        self.lstm_timesteps = 5#self.config_model['lstm_timesteps']
        self.timesteps_total = self.lstm_timesteps 
            
        self.network_shared_lstm = torch.nn.LSTM(self.num_inputs, self.lstm_units, batch_first=True)
        self.network_action = torch.nn.Linear(self.lstm_units, self.num_outputs)
        self.network_value = torch.nn.Linear(self.lstm_units, 1)

        self._shared_out = None
        
        '''
        # FOLLOWING IS KNOWN AS TRAJECTORY API
        # https://docs.ray.io/en/latest/rllib/rllib-sample-collection.html
        # STACK HELP https://stackoverflow.com/questions/65772439/rllib-offlinedata-preparation-for-sac
        - DOCS FOR BUILDER; https://github.com/ray-project/ray/blob/master/rllib/evaluation/sample_batch_builder.py
        # THIS MIGHT BE THE SOLUTION TO THE AMOUNT OF PREVIOUS SAMPLES
        if model_config["lstm_use_prev_action"]:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
                SampleBatch.ACTIONS, space=self.action_space, shift=-1
            )
        if model_config["lstm_use_prev_reward"]:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
                SampleBatch.REWARDS, shift=-1
            )
        # IS THERE A PREV_OBSERVATIONS? see samplebatch documentation
        
        # PREV ACTIONS & REWARDS SEEMS DEPRECATED
        - might be replaced by 'RLmodules' or something?
        '''
        
        # https://github.com/ray-project/ray/blob/master/rllib/examples/models/trajectory_view_utilizing_models.py
        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(self.timesteps_total-1), space=obs_space
        )
        
        #self.view_requirements = {
        #    SampleBatch.OBS: ViewRequirement(space=obs_space),
        #}

        '''
        self.view_requirements["prev_n_rewards"] = ViewRequirement(
            data_col="rewards", shift="-{}:-1".format(self.num_frames)
        )
        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space,
            
        )
        self.view_requirements["state_out_0"] = \
            ViewRequirement(
                    space=space,
                    used_for_training=False,
                    used_for_compute_actions=True,
                    batch_repeat_value=1)
        #'''
        return


    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        #TRAINING_FLAG = input_dict["is_training"]
        """Feeds `inputs` (B x T x ..) through the model

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = inputs
        '''
        if self.MODE_INFERENCE:
            # WHY IS THIS NEEDED, FOR REASON ENABLE THE PRINT SIZE STATEMENTS BELOW!
            x = x.view(1,self.lstm_timesteps,self.obs_size)
        ''' 
        '''
        print(list(x.shape))
        print(list(torch.unsqueeze(state[0], 0).shape))
        print(list(torch.unsqueeze(state[0], 0).shape))
        print(f'{list(seq_lens)} = SQ')
        print('--')
        #'''
        
        print(x.size())
        self._shared_out, [h, c] = self.network_shared_lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        ) # note that h- & c-states are always time-major, regardless of setting
        #self._shared_out = self._shared_out[:,-1,:] # select last timestep
        action_out = self.network_action(self._shared_out)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)] # (B,T,2*act_dim), [(B,lstm_out)]*2 

    @override(ModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        return torch.reshape(self.network_value(self._shared_out), [-1]) # (B*T,)

    @override(ModelV2)    
    def get_initial_state(self):
        # initialize internal rnn states on same device
        linear = self.network_action
        h = [
            linear.weight.new(1, self.lstm_units).zero_().squeeze(0),
            linear.weight.new(1, self.lstm_units).zero_().squeeze(0),
        ]
        return h


#%% LSTM - OLD
class CustomTorchLSTMModel(TorchModelV2, CustomTorchSuperModel): #RecurrentNetwork TorchModelV2
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
        # alternatively the source code for super at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self,RNN_type=True)
        assert not self.time_major, 'time major seems to be true!'

        self.modelIdentifier = model_config['custom_model']
        #self.model_config = model_config['custom_model_config']
        #self.agent_type = self.model_config['aid']
        self.gain_bias0 = 0#3. if 'p0' in self.modelIdentifier else -1. # COMMENTED OUT IN FORWARD AS WELL
        
        #self.MODE_INFERENCE = False # deprecated
        self.DO_BC = False # behavioural cloning
        if self.DO_BC:
            self.BC_idx = list(range(1,(num_outputs//2)*2,2))
            gain = 1. if 'p0' in self.modelIdentifier else -1.
            self.BClayer = BCTorchLayer(num_outputs//2, gain = gain)

        self.DO_BS = False
        self.VALUE_DIM = 9  # 10
        self.VALUE_DIM += self.DO_BS*4*bool(self.VALUE_DIM)
        
        self.AL_DIM  = 0 #2
        
        obs_size = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs  = obs_size - self.AL_DIM
        self.num_inputs_act = self.num_inputs - self.VALUE_DIM
        self.num_inputs_value = self.VALUE_DIM # self.num_inputs - self.num_inputs_act
        
        assert self.num_inputs_act > 0, f'{self.num_inputs}-{self.VALUE_DIM} = {self.num_inputs_act}'
        
        self.num_outputs = num_outputs #num_outputs # inferred as well
        #self.num_outputs_act = 2+1+1 #self.num_outputs + 1
        #self.config_model = model_config['custom_model_config']
        
        self.lstm_units = 16 #self.config_model['lstm_units']
        self.lstm_timesteps = 30#self.config_model['lstm_timesteps']
        self.timesteps_total = self.lstm_timesteps 
        self.timesteps_stride = 1    
        
        #self.network_shared_rnn = torch.nn.LSTM(self.num_inputs, self.lstm_units, batch_first=True)
        #self.network_action = torch.nn.Linear(self.lstm_units, self.num_outputs)
        #self.network_value = torch.nn.Linear(self.lstm_units, 1)
        
        self.action_units_rnn = 16 #self.config_model['lstm_units']
        self.network_action_rnn = torch.nn.LSTM(self.num_inputs_act, self.action_units_rnn, batch_first=True)
        #self.network_action = torch.nn.Linear(self.action_units_rnn, self.num_outputs)
        '''
        self.network_action = SlimFC(
                in_size=self.action_units_rnn,
                out_size=self.num_outputs,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
        )
        '''
        self.action_units = [8]*2 #[4]*2
        self.network_action, hi = \
            setup_sequential_FNN(self.action_units_rnn,
                                 self.action_units, 
                                 activation_fn='relu',
                                 )

        # append output layer
        self.network_action.add_module('out', SlimFC(
                in_size=hi,
                out_size=self.num_outputs,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        #'''
        self.value_units_rnn = 16 #self.config_model['lstm_units']
        self.network_value_rnn = torch.nn.LSTM(self.num_inputs_value, self.value_units_rnn, batch_first=True)
        #self.network_value = torch.nn.Linear(self.value_units_rnn, 1)
        #'''
        
        #'''
        self.value_units = [8]*2 #[4]*2
        self.network_value, hi = \
            setup_sequential_FNN(self.value_units_rnn,
                                 self.value_units, 
                                 activation_fn='relu',
                                 )

        # append output layer
        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        '''
        self.network_value = SlimFC(
                in_size=self.value_units_rnn,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
        )
        
        #'''
        self._shared_out = None
        
        #self.view_requirements = {
        #    SampleBatch.OBS: ViewRequirement(space=obs_space),
        #}
        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(self.timesteps_total-1), space=obs_space
        ) # -1 since it adds +1
        self.timesteps_idx = (range(-1,-self.timesteps_total, 
                                    -self.timesteps_stride))[::-1] #[-N,...,-1]
        
        '''
        self.tanh1 = torch.nn.Tanh()
        self.tanh2 = torch.nn.Tanh()
        '''
        '''
        self.gain_units = [8]*3 #[4]*2
        self.network_gain, hi = \
            setup_sequential_FNN(self.value_units_rnn,
                                 self.gain_units, 
                                 activation_fn='relu',
                                 )
        
        #'''
        return 
    
    #def forward(self, input_dict, state: List[TensorType],
    #            seq_lens: TensorType) -> (TensorType, List[TensorType]):
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens, add_time_dim = False):
        #TRAINING_FLAG = input_dict["is_training"]
        # Add the time dim to observations.
        '''
        if isinstance(input_dict, SampleBatch):
            is_training = input_dict.is_training
        else:
            is_training = input_dict["is_training"]
        '''
        '''
        
        
        B = len(seq_lens)
        observations = input_dict["prev_n_obs"]

        shape = observations.size()
        T = shape[0] // B
        observations = torch.reshape(observations,
                                  [-1, T] + list(shape[1:]))
        
        ## sample output
        (RolloutWorker pid=17996) SEQ:tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        (RolloutWorker pid=17996)         1, 1, 1, 1, 1, 1, 1, 1])
        (RolloutWorker pid=17996) shape:torch.Size([32, 6, 4])
        (RolloutWorker pid=17996) T:1
        (RolloutWorker pid=17996) xshape;
        (RolloutWorker pid=17996) torch.Size([32, 1, 6, 4])
        '''
        if add_time_dim:
            # USED IN BC
            x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
        else:
            # observations (B,T,F)
            observations = input_dict["prev_n_obs"]  # (N,T,F)
            x = observations[...,self.timesteps_idx,:]
        
        x_act = x[...,:self.num_inputs_act]
        self.x_v = x[...,self.num_inputs_act:]
        if self.AL_DIM:
            self._dLoS_target = self.x_v[..., -1, -self.AL_DIM:]
            self.x_v = self.x_v[...,:-self.AL_DIM]
            
        #print(f'xshape; {x.size()}')
        #print(f'x; {x}')
        '''
        self._shared_out, [h, c] = self.network_shared_rnn(
            x, #[torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        ) # note that h- & c-states are always time-major, regardless of setting
        self._shared_out = self._shared_out[:,-1,:] # select last timestep
        action_out = self.network_action(self._shared_out)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)] # (B,T,2*act_dim), [(B,lstm_out)]*2 
        '''
        act, _ = self.network_action_rnn(x_act) # (N,T,lstm_out)
        act = act[...,-1,:] 
        
        self._action_out = self.network_action(act) # (N, a_out_dim)
        
        '''
        self._action_gain = self._action_out[..., [0]]
        #self._action_out = self._action_out[..., 1:]
        #self._dLoS_pred, self._action_logstd = torch.chunk(self._action_out, 2, dim = 1) # 2*(N,a_dim)
        
        self._dLoS_pred = self._action_out[..., 2:]
        self._action_logstd = self._action_out[..., [1]]
        
        self._action_logstd = self._action_logstd*(self._action_logstd.new(self._action_logstd.size(0),2).zero_() + 1.)
        
        self._action_mu = self._dLoS_pred*self._action_gain # (N,a_dim)*(N,1)
        self._action_out = torch.cat((self._action_mu, 
                                      self._action_logstd), dim = 1) # (N,2*a_dim)
        '''
        #self._gain_mu = self._action_out[..., [0]]
        #self._gain_logstd = self._action_out[..., [1]]
        
        # GET INTERMEDIARY OUTPUT
        '''
        self._dLoS_pred = act[..., -2:] # (N, lstm_out) -> (N,2)
        
        # CONTINUE ON FOR GAIN
        self._action_out = self.network_action(act) # (N, a_out_dim)
        self._gain_mu = self._action_out[..., [0]]
        self._gain_logstd = self._action_out[..., [1]]

        ## BOUND SIGNALS
        #self._gain_mu = self.tanh1(self._gain_mu)*10.
        #self._dLoS_pred = self.tanh2(self._dLoS_pred)*1. # TODO
        
        #self._dLoS_pred = self._dLoS_target
        
        self._action_out =  torch.cat((self._gain_mu + self.gain_bias0,
                                       self._gain_logstd,
                                       self._dLoS_pred,
                                       self._dLoS_target,
                                       ), dim = -1)
        #'''
        
        #print('===')
        #print(self._action_out.size())
        #print(self._value_out.size())
        return self._action_out, []
    
    
    @override(ModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        #return torch.reshape(self.network_value(self._shared_out), [-1]) # (B*T,)
        
        value, _ = self.network_value_rnn(self.x_v)
        value = value[...,-1,:]
        self._value_out = self.network_value(value).squeeze(dim = -1) # (N,)
        
        return self._value_out # (B*T,)

    '''
    @override(ModelV2)    
    def get_initial_state(self):
        # initialize internal rnn states on same device
        linear = self.network_action._model[0]
        h = [
            linear.weight.new(1, self.lstm_units).zero_().squeeze(0),
            linear.weight.new(1, self.lstm_units).zero_().squeeze(0),
        ]
        return h
    '''
    
    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """
        NOTE THIS IS THE ADJUSTMENT OF THE POLICY LOSS, NOT THAT
        OF THE CRITIC!
        
        Calculates a custom loss on top of the given policy_loss(es).

        Args:
           policy_loss (List[TensorType]): The list of already calculated
               policy losses (as many as there are optimizers).
               -> thus one, example output:
                   [tensor(2.7049, device='cuda:0', grad_fn=<AddBackward0>)]
           loss_inputs: Struct of np.ndarrays holding the
               entire train batch. NOT TRUE, STRUCT OF TENSORS!
               -> SampleBatch(128: ['obs', 'new_obs', 'actions', 'rewards', 'terminateds', 
                                    'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 
                                    't', 'vf_preds', 'action_dist_inputs', 'action_logp',
                                    'values_bootstrapped', 'advantages', 'value_targets'])
               - for notes on these inputs see Loss_inputs Ray AI discussion (loss inputs) see onenote
               
        Returns:
           List[TensorType]: The altered list of policy losses. In case the
               custom loss should have its own optimizer, make sure the
               returned list is one larger than the incoming policy_loss list.
               In case you simply want to mix in the custom loss into the
               already calculated policy losses, return a list of altered
               policy losses (as done in this example below).
               
        TODO;
        - consider the case where a seperate optimizer makes sense?
            - e.g. if you want different LR rate for the different loss components?
        """
                
        #assert False, f'actions: {actions[[0],...]} action_dist: {action_dist_inputs[[0],...]}'
        #assert False, f'Ad: {advantages.shape} action_dist: {action_dist_inputs.shape} action_logp: {action_logp.shape}'
        #assert False, f'INPUTS {loss_inputs}'
        
        self.BC_coef = 10. # Use_BC is set in __init__
        self.DO_SIL, self.SIL_coef = False, 1. #1/100.
        self.DO_AL, self.AL_coef = self.AL_DIM, 1.
        
        self.BC_loss, self.SIL_loss, self.auxiliary_loss = 0.,0.,0.
        BC_loss, SIL_loss, auxiliary_loss = 0., 0., 0.
        ## Policy loss
        '''
        TO CONSIDER;  (also see notes above)
        the scale of policy loss with regard to additional ones (SIL/AL)
        is important, but also the scale is important with regard to
        value or entropy if they are connected by shared optimizer/parametrs
        -> I dont know if this connection exists
        '''
        ## behavioural cloning loss
        if self.DO_BC:
            BC_loss = self.BClayer.weights.abs().sum() * self.BC_coef
            self.BC_loss = BC_loss.item()
        
        ## Self-imitation loss
        if self.DO_SIL:
            '''
            Self-imitation learning (form of behavioural cloning)
            in this case we only fixate on the policy mean, which we
            push more towards the
            
            TO CONSIDER;
            - this does not stand inline with PPO's idea, which ensures 
            the policy does diverge too much from the old. Hence you might
            get unstable behaviour if you run this.
            - if you do use it consider the scale of the loss; the MSE
            structure and the advantage func can inflate this quite a bit
            where you might lose PPO entirely
            
            IDEA; maybe give trajectory_view of last_info, and do SIL for missed
                or correct interception
            '''
            actions = loss_inputs.get('actions') # sampled actions from the distribution
            advantages = loss_inputs.get('advantages') # advantages for the sampled actions
            advantages_clip = advantages.clamp(min = 0.)
            #action_dist_inputs = loss_inputs.get('action_dist_inputs')
            #action_logp = loss_inputs.get('action_logp')
            action_mu, action_logstd = torch.chunk(self._action_out, chunks = 2, dim = 1)
            SIL_loss = ((action_mu-actions).pow(2.).sum(dim = 1)*advantages_clip).mean() # SIL for mean only ~= BC
            SIL_loss *= self.SIL_coef
            self.SIL_loss = SIL_loss.item()

        ## Auxiliary Loss
        if self.DO_AL:
            '''
            auxiliary learning to learn specific variables
            from observations (e.g. acceleration/dLOS) which are
            considered important for robust solution.
            
            - Generally this should not interfere with PPO, as long
            as you think about the scale of the losses
            
            TODO;
            - try to access from (intermediary) model layers how
                good the learnt properties (e.g. dLoS) are
            - consider if we also want to learn velocity/acceleration norm
                of the adversary
            '''
            #dLoS_target = self._dLoS_target  # (N,s_dim)
            #dLoS_pred = self._dLoS_pred # (N,s_dim)
            # cache these from obs during forward pass
            # NOTE ensure that forward still outputs actions i.e. 
            
            ##
            #self._dLoS_pred = torch.autograd.grad(self._LoS, self._t) 
            ##
            #auxiliary_loss_LoS = (self._LoS_pred-self._LoS_target).pow(2.).mean(dim = 0).mean()
            auxiliary_loss_dLoS = (self._dLoS_pred-self._dLoS_target).pow(2.).mean(dim = 0).mean()
            
            auxiliary_loss = auxiliary_loss_dLoS # + auxiliary_loss_LoS 
            auxiliary_loss *= self.AL_coef
            self.auxiliary_loss = auxiliary_loss.item()
        
        ## cache individual losses for tracking
        self.policy_loss = np.mean([loss.item() for loss in policy_loss])
        
        ## combine and return for optimization routine
        total_loss = [Ploss + auxiliary_loss \
                      + SIL_loss + BC_loss for Ploss in policy_loss]
    
        return total_loss
    
    def metrics(self):
        # TODO, HOW TO ASSSESS THESE IN TENSORBOARD? M
        # MAYBE SEE; https://github.com/ray-project/ray/blob/master/rllib/examples/custom_model_loss_and_metrics.py
        # OR ACCESS/USE A CALLBACK
        return {
            "policy_loss": self.policy_loss,
            
            "BC_loss":self.BC_loss,
            "auxiliary_loss": self.auxiliary_loss,
            "SIL_loss":self.SIL_loss,
            #auxiliary_loss_LoS": self.auxiliary_loss_LoS,
            #auxiliary_loss_dLoS": self.auxiliary_loss_dLoS,
        }


#%% LTC    

'''
import seaborn as sns
#https://ncps.readthedocs.io/en/latest/examples/torch_first_steps.html
wiring = ncpswirings.AutoNCP(16+out_features, out_features)  # 16 units, 1 motor neuron
ltc_model = ncpstorch.CfC(in_features, wiring, batch_first=True)

# we add hidden+out_features as otherwise out is counted with hidden

sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()
#'''

class BCTorchLayer(torch.nn.Module):
    
    def __init__(self, size, gain = 1.):
        super(BCTorchLayer, self).__init__()
        
        weights = torch.eye(size)*gain
        self.weights = torch.nn.Parameter(weights)
        
        return 
        
    def forward(self, x):
        y = torch.mm(x,self.weights) #(N,3)@(3,3) = (N,3)
        return y 
    

class CustomTorchLTCModel(TorchModelV2, CustomTorchSuperModel): #RecurrentNetwork TorchModelV2
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
        # alternatively the source code for super at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self,RNN_type=True)
        assert not self.time_major, 'time major seems to be true!'

        self.modelIdentifier = model_config['custom_model']
        #self.model_config = model_config['custom_model_config']
        #self.agent_type = self.model_config['aid']
        self.agent_sign = 1. if 'p0' in self.modelIdentifier else -1.
        #self.gain_bias0 = 0#3. if 'p0' in self.modelIdentifier else -1. # COMMENTED OUT IN FORWARD AS WELL
        
        #self.MODE_INFERENCE = False # deprecated
        self.DO_BC = False # behavioural cloning
        if self.DO_BC:
            self.BC_idx = list(range(1,(num_outputs//2)*2,2))
            gain = 1. if 'p0' in self.modelIdentifier else -1.
            self.BClayer = BCTorchLayer(num_outputs//2, gain = gain)

        self.DO_BS = False
        self.VALUE_DIM = 9  # 10
        self.VALUE_DIM += self.DO_BS*4*bool(self.VALUE_DIM)

        self.AL_DIM = 0
        
        obs_size = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs  = obs_size - self.AL_DIM
        self.num_inputs_act = self.num_inputs - self.VALUE_DIM
        self.num_inputs_value = self.VALUE_DIM
        
        assert self.num_inputs_act > 0, f'{self.num_inputs}-{self.VALUE_DIM} = {self.num_inputs_act}'
        
        self.num_outputs = num_outputs # inferred as well
        #self.config_model = model_config['custom_model_config']
        
        #self.lstm_units = 32 #self.config_model['lstm_units']
        self.lstm_timesteps = 30#self.config_model['lstm_timesteps']
        self.timesteps_total = self.lstm_timesteps 
        self.timesteps_stride = 1   
        
        #self.network_shared_rnn = torch.nn.LSTM(self.num_inputs, self.lstm_units, batch_first=True)
        #self.network_action = torch.nn.Linear(self.lstm_units, self.num_outputs)
        #self.network_value = torch.nn.Linear(self.lstm_units, 1)
        
        self.action_units_rnn = 32 #self.config_model['lstm_units']
        self.wiring_action = ncpswirings.AutoNCP(self.action_units_rnn+self.num_outputs, self.num_outputs)  # 16 units, 1 motor neuron
        self.network_action_rnn = ncpstorch.CfC(self.num_inputs_act, self.wiring_action, batch_first=True) # input (N,T,F)
        #self.network_action = torch.nn.Linear(self.action_units_rnn, self.num_outputs)
        '''
        self.network_action = SlimFC(
                in_size=self.action_units_rnn,
                out_size=self.num_outputs,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
        )
        '''
        #'''
        self.value_units_rnn = 16 #self.config_model['lstm_units']
        '''
        self.wiring_value = ncpswirings.AutoNCP(self.value_units_rnn+1, 1)  # 16 units, 1 motor neuron
        self.network_value_rnn = ncpstorch.CfC(self.num_inputs_value, self.wiring_value, batch_first=True) # input (N,T,F)
        '''
        self.network_value_rnn = torch.nn.LSTM(self.num_inputs_value, self.value_units_rnn)
        #'''
        ''' 
        
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incomming syanpses has each motor neuron
        
        ''' 
        #self.network_value = torch.nn.Linear(self.value_units_rnn, 1)
        #'''
        self.value_units = [8]*2 #[4]*2
        
        self.network_value, hi = \
            setup_sequential_FNN(self.value_units_rnn,
                                 self.value_units, 
                                 activation_fn='relu',
                                 )
        # append output layer
        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        #'''
        self._shared_out = None
        
        #self.view_requirements = {
        #    SampleBatch.OBS: ViewRequirement(space=obs_space),
        #}
        
        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(self.timesteps_total-1), space=obs_space
        ) # -1 since it adds +1
        self.timesteps_idx = list(range(-1,-self.timesteps_total, 
                                    -self.timesteps_stride))[::-1] #[-N,...,-1]
        return 
    
    #def forward(self, input_dict, state: List[TensorType],
    #            seq_lens: TensorType) -> (TensorType, List[TensorType]):
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens, add_time_dim = False):
        #TRAINING_FLAG = input_dict["is_training"]
        # Add the time dim to observations.
        '''
        if isinstance(input_dict, SampleBatch):
            is_training = input_dict.is_training
        else:
            is_training = input_dict["is_training"]
        '''
        
        '''
        B = len(seq_lens)
        observations = input_dict["prev_n_obs"]

        shape = observations.size()
        T = shape[0] // B
        observations = torch.reshape(observations,
                                  [-1, T] + list(shape[1:]))
        
        ## sample output
        (RolloutWorker pid=17996) SEQ:tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        (RolloutWorker pid=17996)         1, 1, 1, 1, 1, 1, 1, 1])
        (RolloutWorker pid=17996) shape:torch.Size([32, 6, 4])
        (RolloutWorker pid=17996) T:1
        (RolloutWorker pid=17996) xshape;
        (RolloutWorker pid=17996) torch.Size([32, 1, 6, 4])
        '''

        
        if add_time_dim:
            # USED IN BC
            x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
        else:
            # observations (B,T,F)
            #observations = input_dict["prev_n_obs"]
            try:
                observations = input_dict["prev_n_obs"]  # (N,T,F)
                x = observations[...,self.timesteps_idx,:]
            except KeyError:
                #torch.save(input_dict, r"C:\Users\Reinier Vos\Documents\PEG1\TESTBC\inputdict.pt")
                #assert False, input_dict
                x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
        
        x_act = x[...,:self.num_inputs_act]
        self.x_v = x[...,self.num_inputs_act:]
        if self.AL_DIM:
            self._dLoS_target = self.x_v[..., -1, -self.AL_DIM:]
            self.x_v = self.x_v[...,:-self.AL_DIM]
        #self._dLoS_target = 
        
        #print(f'xshape; {x.size()}')
        #print(f'x; {x}')
        '''
        self._shared_out, [h, c] = self.network_shared_rnn(
            x, #[torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        ) # note that h- & c-states are always time-major, regardless of setting
        self._shared_out = self._shared_out[:,-1,:] # select last timestep
        action_out = self.network_action(self._shared_out)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)] # (B,T,2*act_dim), [(B,lstm_out)]*2 
        '''
        act, _ = self.network_action_rnn(x_act)
        act = act[...,-1,:] 
        #self._action_out = self.network_action(act) # (N, actdim)
        self._action_out = act
        
        #self._dLoS_pred = 
        
        if self.DO_BC:
            act_mu, act_logstd = torch.chunk(act, chunks =2, dim = 1)
            act_mu_BC = self.BClayer(x_act[:,-1,self.BC_idx])
            act_mu = act_mu+act_mu_BC    
            self._action_out = torch.cat([act_mu, act_logstd], dim = 1)
        
        #print('===')
        #print(self._action_out.size())
        #print(self._value_out.size())
        return self._action_out, []
    
    
    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        #return torch.reshape(self.network_value(self._shared_out), [-1]) # (B*T,)
        
        value, _ = self.network_value_rnn(self.x_v)
        value = value[...,-1,:]
        self._value_out = self.network_value(value).squeeze(dim = -1) # (N,)
        #self._value_out = value.squeeze(dim = -1)
        
        return self._value_out # (B,)

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """
        NOTE THIS IS THE ADJUSTMENT OF THE POLICY LOSS, NOT THAT
        OF THE CRITIC!
        
        Calculates a custom loss on top of the given policy_loss(es).
    
        Args:
           policy_loss (List[TensorType]): The list of already calculated
               policy losses (as many as there are optimizers).
               -> thus one, example output:
                   [tensor(2.7049, device='cuda:0', grad_fn=<AddBackward0>)]
           loss_inputs: Struct of np.ndarrays holding the
               entire train batch. NOT TRUE, STRUCT OF TENSORS!
               -> SampleBatch(128: ['obs', 'new_obs', 'actions', 'rewards', 'terminateds', 
                                    'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 
                                    't', 'vf_preds', 'action_dist_inputs', 'action_logp',
                                    'values_bootstrapped', 'advantages', 'value_targets'])
               - for notes on these inputs see Loss_inputs Ray AI discussion (loss inputs) see onenote
               
        Returns:
           List[TensorType]: The altered list of policy losses. In case the
               custom loss should have its own optimizer, make sure the
               returned list is one larger than the incoming policy_loss list.
               In case you simply want to mix in the custom loss into the
               already calculated policy losses, return a list of altered
               policy losses (as done in this example below).
               
        TODO;
        - consider the case where a seperate optimizer makes sense?
            - e.g. if you want different LR rate for the different loss components?
        """
                
        #assert False, f'actions: {actions[[0],...]} action_dist: {action_dist_inputs[[0],...]}'
        #assert False, f'Ad: {advantages.shape} action_dist: {action_dist_inputs.shape} action_logp: {action_logp.shape}'
        #assert False, f'INPUTS {loss_inputs}'
        
        self.BC_coef = 10. # Use_BC is set in __init__
        self.DO_SIL, self.SIL_coef = False, 1. #1/100.
        self.DO_AL, self.AL_coef = self.AL_DIM, 1.
        
        self.BC_loss, self.SIL_loss, self.auxiliary_loss = 0.,0.,0.
        BC_loss, SIL_loss, auxiliary_loss = 0., 0., 0.
        ## Policy loss
        '''
        TO CONSIDER;  (also see notes above)
        the scale of policy loss with regard to additional ones (SIL/AL)
        is important, but also the scale is important with regard to
        value or entropy if they are connected by shared optimizer/parametrs
        -> I dont know if this connection exists
        '''
        ## behavioural cloning loss
        if self.DO_BC:
            BC_loss = self.BClayer.weights.abs().sum() * self.BC_coef
            self.BC_loss = BC_loss.item()
        
        ## Self-imitation loss
        if self.DO_SIL:
            '''
            Self-imitation learning (form of behavioural cloning)
            in this case we only fixate on the policy mean, which we
            push more towards the
            
            TO CONSIDER;
            - this does not stand inline with PPO's idea, which ensures 
            the policy does diverge too much from the old. Hence you might
            get unstable behaviour if you run this.
            - if you do use it consider the scale of the loss; the MSE
            structure and the advantage func can inflate this quite a bit
            where you might lose PPO entirely
            
            IDEA; maybe give trajectory_view of last_info, and do SIL for missed
                or correct interception
            '''
            actions = loss_inputs.get('actions') # sampled actions from the distribution
            advantages = loss_inputs.get('advantages') # advantages for the sampled actions
            advantages_clip = advantages.clamp(min = 0.)
            #action_dist_inputs = loss_inputs.get('action_dist_inputs')
            #action_logp = loss_inputs.get('action_logp')
            action_mu, action_logstd = torch.chunk(self._action_out, chunks = 2, dim = 1)
            SIL_loss = ((action_mu-actions).pow(2.).sum(dim = 1)*advantages_clip).mean() # SIL for mean only ~= BC
            SIL_loss *= self.SIL_coef
            self.SIL_loss = SIL_loss.item()
    
        ## Auxiliary Loss
        if self.DO_AL:
            '''
            auxiliary learning to learn specific variables
            from observations (e.g. acceleration/dLOS) which are
            considered important for robust solution.
            
            - Generally this should not interfere with PPO, as long
            as you think about the scale of the losses
            
            TODO;
            - try to access from (intermediary) model layers how
                good the learnt properties (e.g. dLoS) are
            - consider if we also want to learn velocity/acceleration norm
                of the adversary
            '''
            #dLoS_target = self._dLoS_target  # (N,s_dim)
            #dLoS_pred = self._dLoS_pred # (N,s_dim)
            # cache these from obs during forward pass
            # NOTE ensure that forward still outputs actions i.e. 
            
            ##
            #self._dLoS_pred = torch.autograd.grad(self._LoS, self._t) 
            ##
            #auxiliary_loss_LoS = (self._LoS_pred-self._LoS_target).pow(2.).mean(dim = 0).mean()
            auxiliary_loss_dLoS = (self._dLoS_pred-self._dLoS_target).pow(2.).mean(dim = 0).mean()
            
            auxiliary_loss = auxiliary_loss_dLoS # + auxiliary_loss_LoS 
            auxiliary_loss *= self.AL_coef
            self.auxiliary_loss = auxiliary_loss.item()
        
        ## cache individual losses for tracking
        self.policy_loss = np.mean([loss.item() for loss in policy_loss])
        
        ## combine and return for optimization routine
        total_loss = [Ploss + auxiliary_loss \
                      + SIL_loss + BC_loss for Ploss in policy_loss]
    
        return total_loss
    
    def metrics(self):
        # TODO, HOW TO ASSSESS THESE IN TENSORBOARD? M
        # MAYBE SEE; https://github.com/ray-project/ray/blob/master/rllib/examples/custom_model_loss_and_metrics.py
        # OR ACCESS/USE A CALLBACK
        return {
            "policy_loss": self.policy_loss,
            
            "BC_loss":self.BC_loss,
            "auxiliary_loss": self.auxiliary_loss,
            "SIL_loss":self.SIL_loss,
            #auxiliary_loss_LoS": self.auxiliary_loss_LoS,
            #auxiliary_loss_dLoS": self.auxiliary_loss_dLoS,
        }
        ''' 
        @override(ModelV2)    
        def get_initial_state(self):
            # initialize internal rnn states on same device
            linear = self.network_action._model[0]
            h = [
                linear.weight.new(1, self.lstm_units).zero_().squeeze(0),
                linear.weight.new(1, self.lstm_units).zero_().squeeze(0),
            ]
            return h
        ''' 
'''
class TorchRNNModel(RecurrentNetwork, torch.nn.Module):
    '#''
    NOTE RNN models: following transformations occur,
    (N=1 expected for compute_single_action())
        - obs: (N,s) -> (N,1,s) # time minor
        - rnn state: (N,h_dim) -> (1,N,h_dim) # always time major!
            -> out: action (a,) & next rnn state [(h_dim,),(h_dim,)].
    Idea is that RNN was properly trained wrt to time (i.e. never saw T> max_seq_length),
    thus continuous re-use of state is possible. However, reusing the state continuously 
    (>max_seq_length) instead of resetting seems erroneous as now T -> inf theoretically?
    '#''
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=4,
        lstm_state_size=8,
    ):
        torch.nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = torch.nn.Linear(self.obs_size, self.fc_size)
        self.lstm = torch.nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = torch.nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = torch.nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = torch.nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
'''
#%%

class CustomTorchDoubleFNNTuneModel(TorchModelV2, CustomTorchSuperModel): #RecurrentNetwork TorchModelV2
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
        # alternatively the source code for super at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self,RNN_type=True)
        assert not self.time_major, 'time major seems to be true!'

        self.modelIdentifier = model_config['custom_model']
        #self.model_config = model_config['custom_model_config']
        #self.agent_type = self.model_config['aid']
        self.agent_sign = 1. if 'p0' in self.modelIdentifier else -1.
        

        self.DO_BS = True
        self.VALUE_DIM = 7  # 10
        self.VALUE_DIM += self.DO_BS*2*bool(self.VALUE_DIM)

        self.AL_DIM = 0
        
        obs_size = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs  = obs_size - self.AL_DIM
        self.num_inputs_act = self.num_inputs - self.VALUE_DIM
        self.num_inputs_value = self.num_inputs_act + self.num_outputs #self.VALUE_DIM
        #self.num_inputs_value = int(self.num_inputs_act)
        
        assert self.num_inputs_act > 0, f'{self.num_inputs}-{self.VALUE_DIM} = {self.num_inputs_act}'
        
        self.num_outputs = num_outputs # inferred as well
        #self.config_model = model_config['custom_model_config']
        
        #################################
    
        self.network_action = None
        #self.action_units = [64]*3 #[4]*2
        self.action_units = [16]*9 #[4]*2
        
        self.network_action1, hi = \
            setup_sequential_FNN(self.num_inputs_act,
                                 self.action_units, 
                                 activation_fn='relu', # relu, tanh
                                 #layernorm = True,
                                 )

        self.network_action1.add_module('out', SlimFC(
                in_size=hi,
                out_size=self.num_outputs-2,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        self.network_action2, hi = \
            setup_sequential_FNN(self.num_inputs_act+self.num_outputs-2,
                                 self.action_units, 
                                 activation_fn='relu', # relu, tanh
                                 #layernorm = True,
                                 )

        self.network_action2.add_module('out', SlimFC(
                in_size=hi,
                out_size=2,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        #################################
    
        self.network_value = None
        #'''
        self.value_units = [8]*3 #[4]*2
        
        self.network_value, hi = \
            setup_sequential_FNN(self.num_inputs_value,
                                 self.value_units, 
                                 activation_fn='tanh', #
                                 )

        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        #'''

        return 
    
    #def forward(self, input_dict, state: List[TensorType],
    #            seq_lens: TensorType) -> (TensorType, List[TensorType]):
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens, add_time_dim = False):

        
        if add_time_dim:
            # USED IN BC
            x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
        else:
            # observations (B,T,F)
            #observations = input_dict["prev_n_obs"]

            x = input_dict["obs"][:,:self.num_inputs] # (N,F) -> (N,1,T)
        
        x_act = x[...,:self.num_inputs_act]
        #self.x_v = x.clone()
        self.x_v = x_act.clone()

        self._action_out1 = self.network_action1(x_act)
        self._action_out2 = self.network_action2(torch.cat([x_act, self._action_out1], dim = -1))

        self._action_out = torch.cat([self._action_out1[...,:3],
                                      self._action_out2[...,[0]],
                                      self._action_out1[...,3:],
                                      self._action_out2[...,[1]]],
                                     dim = -1)
        
    
        self.x_v = torch.cat([self.x_v, self._action_out], axis = -1) # append action
        
        '''
        self._action_out[...,1:(self.num_outputs // 2)] *= self.agent_sign
        #'''         
        return self._action_out, []
    
    
    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        #return torch.reshape(self.network_value(self._shared_out), [-1]) # (B*T,)
        
        value = self.network_value(self.x_v).squeeze(dim = -1)
        self._value_out = value
        return self._value_out # (B,)
    

class CustomTorchFNNTuneModel(TorchModelV2, CustomTorchSuperModel): #RecurrentNetwork TorchModelV2
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
        # alternatively the source code for super at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self,RNN_type=True)
        assert not self.time_major, 'time major seems to be true!'

        self.modelIdentifier = model_config['custom_model']
        #self.model_config = model_config['custom_model_config']
        #self.agent_type = self.model_config['aid']
        self.agent_sign = 1. if 'p0' in self.modelIdentifier else -1.
        

        self.DO_BS = True
        self.VALUE_DIM = 9  # 10
        self.VALUE_DIM += self.DO_BS*2*bool(self.VALUE_DIM)

        self.AL_DIM = 0
        
        obs_size = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs  = obs_size - self.AL_DIM
        self.num_inputs_act = self.num_inputs - self.VALUE_DIM
        self.num_inputs_value = self.num_inputs #+ self.num_outputs #self.VALUE_DIM
        #self.num_inputs_value = int(self.num_inputs_act)
        
        assert self.num_inputs_act > 0, f'{self.num_inputs}-{self.VALUE_DIM} = {self.num_inputs_act}'
        
        self.num_outputs = num_outputs # inferred as well
        #self.config_model = model_config['custom_model_config']
        
        #################################
    
        self.network_action = None
        #self.action_units = [64]*3 #[4]*2
        self.action_units = [32]*3 #[4]*2
        
        self.network_action, hi = \
            setup_sequential_FNN(self.num_inputs_act + self.num_outputs//2,
                                 self.action_units, 
                                 activation_fn='tanh', # relu, tanh
                                 #layernorm = True,
                                 )

        self.network_action.add_module('out', SlimFC(
                in_size=hi,
                out_size=self.num_outputs,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        #################################
    
        self.network_value = None
        #'''
        self.value_units = [32]*3 #[4]*2
        
        self.network_value, hi = \
            setup_sequential_FNN(self.num_inputs_value,
                                 self.value_units, 
                                 activation_fn='tanh', #
                                 )

        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        #'''

        #'''
        self.view_requirements['prev_actions'] = ViewRequirement(data_col='actions', 
                     space=action_space, shift=-1, 
                     #index=None, batch_repeat_value=1, 
                     used_for_compute_actions=True, 
                     used_for_training=True, 
                     #shift_arr=np.array([0])
                     )
        
        #'''
        self.input_dict = None
        return 
    
    #def forward(self, input_dict, state: List[TensorType],
    #            seq_lens: TensorType) -> (TensorType, List[TensorType]):
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens, add_time_dim = False):

        self.input_dict = input_dict
        if add_time_dim:
            # USED IN BC
            x = self.input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
        else:
            # observations (B,T,F)
            #observations = input_dict["prev_n_obs"]

            x = self.input_dict["obs"][:,:self.num_inputs] # (N,F) -> (N,1,T)
            
        prev_actions = self.input_dict["prev_actions"]
        
        x_act = x[...,:self.num_inputs_act]
        x_act = torch.cat([x_act,prev_actions], dim =-1)
        
        self.x_v = x.clone()
        #self.x_v = x_act.clone()

        self._action_out = self.network_action(x_act)

        #self.x_v = torch.cat([self.x_v,self._action_out], dim =1)
        '''
        self._action_out[...,1:(self.num_outputs // 2)] *= self.agent_sign
        #'''         
        return self._action_out, []
    
    
    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        #return torch.reshape(self.network_value(self._shared_out), [-1]) # (B*T,)
        '''
        try:
            action_sample = self.input_dict["prev_actions"]
            #assert False, action_sample.shape
        except KeyError:
            action_sample = self.x_v.new(self.x_v.shape[0],self.num_outputs//2).zero_()
        self.x_v = torch.cat([self.x_v,action_sample], dim =1)
        '''
        value = self.network_value(self.x_v).squeeze(dim = -1)
        self._value_out = value
        return self._value_out # (B,)
    
    
class CustomTorchRNNTuneModel(TorchModelV2, CustomTorchSuperModel): #RecurrentNetwork TorchModelV2
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
        # alternatively the source code for super at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self,RNN_type=True)
        assert not self.time_major, 'time major seems to be true!'

        self.modelIdentifier = model_config['custom_model']
        #self.model_config = model_config['custom_model_config']
        #self.agent_type = self.model_config['aid']
        self.agent_sign = 1. if 'p0' in self.modelIdentifier else -1.
        

        self.DO_BS = True
        self.VALUE_DIM = 9  # 10
        self.VALUE_DIM += self.DO_BS*2*bool(self.VALUE_DIM)

        self.AL_DIM = 0
        self.num_outputs = num_outputs#-2 # inferred as well
        
        obs_size = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs  = obs_size - self.AL_DIM
        self.num_inputs_act = self.num_inputs - self.VALUE_DIM #+ self.num_outputs//2 #+ 4
        self.num_inputs_value = self.num_inputs  #self.VALUE_DIM
        
        assert self.num_inputs_act > 0, f'{self.num_inputs}-{self.VALUE_DIM} = {self.num_inputs_act}'
        
        #self.config_model = model_config['custom_model_config']
        
        #self.lstm_units = 32 #self.config_model['lstm_units']
        self.lstm_timesteps = 10#10 #self.config_model['lstm_timesteps']
        self.timesteps_total = self.lstm_timesteps 
        self.timesteps_stride = 1#1   
        
        #self.network_shared_rnn = torch.nn.LSTM(self.num_inputs, self.lstm_units, batch_first=True)
        #self.network_action = torch.nn.Linear(self.lstm_units, self.num_outputs)
        #self.network_value = torch.nn.Linear(self.lstm_units, 1)
        
        self.num_inputs_rnn = int(self.num_inputs_act) +4
        self.network_action_pre = None
        self.network_action = None 
        '''
        pre_in, pre_out = self.num_inputs_act, 16
        pre_units = [16]*2 + [pre_out]
        self.network_action_pre, hi = \
            setup_sequential_FNN(pre_in, 
                                 pre_units,
                                 activation_fn='relu',
                                 )

        #self.network_action_pre.add_module('pre_out', SlimFC(
        #        in_size=hi,
        #        out_size=pre_out,
        #        #initializer=normc_initializer(0.01), # notice std!
        #        activation_fn='tanh', # no activation
        #        # note that bias is set to zero in this function
        #    ))
        self.num_inputs_rnn = pre_out
        
        #'''
        self.action_units_rnn = 64 # 32 #self.config_model['lstm_units']
        '''

        self.wiring_action = ncpswirings.AutoNCP(self.action_units_rnn+self.num_outputs, self.num_outputs)  # 16 units, 1 motor neuron
        self.network_action_rnn = ncpstorch.CfC(self.num_inputs_rnn, self.wiring_action, batch_first=True) # input (N,T,F)
        #self.network_action = torch.nn.Linear(self.action_units_rnn, self.num_outputs)
        '''
        
        ##
        self.network_action_rnn = torch.nn.LSTM(self.num_inputs_rnn, 
                                                self.action_units_rnn)

        self.network_action = SlimFC(
                in_size=self.action_units_rnn,
                out_size=self.num_outputs,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
        )
        
        #action_units = [32]*2
        #self.network_action, hi = \
        #    setup_sequential_FNN(self.action_units_rnn,
        #                         action_units, 
        #                         activation_fn='relu',
        #                         )

        #self.network_action.add_module('out', SlimFC(
        #        in_size=hi,
        #        out_size=self.num_outputs,
        #        initializer=normc_initializer(0.01), # notice std!
        #        activation_fn=None, # no activation
        #        # note that bias is set to zero in this function
        #    ))
        #'''
        
        #################################
    
        self.network_value = None
        self.value_units_rnn = 32 # 32 #self.config_model['lstm_units']
        '''
        self.wiring_value = ncpswirings.AutoNCP(self.value_units_rnn+1, 1)  # 16 units, 1 motor neuron
        self.network_value_rnn = ncpstorch.CfC(self.num_inputs_value, self.wiring_value, batch_first=True) # input (N,T,F)
        ''' 
        self.network_value_rnn = torch.nn.LSTM(self.num_inputs_value, self.value_units_rnn)
        
        self.value_units = [16]*2 # 16 #[4]*2
        
        self.network_value, hi = \
            setup_sequential_FNN(self.value_units_rnn,#+self.num_outputs,
                                 self.value_units, 
                                 activation_fn='relu',
                                 )

        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        #'''
        
        
        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(self.timesteps_total-1), space=obs_space
        ) # -1 since it adds +1
        self.timesteps_idx = list(range(-1,-self.timesteps_total, 
                                    -self.timesteps_stride))[::-1] #[-N,...,-1]
        
        #'''
        self.view_requirements['prev_actions'] = ViewRequirement(
                     data_col='actions', 
                     space=action_space, 
                     shift="-{}:-1".format(self.timesteps_total), # notice shift 
                     #index=None, batch_repeat_value=1, 
                     used_for_compute_actions=True, 
                     used_for_training=True, 
                     #shift_arr=np.array([0])
                     )
        
        #'''
        
        return 
    
    #def forward(self, input_dict, state: List[TensorType],
    #            seq_lens: TensorType) -> (TensorType, List[TensorType]):
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens, add_time_dim = False):

        
        if add_time_dim:
            # USED IN BC
            x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
        else:
            # observations (B,T,F)
            #observations = input_dict["prev_n_obs"]
            try:
                observations = input_dict["prev_n_obs"]  # (N,T,F)
                x = observations[...,self.timesteps_idx,:]
                
                
            except KeyError:
                #torch.save(input_dict, r"C:\Users\Reinier Vos\Documents\PEG1\TESTBC\inputdict.pt")
                #assert False, input_dict
                x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
                
            #try:
            prev_actions = input_dict["prev_actions"][...,self.timesteps_idx,:]  # (N,T,F)
            #except KeyError:
                
        x_act = x[...,:self.num_inputs_act]
        
        #assert False, f'{x_act.shape} == {prev_actions.shape}'
        x_act = torch.cat([x_act, prev_actions], dim = -1)
        #'''
        
        
        #'''
        self.x_v = x.clone()
        '''
        if self.AL_DIM:
            self._dLoS_target = self.x_v[..., -1, -self.AL_DIM:]
            self.x_v = self.x_v[...,:-self.AL_DIM]
        #'''
        
        if not isinstance(self.network_action_pre, type(None)):   
            x_act = self.network_action_pre(x_act) 
        #'''
        act, _ = self.network_action_rnn(x_act)
        act = act[...,-1,:] 
        #self._action_out = self.network_action(act) # (N, actdim)
        
        if not isinstance(self.network_action, type(None)):    
            act = self.network_action(act) 
            
        '''
        act_mu, act_logstd = act[..., :3], act[..., 3:] 
        
        #assert False, f'{x_act.shape}\n{x_act[...,-1,:]}\n\n\n==={act_mu}'  
        
        zeros = act_mu.new(len(act_mu),1).zero_()
        act_mu = torch.cat([zeros.clone(), act_mu], dim =1)
        act_logstd = torch.cat([(zeros+1.)*-99., act_logstd], dim =1)
        
        act = torch.cat([act_mu, act_logstd], dim = 1)
        #'''
        
        self._action_out = act        
        
        #self._action_out_detach = self._action_out.detach()
        #self.x_v = torch.cat([self.x_v, _action_out_detach], dim = -1)
        
        '''
        act1, _ = self.network_action1(x_act)
        act1 = act1[...,-1,:] 
        act1 = self.network_action1_out(act1) # (N, actdim)
        
        act2, _ = self.network_action2(x_act)
        act2 = act2[...,-1,:] 
        act2 = self.network_action2_out(act2) # (N, actdim)
        
        self._action_out = torch.cat([act1, act2], dim = 1)
        #'''
        return self._action_out, []
    
    
    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        #return torch.reshape(self.network_value(self._shared_out), [-1]) # (B*T,)
        
        value, _ = self.network_value_rnn(self.x_v)
        value = value[...,-1,:]
        
        #value = torch.cat([value, self._action_out_detach],dim = -1)
        
        if not isinstance(self.network_value, type(None)):    
            self._value_out = self.network_value(value).squeeze(dim = -1) # (N,)
        else:
            self._value_out = value.squeeze(dim = -1)
        
        return self._value_out # (B,)

class CustomTorchRNNTuneModel_EVADER(TorchModelV2, CustomTorchSuperModel): #RecurrentNetwork TorchModelV2
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # based on https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
        # alternatively the source code for super at https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CustomTorchSuperModel.__init__(self,RNN_type=True)
        assert not self.time_major, 'time major seems to be true!'

        self.modelIdentifier = model_config['custom_model']
        #self.model_config = model_config['custom_model_config']
        #self.agent_type = self.model_config['aid']
        self.agent_sign = 1. if 'p0' in self.modelIdentifier else -1.
        

        self.DO_BS = True
        self.VALUE_DIM = 9  # 10
        self.VALUE_DIM += self.DO_BS*2*bool(self.VALUE_DIM)

        self.AL_DIM = 0
        self.num_outputs = num_outputs#-2 # inferred as well
        
        obs_size = get_preprocessor(obs_space)(obs_space).size  # infer shape
        self.num_inputs  = obs_size - self.AL_DIM
        self.num_inputs_act = self.num_inputs - self.VALUE_DIM #+ self.num_outputs//2 #+ 4
        self.num_inputs_value = self.num_inputs  #self.VALUE_DIM
        
        assert self.num_inputs_act > 0, f'{self.num_inputs}-{self.VALUE_DIM} = {self.num_inputs_act}'
        
        #self.config_model = model_config['custom_model_config']
        
        #self.lstm_units = 32 #self.config_model['lstm_units']
        self.lstm_timesteps = 10#10 #self.config_model['lstm_timesteps']
        self.timesteps_total = self.lstm_timesteps 
        self.timesteps_stride = 1#1   
        
        #self.network_shared_rnn = torch.nn.LSTM(self.num_inputs, self.lstm_units, batch_first=True)
        #self.network_action = torch.nn.Linear(self.lstm_units, self.num_outputs)
        #self.network_value = torch.nn.Linear(self.lstm_units, 1)
        
        self.num_inputs_rnn = int(self.num_inputs_act) +4
        self.network_action_pre = None
        self.network_action = None 
        '''
        pre_in, pre_out = self.num_inputs_act, 16
        pre_units = [16]*2 + [pre_out]
        self.network_action_pre, hi = \
            setup_sequential_FNN(pre_in, 
                                 pre_units,
                                 activation_fn='relu',
                                 )

        #self.network_action_pre.add_module('pre_out', SlimFC(
        #        in_size=hi,
        #        out_size=pre_out,
        #        #initializer=normc_initializer(0.01), # notice std!
        #        activation_fn='tanh', # no activation
        #        # note that bias is set to zero in this function
        #    ))
        self.num_inputs_rnn = pre_out
        
        #'''
        self.action_units_rnn = 64 #self.config_model['lstm_units']
        '''

        self.wiring_action = ncpswirings.AutoNCP(self.action_units_rnn+self.num_outputs, self.num_outputs)  # 16 units, 1 motor neuron
        self.network_action_rnn = ncpstorch.CfC(self.num_inputs_rnn, self.wiring_action, batch_first=True) # input (N,T,F)
        #self.network_action = torch.nn.Linear(self.action_units_rnn, self.num_outputs)
        '''
        
        ##
        self.network_action_rnn = torch.nn.LSTM(self.num_inputs_rnn, 
                                                self.action_units_rnn)

        self.network_action = SlimFC(
                in_size=self.action_units_rnn,
                out_size=self.num_outputs,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
        )
        
        #action_units = [32]*2
        #self.network_action, hi = \
        #    setup_sequential_FNN(self.action_units_rnn,
        #                         action_units, 
        #                         activation_fn='relu',
        #                         )

        #self.network_action.add_module('out', SlimFC(
        #        in_size=hi,
        #        out_size=self.num_outputs,
        #        initializer=normc_initializer(0.01), # notice std!
        #        activation_fn=None, # no activation
        #        # note that bias is set to zero in this function
        #    ))
        #'''
        
        #################################
    
        self.network_value = None
        self.value_units_rnn = 32 #self.config_model['lstm_units']
        '''
        self.wiring_value = ncpswirings.AutoNCP(self.value_units_rnn+1, 1)  # 16 units, 1 motor neuron
        self.network_value_rnn = ncpstorch.CfC(self.num_inputs_value, self.wiring_value, batch_first=True) # input (N,T,F)
        ''' 
        self.network_value_rnn = torch.nn.LSTM(self.num_inputs_value, self.value_units_rnn)
        
        self.value_units = [16]*2 #[4]*2
        
        self.network_value, hi = \
            setup_sequential_FNN(self.value_units_rnn,#+self.num_outputs,
                                 self.value_units, 
                                 activation_fn='relu',
                                 )

        self.network_value.add_module('out', SlimFC(
                in_size=hi,
                out_size=1,
                initializer=normc_initializer(0.01), # notice std!
                activation_fn=None, # no activation
                # note that bias is set to zero in this function
            ))
        
        #'''
        
        
        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(self.timesteps_total-1), space=obs_space
        ) # -1 since it adds +1
        self.timesteps_idx = list(range(-1,-self.timesteps_total, 
                                    -self.timesteps_stride))[::-1] #[-N,...,-1]
        
        #'''
        self.view_requirements['prev_actions'] = ViewRequirement(
                     data_col='actions', 
                     space=action_space, 
                     shift="-{}:-1".format(self.timesteps_total), # notice shift 
                     #index=None, batch_repeat_value=1, 
                     used_for_compute_actions=True, 
                     used_for_training=True, 
                     #shift_arr=np.array([0])
                     )
        
        #'''
        
        return 
    
    #def forward(self, input_dict, state: List[TensorType],
    #            seq_lens: TensorType) -> (TensorType, List[TensorType]):
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens, add_time_dim = False):

        
        if add_time_dim:
            # USED IN BC
            x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
        else:
            # observations (B,T,F)
            #observations = input_dict["prev_n_obs"]
            try:
                observations = input_dict["prev_n_obs"]  # (N,T,F)
                x = observations[...,self.timesteps_idx,:]
                
                
            except KeyError:
                #torch.save(input_dict, r"C:\Users\Reinier Vos\Documents\PEG1\TESTBC\inputdict.pt")
                #assert False, input_dict
                x = input_dict["obs"][:,:self.num_inputs].unsqueeze(dim = 1) # (N,F) -> (N,1,T)
                
            #try:
            prev_actions = input_dict["prev_actions"][...,self.timesteps_idx,:]  # (N,T,F)
            #except KeyError:
                
        x_act = x[...,:self.num_inputs_act]
        
        #assert False, f'{x_act.shape} == {prev_actions.shape}'
        x_act = torch.cat([x_act, prev_actions], dim = -1)
        #'''
        
        
        #'''
        self.x_v = x.clone()
        '''
        if self.AL_DIM:
            self._dLoS_target = self.x_v[..., -1, -self.AL_DIM:]
            self.x_v = self.x_v[...,:-self.AL_DIM]
        #'''
        
        if not isinstance(self.network_action_pre, type(None)):   
            x_act = self.network_action_pre(x_act) 
        #'''
        act, _ = self.network_action_rnn(x_act)
        act = act[...,-1,:] 
        #self._action_out = self.network_action(act) # (N, actdim)
        
        if not isinstance(self.network_action, type(None)):    
            act = self.network_action(act) 
            
        '''
        act_mu, act_logstd = act[..., :3], act[..., 3:] 
        
        #assert False, f'{x_act.shape}\n{x_act[...,-1,:]}\n\n\n==={act_mu}'  
        
        zeros = act_mu.new(len(act_mu),1).zero_()
        act_mu = torch.cat([zeros.clone(), act_mu], dim =1)
        act_logstd = torch.cat([(zeros+1.)*-99., act_logstd], dim =1)
        
        act = torch.cat([act_mu, act_logstd], dim = 1)
        #'''
        
        self._action_out = act        
        
        #self._action_out_detach = self._action_out.detach()
        #self.x_v = torch.cat([self.x_v, _action_out_detach], dim = -1)
        
        '''
        act1, _ = self.network_action1(x_act)
        act1 = act1[...,-1,:] 
        act1 = self.network_action1_out(act1) # (N, actdim)
        
        act2, _ = self.network_action2(x_act)
        act2 = act2[...,-1,:] 
        act2 = self.network_action2_out(act2) # (N, actdim)
        
        self._action_out = torch.cat([act1, act2], dim = 1)
        #'''
        return self._action_out, []
    
    
    @override(TorchModelV2)
    def value_function(self):
        #assert self._shared_out is not None, "must call forward() first"
        #return torch.reshape(self.network_value(self._shared_out), [-1]) # (B*T,)
        
        value, _ = self.network_value_rnn(self.x_v)
        value = value[...,-1,:]
        
        #value = torch.cat([value, self._action_out_detach],dim = -1)
        
        if not isinstance(self.network_value, type(None)):    
            self._value_out = self.network_value(value).squeeze(dim = -1) # (N,)
        else:
            self._value_out = value.squeeze(dim = -1)
        
        return self._value_out # (B,)

#%% BEHAVIOURAL CLONING




#%% ACTION DISTRIBUTION 

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from typing import List, Union
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
import torch.distributions
import gymnasium as gym

'''
class MyActionDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return (3,)  # controls model output feature vector size

    def __init__(self, model):
        super(MyActionDist, self).__init__(model)

    def sample(self, model_out, **kwargs):
        mean, stddev, aux = model_out
        dist = tf.distributions.Normal(mean, stddev)
        action = dist.sample()
        return action * aux

    def logp(self, model_out, action, **kwargs):
        mean, stddev, aux = model_out
        dist = tf.distributions.Normal(mean, stddev)
        return dist.log_prob(action / aux)
'''


'''

class ModelExample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Fc = torch.nn.Linear(1,1)

    def forward(self, x):
        x = self.Fc(x)
        return x

a = torch.ones(10,1)
y = ModelExample()(a)
y = y.clone().detach()
'''

class TorchDiagGaussian_gain(TorchDistributionWrapper):
    """Wrapper class for PyTorch Normal distribution.
    SOURCED FROM 
        https://github.com/ray-project/ray/blob/a91ddbdeb98e81741beeeb5c17902cab1e771105/rllib/models/torch/torch_action_dist.py#L372
    THEY REFER TO;
    - ActionDistribution; https://github.com/ray-project/ray/blob/a91ddbdeb98e81741beeeb5c17902cab1e771105/rllib/models/action_dist.py
    - TorchDistributionWrapper; https://github.com/ray-project/ray/blob/a91ddbdeb98e81741beeeb5c17902cab1e771105/rllib/models/torch/torch_action_dist.py#L372
    
    PLAYGROUND CODE
    import torch.distributions

    a, b = torch.ones(10,1), torch.randn(10,1)
    
    dist = torch.distributions.normal.Normal(a, torch.exp(b))
    c = dist.sample()
    l = dist.log_prob(a)
    l.shape, c.shape
        
    """

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        ## convert to tensors 
        super().__init__(inputs, model)
         # split
        self.aux = self.inputs[..., 2:].clone().detach() # (N,2), detach to disattach from graph
        self.aux_pred, self.aux_true = torch.chunk(self.aux, 2, dim=1)
        
        self.inputs = self.inputs[..., :2] #(N,2)
        mean, log_std = torch.chunk(self.inputs, 2, dim=1)
        
        
        self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std)) 
        # .normal.Normal always assumes its diagonal, hence mu=(N,2) & std=(N,2) != (N,2,2)
        # the gain distribution (not the action distribution!)
        self.last_gain = None
        return 

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        ''' 
        FUNCTION CALLED IN EVALUATE (DETERMINSITC POLICY) SETTING
        '''
        gain_mean = self.dist.mean # (N,1)
        
        self.last_gain = gain_mean
        self.last_sample = gain_mean*(1e-12+self.aux_true) # (N,1)*(N,2) = (N,2)
        '''
        ## TODO EXTRA_ACTION_CODE
        self.last_sample = self.aux_true
        self.last_sample = torch.cat((self.last_sample, self.last_gain), dim = -1)
        ''' 
        return self.last_sample
    
    @override(ActionDistribution)
    def sample(self) -> TensorType:
        ''' 
        FUNCTION CALLED IN TRAINING (STOCHASTIC POLICY) SETTING
        '''
        gain_sample = self.dist.sample() # (N,1)
        
        self.last_gain = gain_sample
        self.last_sample = gain_sample*(1e-12+self.aux_true) # (N,1)*(N,2) = (N,2)
        '''
        ## TODO EXTRA_ACTION_CODE
        self.last_sample = self.aux_true
        self.last_sample = torch.cat((self.last_sample, self.last_gain), dim = -1)
        ''' 
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        ''' 
        FUNCTION CALLED IN TRAINING (STOCHASTIC POLICY) SETTING

        here we assume that the actions passed come from a samplebatch 
        with (at least) obs-action pairs, where the obs are used to 
        setup the distribution self.dist (from .__init__ call)
        
        obs (N,F) implies a gain distribution (N,1) as well as 
        dLoS predictions (N,2) and the ultimate action is actions = dLoS*gain. 
        BUT now we have the dist already defined and actions coming in,
        thus we need to identify the gains again and
        
        in other words this function re-transforms actions to the gain-space
        
        
        THIS IS NOT COMPLETELY CORRECT, BECAUSE:
        - the model can have learnt to improve its dLoS estimates
        - discrepancies/rounding might mean the dimensions dont 
            give the exact same gains
        - 
        '''
        #assert False, f'\nA = {actions.shape}\n aux:{self.aux_true.shape}'
        
        '''
        #implied_gains = actions/(1e-12+self.aux_pred) # (N,2)/(N,2) = (N,2) action implied gains
        implied_gains = actions/(1e-12+self.aux_true) # (N,2)/(N,2) = (N,2) SHOULD BE GAINS
        
        logp_a1g = self.dist.log_prob(implied_gains[...,[0]]) # first dim logp, (N,1)
        logp_a2g = self.dist.log_prob(implied_gains[...,[1]]) # second dim logp, (N,1)
        # in the best case these two are exactly similar!
        
        
        logp_out = ((logp_a1g + logp_a2g)/2.).sum(-1) # (N,) mean of dims and then sum
        #logp_out = logp_a1g.sum(-1)
        '''
        ## TODO EXTRA_ACTION_CODE
        logp_out = self.dist.log_prob(actions[...,[-1]]).sum(-1)
        '''
        logp_out_mean = logp_out.mean()
        if logp_out_mean > 200:
            assert False, f'POS - ACT = \n{actions} \n AUX ={self.aux_true}\n G= {implied_gains}'
        elif logp_out_mean < -200:
            assert False, f'NEG - ACT = \n{actions} \n AUX ={self.aux_true}\n G= {implied_gains}'
        #'''
        #assert False, f'SHAPE: {logp_out.shape},\n\n {logp_out}'
        # notice (N,) due to sum(-1) i.e. not sum() which would imply (1,)
        # this is because logp_out*reward =((N,1)*(N,1)) is done next
        #print(logp_out)
        return logp_out

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self) -> TensorType:
        assert self.last_gain is not None
        return self.dist.log_prob(self.last_gain)
    
    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return self.dist.entropy().sum(-1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.distributions.kl.kl_divergence(self.dist, other.dist).sum(-1)

    
    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        #return np.prod(action_space.shape) * 2
        return 6