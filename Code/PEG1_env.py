import numpy as np # TODO REPLACE WITH BOTTLENECK or numba?
import random
import os
import warnings
from datetime import datetime

import gymnasium as gym
from gymnasium.utils import seeding

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import torch

import scipy.stats as scs
import scipy.linalg as scl
import scipy.integrate as sci 
from scipy.spatial.transform import Rotation as R

import latexify

from ray.rllib.algorithms import algorithm # REMOVE?

warnings.filterwarnings("ignore", 
                        #module = 'scipy', 
                        category = UserWarning)

#%%


from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            

#%% UTILITY FUNCTIONS 


def save_divide_scalar(x,y):
    # for np arrays 
    # c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    div = 0.
    if (y != 0):
        div = (x/y)#*(y != 0) 
    return div

class Rotations:
    '''
    Functions return scipy rotational matrix
    instances with .apply() methods
    '''
    
    ''' 
    NOTE THAT THE TRANSFORMATIONS ARE WRONG WITH REGARD TO CONVENTION
    BUT THEY ARE CORRECT WITH REGARD TO EACHOTHER so one rotation of 'angles'
    is offset by rev(angles)*-1, such that rotation matrices are transposes/inv of eachother
    -> I think it does mean that the extraction of angles for 
        e.g. use in the pqr conversion is wrong
    
    the correct version would be (note [roll, pitch, yaw] =  [phi, theta, psi])
        IB: 'zyx' [psi, theta, phi], notice reverse order but positive
        BI: 'xyz' [-phi, -theta, -psi], notice normal order but negative
    WHY: because rotations essentially move through the consec rotmats
        and are post-multiply, NOT pre-multiply (which we assumed), so
            V^B = R(z)R(y)R(x) V^I occurs in apply, notice the order zyx
        we thought it was (pre-multiply)
            V^B = V^I R(x)R(y)R(z) 
        
    IMPORTANTLY, for this settings we observe that 
        observations and thrust (controls) are still consistent, 
        so thrust positive in body ref frame in case distance (z) positive
        in body frame. All other aspects also align, e.g. initial velocity
        
        
    TO RESOLVE: 
        - adjust the inertial2body & body2inertial funcs
        - CHECK body2inertial_align to do what we want*
        - do NOT change the pqr-euler functions
        - do NOT change the pqr-quaternions functions
        - CHECK all rotmat instances**, CRTL-F on R./rotmat/from_euler/as_euler/xyz/zyx
        - CHECK all xyz and zyx strings; these sometimes are used outside class
            in order to extract the euler angles
            - I.E. sometimes we do rotmat.as_euler(string) somewhere to get
            [phi,theta,psi] explicitly, where said rotmat comes from this class
            - TIP: it would be best if you set the string to extract the angles
              in this class as cls.attribute such that in future these also change
              if you adjust this class
            - IMPORTANT, for all the .to_euler(xyz) you need to multiply the result
              by -1  
        - CHECK initializations and integrator functions
        
    * = this step might require more thorough analysis
    ** = in general all rotmat's should originate from this class so
        adjusting this one, should be all we need to do
    '''
    
    def inertial2body(angles):
        # euler angles = [phi,theta,psi] (3,) in inertial frame
        RotMat = R.from_euler('xyz', 
                              np.array([angles[0], angles[1], angles[2]])*1.,
                              degrees = False)
        # [phi, theta, psi]
        return RotMat
        
    def body2inertial(angles):
        # euler angles = [phi, theta,psi] (3,) in inertial frame
        RotMat = R.from_euler('zyx', 
                              np.array([angles[2], angles[1], angles[0]])*-1.,
                              degrees = False)
        # [psi, theta, phi] *-1, notice reversed
        return RotMat
    
    def inertial2body_rates(angles):
        # angles are euler angles in inertial frame
        phi, theta, psi = angles
        RotMat = np.array([[  1.,             0.,       -np.sin(theta)      ],
                           [  0.,    np.cos(phi),  np.cos(theta)*np.sin(phi)],
                           [  0., -np.sin(theta),  np.cos(theta)*np.cos(phi)],
                           ])
        # use pqr = RotMat @ deuler
        return RotMat
    
    def body2inertial_rates(angles):
        # angles are euler angles in inertial frame
        phi, theta, psi = angles
        notSingular = 1.
        ''' 
        if abs(theta) >= (0.9*(0.5*np.pi)):
            notSingular = 0.
        #''' 
        sec_theta = 1/(np.cos(theta)+1e-8) # stable secant of theta
        RotMat = np.array([[  1., np.sin(phi)*np.tan(theta)*notSingular,  np.cos(phi)*np.tan(theta)*notSingular],
                           [  0.,               np.cos(phi),               -np.sin(phi)],
                           [  0., np.sin(phi)*sec_theta*notSingular,  np.cos(phi)*sec_theta*notSingular],
                           ])

        # use deuler = RotMat @ pqr
        return RotMat
    
    
    def body2inertial_align(tbaligned_B: np.array, target_I: np.array, normalize = 0):
        ''' 
        finds angles to align tbaligned_B vector in body frame with 
        vector target_I in the inertial frame. Normalizes if possible
        
        '''
        if (not any(tbaligned_B)) or (not any(target_I)):
            # zero vector passed
            return np.array([0.,0.,0.]) # dummy
        
        if normalize in [1,3]:
            tbaligned_B = tbaligned_B/(np.linalg.norm(tbaligned_B)+1e-8) 
        if normalize in [2,3]:
            target_I = target_I/(np.linalg.norm(target_I)+1e-8) 
        
        ''' 
        #with warnings.catch_warnings(): 
        with suppress_stdout():
            with warnings.catch_warnings(): 
                RotMat_estimate, _ = R.align_vectors(tbaligned_B[None,:], target_I[None,:]) 
        
        '''
        RotMat_estimate, _ = R.align_vectors(tbaligned_B[None,:], target_I[None,:]) 

        #''' 
        # find angles to get tbaligned_B to target_I (body to inert)
        angles_I = RotMat_estimate.as_euler('xyz')
        return angles_I
    
    
    def quats_body2quats_rates(pqr):
        # angles are euler angles in inertial frame
        p,q,r = pqr
        RotMat_pqr = 0.5*np.array([[0.,   r, -q,  p],
                                   [-r,  0.,  p,  q],
                                   [ q,  -p, 0.,  r],
                                   [-p,  -q, -r, 0.],
                               ]) 

        # use dquats = RotMat_pqr @ quats, expects quats in x,y,z,w format
        return RotMat_pqr
    
    def quats_quats2body_rates(quats):
        # angles are euler angles in inertial frame
        x,y,z,w = quats
        RotMat_quats = 2.*np.array([[ w,  z, -y, -x],
                                    [-z,  w,  x, -y],
                                    [ y, -x,  w, -z],
                               ]) 

        # use pqr = RotMat_pqr @ dquats, expects quats in x,y,z,w format
        return RotMat_quats
    
    def check():
        '''
        following resemblences are quaternion integration EOM 
        '''
        
        euler_I = np.array([-0.41199952, -0.33,  0.71587637]) # euler
        
        
        x_I = np.array([1.,2.,3.]) # random vector
        x_I /= np.linalg.norm(x_I)
        
        
        rotmat_euler_I2B = R.from_euler('xyz', euler_I) 
        rotmat_euler_B2I = rotmat_euler_I2B.inv()
        
        
        q_euler_I = rotmat_euler_I2B.as_quat()
        
        
        # integrate would be done here
        
        rotmat_q_I2B = R.from_quat(q_euler_I)
        rotmat_q_B2I = rotmat_q_I2B.inv()
        
        x_euler_B = rotmat_euler_I2B.apply(x_I)
        x_q_B = rotmat_q_I2B.apply(x_I)
        
        assert np.allclose(x_euler_B, x_q_B)
        
        euler_I2B_imp = rotmat_euler_I2B.as_euler('xyz') 
        euler_B2I_imp = rotmat_euler_B2I.as_euler('zyx')[::-1]*-1 
        
        print(euler_I2B_imp)
        print(euler_B2I_imp)
        
        assert np.allclose(euler_I2B_imp, euler_B2I_imp)
        
        return 
    
    def get_rotmatB2I_XnoRoll(T_I):
        ''' 
        # for the T_B = [1,0,0] i.e. body x-axis to desired inertial vector
        
         zyx
            
        [x, = [cos(psi)cos(theta) , @[1,0,0]
         y,    sin(psi)cos(theta) , 
         z]_I -sin(theta) ] & phi = 0

        USE:
            T_B = np.array([1,0,0])

            T_I = np.random.randn(3)
            #T_I =  np.array([0.3,-0.8,0.1])
            #T_I = np.array([-0.39794304, -0.61183188, -0.68359571])
            T_I /= np.linalg.norm(T_I)
        ''' 
        T_I = T_I/np.linalg.norm(T_I)
        
        theta = np.arcsin(-1*(T_I[2]))
        theta_cos = np.cos(theta)
        
        psi_cos = np.arccos(T_I[0]/theta_cos) # [0; pi]
        psi_sin = np.arcsin(T_I[1]/theta_cos) # [-0.5*pi; 0.5*pi]
        
        psi_cos_sign = int(psi_cos/abs(psi_cos))
        psi_sin_sign = int(psi_sin/abs(psi_sin))
        
        close = np.allclose(psi_cos, abs(psi_sin))
        
        if (close) and (psi_sin_sign == 1):
            # case A
            psi = psi_cos
            case = 'A'
        elif (close) and (psi_sin_sign == -1):
            # case B
            psi = psi_sin
            case = 'B'
        elif (not close) and (psi_sin_sign == 1):
            # case C
            psi = psi_cos
            case = 'C'
        elif (not close) and (psi_sin_sign == -1):
            # case D
            psi = -np.pi - psi_sin
            case = 'D'
        
        print(f'\n{case}\n')
        print([psi_cos, psi_sin])
        '''
        T_I_imp = np.array([np.cos(psi)*np.cos(theta),
                            np.sin(psi)*np.cos(theta),
                            -np.sin(theta),
                            ])
        msg = f'\nT_I:    {T_I}\nT_I_imp:{T_I_imp}'
        assert np.allclose(T_I_imp, T_I), msg
        
        rotmat_BI = get_rotmatB2I_XnoRoll(T_I)
        T_I_imp = rotmat_BI.apply(T_B)
        # Verify the result
        msg = f'\nT_I:    {T_I}\nT_I_imp:{T_I_imp}'
        assert np.allclose(T_I_imp, T_I), msg

        #'''

        phi = 0 
        #https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        matrix_xyz = np.array([
            [np.cos(theta)*np.cos(psi), 
             -np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi), 
             np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)],
            
            [np.cos(theta)*np.sin(psi), 
             np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi), 
             -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)],
            
            [-np.sin(theta), 
             np.sin(phi)*np.cos(theta), 
             np.cos(phi)*np.cos(theta)]
        ])
        rotmat_BI = R.from_matrix(matrix_xyz)
        return rotmat_BI
    
    def get_rotmatB2I_ZnoRoll(T_I):
        ''' 
        # for the T_B = [0,0,1] i.e. body z-axis to desired inertial vector

         zyx
            
        [x, = [cos(psi)sin(theta) , @[0,0,1]
         y,    sin(psi)sin(theta) , 
         z]_I  cos(theta) ] & phi = 0
        
        USE:
            T_B = np.array([0,0,1])

            T_I = np.random.randn(3)
            #T_I =  np.array([0.3,-0.8,0.1])
            #T_I = np.array([-0.39794304, -0.61183188, -0.68359571])
            T_I /= np.linalg.norm(T_I)
        ''' 
        T_I = T_I/np.linalg.norm(T_I)
        
        theta = np.arccos(T_I[2])
        theta_sin = np.sin(theta)
        
        psi_cos = np.arccos(T_I[0]/theta_sin) # [0; pi]
        psi_sin = np.arcsin(T_I[1]/theta_sin) # [-0.5*pi; 0.5*pi]
        
        psi_cos_sign = int(psi_cos/abs(psi_cos))
        psi_sin_sign = int(psi_sin/abs(psi_sin))
        
        close = np.allclose(psi_cos, abs(psi_sin))
        
        if (close) and (psi_sin_sign == 1):
            # case A
            psi = psi_cos
            case = 'A'
        elif (close) and (psi_sin_sign == -1):
            # case B
            psi = psi_sin
            case = 'B'
        elif (not close) and (psi_sin_sign == 1):
            # case C
            psi = psi_cos
            case = 'C'
        elif (not close) and (psi_sin_sign == -1):
            # case D
            psi = -np.pi - psi_sin
            case = 'D'
        
        print(f'\n{case}\n')
        print([psi_cos, psi_sin])
        '''
        T_I_imp = np.array([np.cos(psi)*np.cos(theta),
                            np.sin(psi)*np.cos(theta),
                            -np.sin(theta),
                            ])
        msg = f'\nT_I:    {T_I}\nT_I_imp:{T_I_imp}'
        assert np.allclose(T_I_imp, T_I), msg
        
        rotmat_BI = get_rotmatB2I_ZnoRoll(T_I)
        T_I_imp = rotmat_BI.apply(T_B)

        # Verify the result
        msg = f'\nT_I:    {T_I}\nT_I_imp:{T_I_imp}'
        assert np.allclose(T_I_imp, T_I), msg
        #'''

        phi = 0 
        #https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        matrix_xyz = np.array([
            [np.cos(theta)*np.cos(psi), 
             -np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi), 
             np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)],
            
            [np.cos(theta)*np.sin(psi), 
             np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi), 
             -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)],
            
            [-np.sin(theta), 
             np.sin(phi)*np.cos(theta), 
             np.cos(phi)*np.cos(theta)]
        ])
        rotmat_BI = R.from_matrix(matrix_xyz)
        return rotmat_BI
    
    def get_rotmatB2I_ZnoYaw(T_I):
        ''' 
        # for the T_B = [0,0,1] i.e. body z-axis to desired inertial vector

         zyx
            
        [x, = [sin(theta)cos(phi), @[0,0,1]
         y,    -sin(phi), 
         z]_I  cos(theta)cos(phi) ] & psi = 0
        
        USE:
            T_B = np.array([0,0,1])

            T_I = np.random.randn(3)
            #T_I =  np.array([0.3,-0.8,0.1])
            #T_I = np.array([-0.39794304, -0.61183188, -0.68359571])
            T_I /= np.linalg.norm(T_I)
        ''' 
        T_I = T_I/np.linalg.norm(T_I)
        phi = np.arcsin(-T_I[1])
        phi_cos = np.cos(phi) #+ 1e-6
        
        theta_sin = np.arcsin(T_I[0]/phi_cos) # [-0.5*pi; 0.5*pi]
        theta_cos = np.arccos(T_I[2]/phi_cos) # [0; pi]
        
        theta_cos_sign = int(theta_cos/abs(theta_cos))
        theta_sin_sign = int(theta_sin/abs(theta_sin))
        
        close = np.allclose(theta_cos, abs(theta_sin))
        
        if (close) and (theta_sin_sign == 1):
            # case A
            theta = theta_cos
            case = 'A'
        elif (close) and (theta_sin_sign == -1):
            # case B
            theta = theta_sin
            case = 'B'
        elif (not close) and (theta_sin_sign == 1):
            # case C
            theta = theta_cos
            case = 'C'
        elif (not close) and (theta_sin_sign == -1):
            # case D
            theta = -np.pi - theta_sin
            case = 'D'
        
        print(f'\n{case}\n')
        print([theta_cos, theta_sin])
        '''
        T_I_imp = np.array([np.cos(psi)*np.cos(theta),
                            np.sin(psi)*np.cos(theta),
                            -np.sin(theta),
                            ])
        msg = f'\nT_I:    {T_I}\nT_I_imp:{T_I_imp}'
        assert np.allclose(T_I_imp, T_I), msg
        
        rotmat_BI = get_rotmatB2I_ZnoYaw(T_I)
        T_I_imp = rotmat_BI.apply(T_B)

        # Verify the result
        msg = f'\nT_I:    {T_I}\nT_I_imp:{T_I_imp}'
        assert np.allclose(T_I_imp, T_I), msg
        #'''
        psi = 0 
        #https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        matrix_xyz = np.array([
            [np.cos(theta)*np.cos(psi), 
             -np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi), 
             np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)],
            
            [np.cos(theta)*np.sin(psi), 
             np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi), 
             -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)],
            
            [-np.sin(theta), 
             np.sin(phi)*np.cos(theta), 
             np.cos(phi)*np.cos(theta)]
        ])
        rotmat_BI = R.from_matrix(matrix_xyz)
        return rotmat_BI
    
    
#%% GYM ENVIRONMENT MAIN FUNCTIONS

class PursuitEvasionEnv(MultiAgentEnv, TaskSettableEnv):
    '''
    README: following outlines the design ideology
    
    SPACES:
    - keep a state_space container which is consistent across agents
        - e.g. contains position, velo, heading etc.
    - Observation spaces and action spaces can be completely different
        - obs: agents can see different things e.g. pursuer has more sensors
        - act: e.g. evaders can control acceleration while pursuers cant
            - As a consequence this implies different/seperate models for agents
    - Observations are computed from the consistent state space class 
    
    
    LEARNING:
    - GEN: i think it would be interesting to use independent learning
        - rather than centralized
        - might even entail different trainers
    
    TODO
    - evrything to tensor format?
    - see ideas for curriculum learning in .ipynb
    '''
    metadata = {
    "render.modes": ["human"]
    }
    render_mode = "human"
    name = 'PEG1'
    
    def __init__(self, config=None, verbose = False):
        ## Setup configuration
    
        self.setting_eval = config.get('setting_eval', False) # we set this in the reset not at initialization!
        self.verbose = verbose
        
        # initialization settings (position & heading)
        #self.r0_e_range0 = np.array(config.get('r0_e_range', [0.5,1.0]))
        self.r0_e_range0 = np.array(config.get('r0_e_range', [0.5,1.0])) # 7.5 seconds
        #self.r0_e_range0 = np.array(config.get('r0_e_range', [0.25,0.5])) # 5 seconds
        self.cone0_e_range = config.get('cone0_e_range', 0.25*np.pi)
        # configure sampling conditions for initialization, ensure > 0 all
        # for documentation see .reset_state_space()
        
        # time settings
        self.t_delta = config.get('t_delta', 0.1) # 0.1 rano
        self.t_limit0 = config.get('t_limit', 7.5) # 20s rano
        # following is required for proper CL
        self.t_idx_limit0 = int(np.ceil(self.t_limit0/self.t_delta))
        self.t_idx_limit = int(self.t_idx_limit0) # initialize

        ## Setup 
        #'''
        
        ''' 
        config['config['SA']'] = {'Dimension':2, # 2d or 3d of state space
                          'DOF':1, # [1,2,4] degrees of freedom of action space
                          ## additional
                          'e_v_rel':0.75,
                          }
        #''' 
        
        ## Setup agents
        self.p_n = config.get('p_n', 1)
        self.e_n = config.get('e_n', 1)
        
        self.agent_id = [*[f"p{i}" for i in range(self.p_n)],
                       *[f"e{j}" for j in range(self.e_n)],
                       ]
        
        self.agent_id_stepped = self.agent_id.copy()
        self._agent_ids = set(self.agent_id) # note that this one is alphabetically ordered!
        self.agent_specs = {aid:{'idx':aid_idx} for aid_idx, aid in enumerate(self.agent_id)}
        self.agent_n = len(self.agent_id)
        assert len(self.agent_id) == len(self._agent_ids), 'duplicate agent ids encountered'
        #assert bool(self.p_n)+bool(self.e_n) == 2, 'at least one of each agents required'
    
        ### = S.O.R.A SETUP =
        self.t = None
        self.t_idx = None
        self.BS_idxLag = {aid: None for aid in self.agent_id} # BlindSight idx lag 
        ## State space
        # configure DOF and Dim
        self.configure_StateActionSpace(config) 
        
        ## Observation space
        self._spaces_in_preferred_format = True
        #self._obs_space_in_preferred_format = True 
        '''
        Note: the attribute ._spaces_in_preferred_format informs ray that 
        individual agents have different observational/action spaces. This is 
        required for proper multi-agent environment, it does not inform the policies
        operating on this environment
        SOURCE:  https://discuss.ray.io/t/multiagents-type-actions-observation-space-defined-in-environement/5120/6
        '''
        self.observation_space = gym.spaces.Dict(
            {# pursuer observation spaces
             **{aid: gym.spaces.Box(low=self.observation_space_limitB, # reverse top
                                       high=self.observation_space_limitT,
                                       shape=(len(self.observation_space_limitB),),
                                       dtype=np.float64) for aid in self.agent_id if 'p' in aid},
             # evader observation spaces
             **{aid: gym.spaces.Box(low=self.observation_space_limitB, # reverse top
                                      high=self.observation_space_limitT,
                                      shape=(len(self.observation_space_limitB),),
                                      dtype=np.float64) for aid in self.agent_id if 'e' in aid},
            }
        )
        ## REWARD
        # TODO MOVE TO INTRO & CONNECT TO CL
        self.escape_factor = config.get('escape_c', 1e8) # 2
        self.intercept_distance0 = config.get('intercept_distance', 0.15) #0.15) # [0.15,0.25]
        self.intercept_distance = float(self.intercept_distance0)
        # multiple governing how far from initial distance must be reached to reward espace
        
        ## Action space
        self.action_space = gym.spaces.Dict(
            {# action spaces
             **{aid: gym.spaces.Box(low=self.action_space_limit[aid]['B'],  # reverse top
                                        high=self.action_space_limit[aid]['T'], 
                                        shape=(len(self.action_space_limit[aid]['B']),), 
                                        dtype=np.float64) for aid in self.agent_id}, 
             }
        )
        for aid in self.A_gain:
            ## override agent action space in case
            
            dim_a_gain = 2 # self.dim_a
            ''' 
            self.action_space[aid] = \
                gym.spaces.Box(low=np.array([-10.]*dim_a_gain),  # reverse top
                                high=np.array([10.]*dim_a_gain), 
                                shape=(2,), 
                                dtype=np.float64)
            ''' 
            self.action_space[aid] = \
                gym.spaces.Box(low=np.array([0., -1.]),  # K1 (I,R,V), K2 (los/dlos)
                                high=np.array([2.*np.pi - 1/6*np.pi, 1.]), 
                                shape=(2,), 
                                dtype=np.float64)
            #'''
        
        ### ===
        ## Setup state sizes/dimensions
        self.agent_dims = {}
        for aid in self.agent_id:
            self.agent_dims[aid] = [
                6,
                get_preprocessor(self.observation_space[aid])(self.observation_space[aid]).size,
                self.dim_a, #get_preprocessor(self.action_space[aid])(self.action_space[aid]).size,
                self.agent_specs[aid]['Trnn'],
                
            ]
            # [state_size, observation_size, action_size, rnn_time_dim]

        ## seeding
        # TODO action_space & observation_space can also be seeded
        self.remote, self.worker_idx, self.num_workers, \
            self.vector_env_index, self.seed_unique0, self.seed_unique = \
                None, None, None, None, None, None
        try:
            self.remote = config.remote
            self.worker_idx = config.worker_index # worker index
            self.num_workers = config.num_workers  
            self.vector_env_index = config.vector_index # env per worker index? or index of env globally?
            '''
            I hope that the vector index can tell me the environment index
            within a worker
            
            atm i do not know whether
            - env_vector_index says anything at all outside VectorEnv types
            - (if it does say anything) env_vector_index would be the index 
                within a worker or across all workers (i.e. whether the 
               counter is reset within a worker)
            
            if so then the following seed should ensure everybody is
            (& remains) unique
            
            LINKS;
            https://discuss.ray.io/t/rollout-worker-index-with-externalenv/813/2
            https://discuss.ray.io/t/reproducible-training-setting-seeds-for-all-workers-environments/1051/11
            '''
            
            self.seed_unique0 = int((self.worker_idx)*10000 + self.vector_env_index*500) 
            # minimum 500 envs space before repetition 
            self.seed_unique = int(self.seed_unique0)
            # large values to ensure incrementing of env does not encroach into eachothers seeds
        except AttributeError:
            pass
        
        ## CURRICULUM
        # initialize curriculum task 
        self.set_task(config.get('CL_level', 0)) # start curriculum learning at level 0
        
        ## finalize & return
        self.time_initialized = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.resetted = False
        self.offline = False
        self.state_space_offline = None
        self.history_all = []
        super().__init__()
        return
    
    def set_seed(self,seed):
        if not isinstance(seed, type(None)):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            # scs stats is not needed to be set, since it uses numpy.random
            # note that noise_dist seed is defined in reset
            '''
            if self.DO_NOISE:
                # reset the rng for the noise distribution
                #https://stackoverflow.com/questions/16016959/scipy-stats-seed
                self.noise_dist.random_state = np.random.RandomState(seed)
            '''
        return 
    
    def set_setting_eval(self, setting_eval):
        self.setting_eval = setting_eval
        return 



#%% CONFIGURE SA
    def configure_StateActionSpace(self, config: dict):
        
        self.dim_s = int(config['SA']['Dimension'])
        self.dim_a = int(config['SA']['DOF'])
        self.A_gain = config['SA'].get('A_gain',{})
        self.SAcode = f'{self.dim_s}D{self.dim_a}DOF'
        self.SAcode_num = None
        
        ## initialization related
        config['CL'] = config.get('CL',{})
        config['BS'] = config.get('BS',{})
        config['NOISE'] = config.get('NOISE',{})
        config['specs'] = config.get('specs',{})
        ## dynamics related
        specs_default = {## observation related
                         'Trnn':10,
                         'Tdelay':0, 
                         #'v_rel':1.0,
                         'a_pos':True, 
                         }
        self.rate_limit = config['SA'].get('rate_limit',0.85*np.pi) # TODO THETA
        
        self.agent_specs['p0'].update({**specs_default.copy(),
                                        #'Tdelay':0, 
                                        'v_rel':1., ## alex
                                        'tau': 0.03,
                                        'rate_limit':self.rate_limit, 
                                      })
        self.agent_specs['p0'].update(config['specs'].get('p0',{}).copy()) # update/override with provided specs
        
        self.agent_specs['e0'].update({**specs_default.copy(),
                                          # 'Tdelay':0, 
                                           'v_rel':0.5,
                                           'tau': 0.01,#,
                                           'rate_limit':self.rate_limit, 
                                       })
        self.agent_specs['e0'].update(config['specs'].get('e0',{}).copy())  # update/override with provided specs
        
        # max 10 (= full circle in 0.1 seconds)
        
        ### CONTROLPANEL AAAA
        
        #self.state_space_next = np.zeros((2,3,6), dtype = np.float64)
        ##
        self.DO_BS = config['BS'].get('do_bs',False) # Blindsighting game
        self.BS_CL_ACTIVE = config['BS'].get('bs_cl_active',3) # integer, at (i.e. >=) which CL level BS becomes active
        self.BS_gamma_thres = config['BS'].get('bs_gamma_thres',0.9) if self.DO_BS else None # TODO MOVE TO CONFIG
        
        ##
        self.AL_DIM = 0 # 2
        self.AL_idx = list(range(-self.AL_DIM,0))*bool(self.AL_DIM)
        self.VALUE_DIM = 9 # 10
        
        self.VALUE_DIM += self.DO_BS*2*bool(self.VALUE_DIM) # add blindsight features
        ##
        
        self.DO_NOISE = config['NOISE'].get('do_noise',False) # self.noise_dist is moved to .reset()
        #obs_dim = 6+5#+13
        
        self.noise_dist_params = {}
        if self.DO_NOISE:
            self.noise_scalar = 1.
            self.noise_dist_params = {# (x,dx,ddx)
                    'mu':np.array([0.01, 0.05, 0.05])*self.noise_scalar*0., 
                    'var':np.array([0.002, 0.002, 0.002])*(self.noise_scalar**2), 
                    'cov':np.array([0.3, 0.3, 0.3])*(self.noise_scalar**2)*0., 
                } # ensure lists!
            # setup mu vector
            self.noise_dist_mu = np.repeat(self.noise_dist_params['mu'],3) # (9,)
            
            # setup diagonal block sigma matrix 
            self.noise_dist_sigma = []
            for d in range(3):
                sigma_block = np.eye(3)*self.noise_dist_params['var'][d]
                sigma_block[~np.eye(3,dtype=bool)] = self.noise_dist_params['cov'][d]
                self.noise_dist_sigma.append(sigma_block)
            self.noise_dist_sigma = scl.block_diag(*self.noise_dist_sigma)

            # check sizes
            assert len(self.noise_dist_mu) == \
                self.noise_dist_sigma.shape[0] == \
                    self.noise_dist_sigma.shape[1], 'Noise dist dims not correct'
            '''
            self.noise_dist = scs.multivariate_normal(self.noise_dist_mu,
                                                      self.noise_dist_sigma, # sig^2
                                                      #seed = # infers numpy seed
                                                      ) 
            #'''
            # NOTE THAT THIS DIST RNG IS RESET WITH SET SEED in the .reset()!
        
        ## Velocity
        # option 1: constant (random magnitude) velocity (for SA 21/32/33)
        self.V_REL_RANDOM = True # whether to randomly sample game statics
        self.V_REL_RANDOM_CL = True # whether to include v_rel in CL process
        # option 2: control of velocity (for SA 22/34)
        self.V_REL_CONTROL = config['SA'].get('v_rel_control', False)
        self.V_REL_CONTROL_CL = True
        # 
        self.A_CONTROL_noG = config['SA'].get('a_control_noG', False)
        self.A_CONTROL_CL = config['CL'].get('a_control_cl', False)
        
        ##
        self.A_CONTROL_3D3DOF = config['SA'].get('a_control_3D3DOF', False)
        
        ##
        self.SELFSPIEL = config['SA'].get('SELFSPIEL',False)
        

        ##
        # BBBBBBBB
        self.determine_state_space_reward = self.zerosum_reward_main_USED
        self.CL_START_DIS_REWARD = config['CL'].get('CL_start_dis_reward', False) # whether to give dense distance reward at CL = 0
        if self.CL_START_DIS_REWARD:
            self.determine_state_space_reward = self.zerosum_reward_main_OLDv1
        self.TRUNC_EARLY = config['CL'].get('trunc_early', False) # only works if CL_START_DIS_REWARD!
        
        self.INTERCEPT_DISTANCE_CL = config['CL'].get('intercept_distance_cl', False) # whether intercept_distance is subject to CL
        
        if self.A_gain:
            self.obs_last = {aid:None for aid in self.agent_id}
            assert (self.dim_a in [2,3]) & (self.dim_s == 3), 'a_dim != 2 and/or s_dim != 3 for a_gain not currently supported'
        '''
        NOISE GENERATOR NOTES;
        
        IDEA IS TO SETUP THE NOISE GENERATOR HERE and sample from
        it every iteration (instead of setting up generator every time)
        
        Also, we define the generator for the _3d_main func,
        and add it there!
        
        note;
        - correlated noise might be possible for the same axis,
            or if two inputs are related in the same way e.g. d & dd
        - TODO MAKE CONFIG SPECIFIC I.E. ADD TO SPECS, ALSO MAKE SKIPPABLE
            IN CASE WE DONT DEFINE IDEAL CASE
        ''' 

        ## CONSTANT (OR INITIAL) VELOCITY SETTING
        # note that for configs with acceleration v_rel 
        # only defines the initial velocity, after that its free
        
        self.errors_last = {aid:None for aid in self.agent_id}
        self.OVERRIDE_ACTION_E0 = False
        
        self.action_space_limit = {aid: {} for aid in self.agent_id}
        
        ## CONFIG SPECIFICS
        self.thrust_orientation = [1.,0.,0.]
        self.rotmat_Z2X = np.eye(3)
        self.ANGLES_SIGN = 1. # in 3D everything is fine
        if self.dim_s == 2:
            self.ANGLES_SIGN = -1. # in 2d all psi are reversed in sign! i.e. body to inertial is reversed
            if self.dim_a == 1:
                self.SAcode_num = 21 # 2D1DOF
                '''
                2 dimensional task environment and 1 DOF.
                Specifically;  
                - environment considers (x,y)-axis (i.e. z = 0)
                - Controller controls yaw-rate (rotation around z-axis)
                - Constant velocity
                '''

                
                ## define functions
                self.initialize_state_space = self.initialize_state_space_2D
                self.update_state = self.update_state_2D1DOF
                
                self.observation_pursuer = self.observation_pursuer_2D1DOF
                #self.observation_evader = currently wrapper of pursuer
                
                ## define space limits
                # actions
                for aid in self.agent_id:
                    self.action_space_limit[aid] = {
                                'B':np.array([-self.agent_specs[aid]['rate_limit']]*self.dim_a),
                                'T':np.array([self.agent_specs[aid]['rate_limit']]*self.dim_a),
                        } 

                # TODO CONSIDER SCALING BY /self.t_delta? NO, integrator already considers howmany time step size
                # observations
                obs_Nx = 0 
                self.observation_space_limitB = np.array([-np.pi, -np.inf,
                                                          *([-np.inf]*(obs_Nx+self.AL_DIM+self.VALUE_DIM)),
                                                          ], dtype = np.float64) # bottom limits
                self.observation_space_limitT = np.array([ np.pi, np.inf,
                                                          *([np.inf]*(obs_Nx+self.AL_DIM+self.VALUE_DIM)),
                                                          ], dtype = np.float64) # top limits
                # d, dd, psi, dpsi
                # TODO evader/pursuer agent specific?
                
            elif self.dim_a == 2:
                self.SAcode_num = 22 # 2D2DOF
                '''
                2 dimensional task environment and 2 DOF.
                Specifically;  
                -
                
                TODO COMBINE WITH self.dim_a = 1
                '''
                raise Exception('2DOF 2D not implemented')
                # heading control and thrust/acceleration/velocity control
            
        elif self.dim_s == 3 and (self.dim_a == 2 or self.dim_a == 3) and not self.A_CONTROL_3D3DOF:
            '''
            3 dimensional task environment with 2 or 3 DOF (specified later).
            Both cases have significant overlap with constant velocity
            
            Specifically;  
            - environment considers (x,y,z)-axes
            - Constant velocity
            '''
                
            ## define space limits
            # action
            for aid in self.agent_id:
                self.action_space_limit[aid] = {
                            'B':np.array([-self.agent_specs[aid]['rate_limit']]*self.dim_a),
                            'T':np.array([self.agent_specs[aid]['rate_limit']]*self.dim_a),
                    }


            
            # TODO CONSIDER SCALING BY /self.t_delta? NO, integrator already considers howmany time step size
            # observations
            obs_Nx = 5 ## TODO EXTRA_ACTION_CODE set to 0
            self.observation_space_limitB = np.array([*([-np.pi, -np.inf]*self.dim_a),
                                                      *([-np.inf]*(obs_Nx+self.AL_DIM+self.VALUE_DIM)),
                                                      ], dtype = np.float64)
            self.observation_space_limitT = np.array([*([np.pi, np.inf]*self.dim_a), 
                                                      *([np.inf]*(obs_Nx+self.AL_DIM+self.VALUE_DIM)),
                                                      ], dtype = np.float64) 
            
                
            if self.dim_a == 2:
                self.SAcode_num = 32 # 3D2DOF
                '''
                Specifically;  
                - Controller controls pitch- and yaw-rate (rotation around y- & z-axis)
                '''
                ## define functions
                self.initialize_state_space = self.initialize_state_space_3D
                self.update_state = self.update_state_3D2DOF
            
                self.observation_pursuer = self.observation_pursuer_3D2DOF
                #self.observation_evader = currently wrapper of pursuer
            
            elif self.dim_a == 3:
                self.SAcode_num = 33 # 3D3DOF
                '''
                Specifically;  
                - Controller controls roll-, pitch- and yaw-rate
                '''
                ## define functions
                self.initialize_state_space = self.initialize_state_space_3D
                #note we initialize 3dof like 2dof i.e. no roll yet and alignment with evader
                self.update_state = self.update_state_3D3DOF
                
                self.observation_pursuer = self.observation_pursuer_3D3DOF
                #self.observation_evader = currently wrapper of pursuer


        #######################################################################        
        elif (self.dim_s == 3 and self.dim_a == 4) or ((self.dim_s == 3) and (self.dim_a == 3) and self.A_CONTROL_3D3DOF):
            self.SAcode_num = 34 # 3D4DOF
            '''
            3 dimensional task environment and 4 DOF.
            Specifically;  
            -
            '''
            for aid in self.agent_id:
                self.action_space_limit[aid] = {
                            'B':np.array([-1.] + [-self.agent_specs[aid]['rate_limit']]*3),
                            'T':np.array([ 1.] + [self.agent_specs[aid]['rate_limit']]*3),
                    }
                '''
                ## QUATCONTROL
                self.dim_a = 5
                action_space_limits = np.array([
                    [-1.] + [-self.rate_limit/2]*4,
                    [1.] + [self.rate_limit/2]*4,
                    ]) # notice a_c (or v_c) is [-1,1] and is scaled in the update func
                #'''
                
            self.OVERRIDE_ACTION_E0 = False # TODO REMOVE
            self.DUMMY_EVADER_3D4DOF = config['SA'].get('dummy_evader_3D4DOF', False)
            self.DUMMY_POINT_3D4DOF = config['SA'].get('dummy_point_3D4DOF', False)
            if self.DUMMY_POINT_3D4DOF:
                self.DUMMY_EVADER_3D4DOF = True
                
            
            if self.V_REL_CONTROL:
                ## control velocity instead of thrust
                self.update_state = self.update_state_3D4DOF_Vcontrol
                # note that self.thrust_orientation = [1.,0.,0.] is used

            else:
                ## default case; control of thrust with gravity
                #self.update_state = self.update_state_3D4DOF # ideal
                #self.update_state = self.update_state_3D4DOF_TF # non-ideal
                self.update_state = self.update_state_3D4DOF_QUAT # non-ideal, using quarternions
                self.thrust_orientation = [0.,0.,1.] ## ALEX
                self.rotmat_Z2X = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]) 
                # rotation matrix for theta = 0.5pi i.e. (x,y,z)->(z,y,-x)
                
                self.DUMMY_EVADER_3D4DOF_V_MAX = 3.#6.

                ## extract agent specs
                a_g_p0, a_g_e0 = 9.81, 0. ## alex
                
                '''
                thrust_mag_p0_B, thrust_mag_p0_T = 9.81*1, 9.81*1
                thrust_mag_e0_B, thrust_mag_e0_T = 1., 9.81*1 
                a_T_range_p0 = self.agent_specs['p0'].get('a_T_range',[thrust_mag_p0_B, thrust_mag_p0_T-thrust_mag_p0_B]) # ensure both are >0!
                a_T_range_e0 = self.agent_specs['e0'].get('a_T_range',[thrust_mag_e0_B, thrust_mag_e0_T-thrust_mag_e0_B]) # ensure both are >0!
                '''
                thrust_mag_p0 = 9.81*1.
                a_T_range_p0 = self.agent_specs['p0'].get('a_T_range',[thrust_mag_p0*1, thrust_mag_p0*1]) # ensure both are >0!
                
                thrust_mag_e0 = 9.81*5. #0.5
                a_T_range_e0 = self.agent_specs['e0'].get('a_T_range',[thrust_mag_e0*1, thrust_mag_e0*1]) # ensure both are >0!
                    
                
                # Note; max thrust is total of both values in this list                

                drag_C_p0 = [#0.5, 0.5, 0.5, # v ## ALEX
                             0.5, 0.5, 0.5, # v ## ALEX
                             #0., 0., 0., # v
                             
                             0., 0., 0., # v^2
                             #0.25, 0.25, 0.25, # v^2
                             
                             False, False, False, # null v
                             ]
                
                self.null_delay = 2#3
                drag_C_e0 = [#0.5, 0.5, 0.5, # v
                             0., 0., 0., # v
                             #1.5, 1.5, 1.5, 
                             #5., 5., 5., 
                             
                             0., 0., 0., # v^2
                             #0.75, 0.75, 0.75, # v2
                             #5., 5., 5., 
                             
                             True, True, True, # null v
                             #False, False, False, # null v
                             ]
                
                self.a_e0_CL_factors = self.agent_specs['e0'].get('a_e0_CL_factors',[0.5, 0.5]) # keep 1. at bottomto counteract gravity
                # note that this is a shrinkage of the effective range of 0.75, not 0.5
            
                ## Alternative: control of thrust w/o gravity
                if self.A_CONTROL_noG:
                    
                    self.DUMMY_EVADER_3D4DOF_V_MAX = 3.#3.
                    
                    ## extract agent specs
                    a_g_p0, a_g_e0 = 0., 0.
                    thrust_mag_B, thrust_mag_T = 1., 9.81*1 # note that whatever number is input here, max thrust is x2 
                    
                    a_T_range_p0 = self.agent_specs['p0'].get('a_T_range',[thrust_mag_B, thrust_mag_T-thrust_mag_B]) # ensure both are >0!
                    a_T_range_e0 = self.agent_specs['e0'].get('a_T_range',[thrust_mag_B, thrust_mag_T-thrust_mag_B]) # ensure both are >0!
                    # Note; max thrust is total of both values in this list
                
                ## Additional case 3D3DOF
                if (self.dim_s == 3) and (self.dim_a == 3) and self.A_CONTROL_3D3DOF:
                    # NOTE THAT THIS CLAUSE OVERRIDES PREVIOUS ATTRIBUTES
                    self.SAcode_num = 334 # 3D3DOF, known as 334 = acceleration for 33
                    self.update_state = self.update_state_3D3DOF_Acontrol # override
                    
                    a_g_p0, a_g_e0 = 0., 0.
                    
                    self.DUMMY_EVADER_3D4DOF_V_MAX = 3.#3.
                    
                    a_T_range_p0 = self.agent_specs['p0'].get('a_T_range',[np.nan, 2*9.81]) # ensure both are >0!
                    a_T_range_e0 = self.agent_specs['e0'].get('a_T_range',[np.nan, 2*9.81]) # ensure both are >0!
                    
                    ''' 
                    ## Action limits (override)
                    for aid in self.agent_id:
                        self.action_space_limit[aid] = {
                                    'B':np.array([-1.]*3),
                                    'T':np.array([ 1.]*3),
                                }
                    '''
                    for aid in self.agent_id:
                        self.action_space_limit[aid] = {
                                    'B':np.array([-2*9.81]*3),
                                    'T':np.array([ 2*9.81]*3),
                                }
                    #''' 
                    # notice a_c (or v_c) is [-1,1] and is scaled in the update func
                    
                    
                    
                ## update agent specs
                self.agent_specs['p0'].update({
                        'drag_C':self.agent_specs['p0'].get('drag_C',drag_C_p0),
                        'a_T_range':a_T_range_p0, # ensure both are >0!
                        'a_g':a_g_p0,
                    })
                self.agent_specs['e0'].update({
                        'drag_C':self.agent_specs['e0'].get('drag_C',drag_C_e0),
                        'a_T_range':a_T_range_e0, # ensure both are >0!
                        'a_g':a_g_e0,
                    })
                
                self.a_e0_CL_factors = np.array(self.a_e0_CL_factors)
                
            ##

            
            # TODO CONSIDER SCALING BY /self.t_delta? NO, integrator already considers howmany time step size
            # observations
            
            obs_Nx = 20#17 #4# 1+7#1+9#6#1+9#5+8 ## TODO EXTRA_ACTION_CODE set to 0
            obs_Nlos = 0#3
            
            
            self.observation_space_limitB = np.array([*([-np.pi, -np.inf]*obs_Nlos),
                                                      *([-np.inf]*(obs_Nx+self.AL_DIM+self.VALUE_DIM)),
                                                      ], dtype = np.float64)
            self.observation_space_limitT = np.array([*([np.pi, np.inf]*obs_Nlos), 
                                                      *([np.inf]*(obs_Nx+self.AL_DIM+self.VALUE_DIM)),
                                                      ], dtype = np.float64) 
            
            self.initialize_state_space = self.initialize_state_space_3D4DOF
            #note we initialize 3dof like 2dof i.e. no roll yet and alignment with evader
            #self.update_state = self.update_state_3D4DOF
            
            self.observation_pursuer = self.observation_pursuer_3D4DOF
            
        else:
            raise Exception('Invalid SA config provided!')
        
            
        ## velocity magnitude control
        # constant velocity setting
        if self.SAcode_num in [21,32,33]:
            if self.V_REL_RANDOM:
                # extract velocity settings 
                self.agent_specs['e0']['v_rel_mean0'] = self.agent_specs['e0'].get('v_rel_mean',0.5) # 0. stationary, 0.75 rano2021
                self.agent_specs['e0']['v_rel_range0'] = self.agent_specs['e0'].get('v_rel_range',0.25)# range for uniform distribution
                
                # set initial velocity settings 
                self.agent_specs['e0']['v_rel_mean'] = float(self.agent_specs['e0']['v_rel_mean0'])
                self.agent_specs['e0']['v_rel_range'] = float(self.agent_specs['e0']['v_rel_range0'])
                
                # set initial velocity 
                self.agent_specs['e0']['v_rel'] = float(self.agent_specs['e0']['v_rel_mean']) # if no sampling, then pick mean
                # NOTE THAT self.agent_specs['e0']['v_rel'] IS USED THROUGHOUT SCRIPT AS |V|
                
            # following cannot be true for this config
            self.V_REL_CONTROL = False 
            self.V_REL_CONTROL_CL = False
            
            self.A_CONTROL_CL = False
            
        # velocity controls 
        elif self.SAcode_num in [34, 334]:
            if self.V_REL_CONTROL:
                # extract velocity settings 
                self.agent_specs['p0']['v_rel_max0'] = self.agent_specs['p0']\
                            .get('v_rel_max',self.agent_specs['p0']['v_rel'])
                            
                self.agent_specs['e0']['v_rel_max0'] = self.agent_specs['e0']\
                            .get('v_rel_max',self.agent_specs['e0']['v_rel'])
                            
                # set initial velocity settings 
                self.agent_specs['e0']['v_rel_max'] = float(self.agent_specs['e0']['v_rel_max0'])
                self.agent_specs['e0']['v_rel_max'] = float(self.agent_specs['e0']['v_rel_max0'])
                
                # set initial velocities 
                self.agent_specs['p0']['v_rel'] = float(self.agent_specs['p0']['v_rel_max'])
                self.agent_specs['e0']['v_rel'] = float(self.agent_specs['e0']['v_rel_max'])
                
                # following cannot be true for this config
                self.A_CONTROL_CL = False
                
            elif self.A_CONTROL_CL:
                # acceleration control subject to CL (w or w/o gravity)
                
                # set level 0 thrust ranges 
                self.agent_specs['e0']['a_T_range0'] = self.agent_specs['e0']['a_T_range'].copy()
                
                # set initial thrust ranges
                self.agent_specs['e0']['a_T_range'][0] *= self.a_e0_CL_factors[0]
                self.agent_specs['e0']['a_T_range'][1] *= self.a_e0_CL_factors[1]
                
                # following cannot be true for this config
                self.V_REL_CONTROL_CL = False

                
            # following cannot be true for this config
            self.V_REL_RANDOM = False 
            self.V_REL_RANDOM_CL = False
            
            
        ## SELFSPIEL SPECIFIC (related to single agent)
        if self.SELFSPIEL:
            # dont offer the following in case of selfspiel
            self.V_REL_RANDOM = False 
            self.V_REL_RANDOM_CL = False
            
            self.V_REL_CONTROL_CL = False
            
            self.A_CONTROL_CL = False
            
            self.DUMMY_EVADER_3D4DOF = False
            
            # set all agent velocities/acceleration capabilities the same
            self.agent_specs['e0'].update(self.agent_specs['p0'].copy())  # update with provided specs
        
        
        ## FINALIZE:
        self.thrust_orientation = np.array(self.thrust_orientation)
            
        return
        
    
#%% CURRICULUM LEARNING
    ## CURRICULUM LEARNING (CL) RELATED
    
    '''
    TODO IDEA IS TO HAVE A 2D CURRICULUM, WHERE WE SLOWLY BUILD
    TOWARDS A LARGE RANGE (THIS IS MONOTONIC) AND CONTROL EVOLUTION
    BY CHANGING VELOCITY SETTING (NOT MONOTONIC BUT RATHER DYNAMIC)
    '''
    def set_task_range(self):
        ''' 
        Range level as subject to task level 
        IDEA; start at [0.5, 1] and slowly build up to 
        a larger range []
        '''
        return 
    
    def set_task_velocity(self):
        '''
        IDEA; game of relativity
        
        if one agent fails at evolution we make it relatively
        faster than its adversary this should make it harder for the agent
        that is performing well as easier for the agent which is not
            
            - currently this is only possible for cases (i.e. DOF) with 
            constant velocity
        
        - f
        
        
        '''        
        
        return 
    
    
    #############
    def get_task(self, rtn_info = False):
        
        if not rtn_info:
            return self.CL_taskLevel
        else:
            info = {}
            return self.CL_taskLevel, info

    def set_task(self, taskLevel):
        ''' 
        == README ==
            NOTE THAT SET TASK IS CALLED AFTER ENV.RESET() AND
            THUS INITIALIZATION HAS ALREADY TAKEN PLACE
            (this is undesireable behaviour but we hack around it, see ***)
            
        = NOTE ON PPO (on-policy method) & CL =
        PPO is on-policy and do not have replay buffer (these are logically related), 
        therefore curriculum learning with different reward scales should not 
        be a problem. This is because samples from the previous sampling 
        stage (with a different reward scale) will not be mixed with sampling
        in the current stage and therefore will not be used in the PPO algorithm.
        
        = CL & Value function learning = 
        I do not know if CL with reward at different scales is good for learning 
        the value function, to me it seems as if this would have to constantly 
        adjust its scale (to limit large MSE errors) while the inherent task 
        (i.e. interception) remains constant. I think that this problem might 
        be emphasized if the critic has no estimate of distance. 
            To stablize training we might therefore need to give the distance
            measure to the value function, but not the policy; to make sense of
            this difference. Generally I do NOT think this would be bad in 
            general (giving critic more info on global state). Otherwise we 
            have to revise the scale.
        
        TODO;
        - CL based on rel_v
        - CL based on interception distance
        - consider IF MIN == MAX THEN THE AGENT CAN ALWAYS SOLVE THE ENV!
        ''' 
        
        self.CL_taskLevel = int(min(5,taskLevel))
        if self.verbose:
            print(f'CL taskLevel = {self.CL_taskLevel}')
        ## time
        t_idx_limit_prev = int(self.t_idx_limit) # requried for padding
        self.t_limit = float(self.t_limit0*(1+1.*self.CL_taskLevel))
        # increase timelimit
        self.t_idx_limit = int(np.ceil(self.t_limit/self.t_delta)) # +1 for initial state
        
        
        ## initialization range
        self.r0_e_range = self.r0_e_range0 *(1+1.*self.CL_taskLevel)
        # TODO make range bigger instead of only further
        if self.INTERCEPT_DISTANCE_CL:
            self.intercept_distance = self.intercept_distance0*2/(2+min(max(0,self.CL_taskLevel-0),5))
            # (2/3), (2/4), (2/5), (2/6), starting at CL = 1 (at max before V_rel, A_control starts)
            self.intercept_distance = max(0.05, self.intercept_distance)
        
        ## speed
        if self.SAcode_num in [21,32,33]:
            if self.V_REL_RANDOM and self.V_REL_RANDOM_CL: 
                # TODO CLEAN UP
                # we choose unfirom distribution as its bounded and should provide nice range of tasks
                factor = (1+0.5*max(0,self.CL_taskLevel-3)) # for mean = 0.5 og this caps out at level 5 and starts at 4
                self.agent_specs['e0']['v_rel_mean'] = \
                    min(1, self.agent_specs['e0']['v_rel_mean0']*factor) # clipped to max 1!
            
        if self.SAcode_num in [34,334]:
            if self.V_REL_CONTROL and self.V_REL_CONTROL_CL:
                # we choose unfirom distribution as its bounded and should provide nice range of tasks
                factor = (1+0.5*max(0,self.CL_taskLevel-3)) # for max = 0.5 og this caps out at level 5 and starts at 4
                self.agent_specs['e0']['v_rel_max'] = \
                    min(1, self.agent_specs['e0']['v_rel_max0']*factor) # clipped to max 1!
            
            elif self.A_CONTROL_CL:
                
                factor = (1+0.5*max(0,self.CL_taskLevel-3)) # for max = 0.5 og this caps out at level 5 and starts at 4
                
                a_e0_CL_factors = np.minimum(self.a_e0_CL_factors*factor, 1.) # clipped to max 1!
                
                self.agent_specs['e0']['a_T_range'][0] =\
                    a_e0_CL_factors[0]*self.agent_specs['e0']['a_T_range0'][0]
                self.agent_specs['e0']['a_T_range'][1] =\
                    a_e0_CL_factors[1]*self.agent_specs['e0']['a_T_range0'][1]

                
        ## reward
        self.reward_MaxStepScaler = float(1./self.t_idx_limit) # ensures dense reward remains consistent in scale
        # TODO REWARD PER TIMESTEP WILL CHANGE ACROSS LEVELS!
        self.reward_CL_taskLevel = 10. # + 10.*self.CL_taskLevel  # keep this stable for value func
        '''
        NOTE ON TERMINAL REWARD VS DENSE REWARD;
        
        CL increases reward based on initial interception distance; 
        Hence (hypothesis); since the reward for improved distance is invariant 
        to episode length, the system will move away from this dense reward in 
        favour of the larger sparse reward
        - Also since time might not scale as well, it might improve on time 
            as well, since epsidoes which dont reach intercept due to truncation
            these are bad ones
            -> TODO SEE IF WE CAN FURTHER ENCOURAGE/IMPROVE THIS BEHAVIOUR
            
        '''
        
        
        ## PAD STATE SPACE DUE TO NEW LENGTH
        ''' 
        ***
        NOTE THAT SET TASK IS CALLED AFTER ENV.RESET() AND
        THUS INITIALIZATION HAS ALREADY TAKEN PLACE + first obs computation
        (this is undesireable behaviour but we hack around it)
        
        THIS ALSO MEANS THAT THE FIRST ITERATION AFTER SET_TASK
        WILL STILL BE PERFORMED WITH THE SETTINGS OF THE LAST ONE E.G. r0
        '''
        t_limit_diff = max(int(self.t_idx_limit - t_idx_limit_prev),0)
        if hasattr(self, 'state_space'):
            # if reset() has not been called, state_space does not exist yet!
            # also t_limit_diff would be zero!
            self.state_space = np.pad(self.state_space, [(0,t_limit_diff),*((0,0),)*3] )
        
        return
        
#%% RESET FUNCTIONS
    

    def override_agentE_state(self, aid, state_aid, raise_z_coordinate = False, set_r0 = False):
        ''' 
        Calling this function cannot be undone
        ''' 
        #assert self.setting_eval, 'This function can only be called in eval setting'
        #assert self.resetted,'This function has to be called right before reset!'
        self.offline = True
        try:
            # remove agent from integration list
            self.agent_id_stepped.remove(aid) 
            if self.A_gain:
                self.A_gain.pop(aid, False)
        except ValueError:
            # agent already removed
            pass
        

        
        self.state_space_offline = state_aid.copy() # (T,D,3), with trailing xyz
        assert self.state_space_offline.shape[1:] == (3,3)
        if raise_z_coordinate:
            z_min = np.min(self.state_space_offline[:,0,2])
            self.state_space_offline[:,0,2] += (-z_min + 1)*(z_min < 0)
            # raise z coordinate to z_min = 1 in case negative
        
        if set_r0:
            xyz0 = self.state_space_offline[[0],0,:3].copy() # (1,3)
            xyz0_unit = xyz0/np.linalg.norm(xyz0)
            
            self.state_space_offline[:,0,:3] -= xyz0 # evader starts at the origin
            self.state_space_offline[:,0,:3] += xyz0_unit*set_r0
            
        ## override with env settings data characteristics
        self.t_idx_limit = len(self.state_space_offline)
        self.t_limit = self.t_idx_limit*self.t_delta # TODO INFER THIS FROM A TIMESTEP ARRAY
        
        ## override the initialization function
        self.initialize_state_space = self.initialize_offline_data
        
        return 
        
    def initialize_offline_data(self):
        
        assert not isinstance(self.state_space_offline, type(None)), 'offline data has to be defined'
        # override previously defined state_space
        self.state_space[:len(self.state_space_offline), 1, :,:3] = \
            self.state_space_offline # override agent e0 state
        
        ##
        if self.SAcode_num in [21,32,33,334]:
            # set velocity vector
            '''
            NOTE currently alignment with initial position is not conducted
            '''
            self.state_space[0,0,1,0] = 1. # align p0 velocity with x-axis   
            
            '''
            v_aid_unit = self.state_space[0,1,0,:3].copy()
            v_aid_unit /= np.linalg.norm(v_aid_unit)
            self.state_space[self.t_idx,0,1,:3] = v_aid_unit.copy() # velocity
            #'''
        elif self.SAcode_num in [34]:
            
            # HHHHHHHHHHHH
            aidx, aid = 0, 'p0' # p0
            aid_specs = self.agent_specs[aid].copy()
            '''
            ## DO NOT ALWAYS DO THIS
            v_orient = self.thrust_orientation.copy() # already unit vector
            v_aid_unit = self.state_space[0,1,0,:3].copy()
            v_aid_unit /= np.linalg.norm(v_aid_unit)
            
            angles_aid = Rotations.body2inertial_align(v_orient, v_aid_unit)

            
            RotMat_aid = Rotations.body2inertial(angles_aid)
            v_aid_new = RotMat_aid.apply(v_orient*1) 

            
            self.state_space[self.t_idx,aidx,1,:3] = v_aid_new.copy() # velocity
            self.state_space[self.t_idx,aidx,0,3:] = angles_aid # angle

            #'''
            
            
            #  P,Q,R & T states
            a_T_range_sum = sum(aid_specs['a_T_range'])
            pqr_lim = self.rate_limit
            self.state_space[self.t_idx,aidx,1,3:] = 0. # p,q,r
            self.state_space[self.t_idx,aidx,2,3] = 0.5*a_T_range_sum # 
            
            #''' 
            ## ALWAYS DO THIS
            # IMU states
            RotMat_BI = Rotations.body2inertial(self.state_space[self.t_idx,aidx,0,3:])
            RotMat_IB = Rotations.inertial2body(self.state_space[self.t_idx,aidx,0,3:])
            
            dx_B, dy_B, dz_B = RotMat_IB.apply(self.state_space[self.t_idx,aidx,1,:3]) # v
            a_g, drag_C = aid_specs['a_g'], aid_specs['drag_C']
            a_T = self.state_space[self.t_idx,aidx,2,3]
            
            F_B = [-drag_C[0]*dx_B - drag_C[3]*abs(dx_B)*dx_B       - drag_C[6]*(dx_B)/(self.t_delta*self.null_delay),  # Fx_B  ## ALEX
                   -drag_C[1]*dy_B - drag_C[4]*abs(dy_B)*dy_B       - drag_C[7]*(dy_B)/(self.t_delta*self.null_delay),  # Fy_B
                   -drag_C[2]*dz_B - drag_C[5]*abs(dz_B)*dz_B + a_T - drag_C[8]*(dz_B)*(1/(1.+abs(a_T)**(0.25)))/(self.t_delta*self.null_delay),   # Fz_B
                   ]
            self.state_space[self.t_idx,aidx,2,[0,1,2]] = RotMat_BI.apply(F_B) - [0.,0.,a_g] # acc
            
        ## 
        self.r0_e0 = np.linalg.norm(self.state_space_offline[0,0,:3])
        
        return 
    
    # TODO; COMBIEN WITH OTHER OFFLINE FUNC
    def override_agentBOTH_state(self, state_both, force = False):
        ''' 
        TODO COMBINE THIS WITH override_agentE_state; 
        
        THIS FUNCTION MEANS NO AGENTS AREE ACTIVE WE HAVEALL DATA
        WE INTEND TO USE THIS FUNCTION TO 'PUSH' THROUGH DATA AND GET
        CONSISTENT HISTORY SETS
        
        intent; ideally we call the override_agent_state multiple times
        and have a single initialization function for offline data
        which recognizes howmany agents have been overriden and 
        places the data.
        ''' 
        if not force:
            assert not self.offline, 'OVERRIDE HAS ALREADY OCCURED, THIS IS CURRENTLY NOT SUPPORTED -> REINITIALIZE THE ENV' 
            warnings.warn('ALL DATA IS NOW OFFLINE, ESSENTIALLY DETERMINISTIC & PASSIVE ENVIRONMENT')
        #assert self.setting_eval, 'This function can only be called in eval setting'
        #assert self.resetted,'This function has to be called right before reset!'
        self.offline = True

        # remove ALL agents from integration list
        self.agent_id_stepped = []
        self.A_gain = {}

        
        self.state_space_offline_BOTH = state_both.copy() # (T,D,3), with trailing xyz
        assert self.state_space_offline_BOTH.shape[1:] == (2,3,3)
        
        ## override with env settings data characteristics
        self.t_idx_limit = len(self.state_space_offline_BOTH)
        self.t_limit = self.t_idx_limit*self.t_delta # TODO INFER THIS FROM A TIMESTEP ARRAY
        
        ## override the initialization function
        self.initialize_state_space = self.initialize_offlineBOTH_data
        
        return 
        
    # TODO; COMBIEN WITH OTHER OFFLINE FUNC
    def initialize_offlineBOTH_data(self):
        
        assert not isinstance(self.state_space_offline_BOTH, type(None)), 'offline data has to be defined'
        # override previously defined state_space
        self.state_space[:len(self.state_space_offline_BOTH), :, :,:3] = \
            self.state_space_offline_BOTH # notice overriding of both agents (=dim 1)
        
        ## 
        self.r0_e0 = np.linalg.norm(self.state_space_offline_BOTH[0,1,0,:3]) # T = 0, agent E
        
        return 
    
        
    def reset_agents(self):
        pass 
    

    def reset_state_space(self):
        ''' 
        
        State space initialization or reset
        - State space is always 6 dimensional, with 3 derivatives (0th,1st & 2nd)
            regardless of the SA DOF and dimensions chosen. The irrelevant
            states will remain zero.
            - Note that setting to zero is not necessarily correct, e.g.
            constant velocity & rate control implies some acceleration in
            cartesian space, but we dont track it.
        - 

        
        State space format:
           [[x,y,z, phi, theta, psi], (ground states)
            [dx,dy,dz, dphi, dtheta, dpsi], (first derivative; v & rates)
            [ddx,ddy,ddz, ddphi, ddtheta, ddpsi]] (second derivative; a & torque)
    
        TODO
        - save state space progression in case of evaluation mode!
        - Consider using RoMA for rotation matrices
        - consider changing everything to tensors/bottleneck
        ''' 
        
        ## State space setup
        self.state_space = np.zeros((self.t_idx_limit+1,self.agent_n,3,6), 
                                    dtype=np.float64)  #(T,N,3,S)

        # Set initial evader position (random)
        self.initialize_state_space()
        
        ## finalize
        self.state_space = self.state_space.astype(dtype=np.float64)
        ''' 
        IDEA either define stationary or moving agent; 
            stationary has v = 0 
                moving has v!= 0 and then we can say its noisy movement around straight trajectory
        ''' 

        if self.setting_eval:
            # setup state container
            self.state_space_history = np.full(self.state_space.shape,
                                               np.nan, 
                                                dtype = np.float64) #(T,N,3,S), inferred
            '''
            Note on state_space_history vs state_space
            state_space is running and contains valid values (zero is empty)
            to ensure integration always works
            state_space_history is a copied version and contains nans until filled
            it is meant to be used for analysis.
            '''
            self.state_space_history[0,:,:,:] = self.state_space[0,:,:,:].copy()
        return self.state_space
    
    
    def seed_obseration_space(self, seed):
        for aid in self.observation_space:
            self.observation_space[aid].seed(seed)
        return 
        

    def seed_action_space(self, seed):
        for aid in self.action_space:
            self.action_space[aid].seed(seed)
        return 
 
    
    def reset(self, seed=None, options=None, setting_eval = False, ForceSeed = None):
        # Call super's `reset()` method to set the np_random with the value of `seed`.
        # Note: This call to super does NOT return anything.
        super().reset(seed=seed, options = options)
        self.time_reset = datetime.now()
        ## Initialize time
        self.t = 0.   
        self.t_idx = 0
        self.BS_idxLag = {aid: 0 for aid in self.agent_id} # BlindSight idx lag
        
        ## initialize miss mechanic
        self.miss_min_distance = 9999.
        self.miss_final_stage = False
        
        ##
        self.seed_unique_saved = seed # provided by ray (or not! = None)
        if self.setting_eval or setting_eval:
            self.setting_eval = True
            if self.verbose:
                print('Evaluation mode switched on!')
                print(self.SAcode)
            self.observation_space_history = {}
            self.action_space_history = {}
            self.reward_space_history = {}
            self.info_space_history = {}
            
            ## seeding
            '''
            Notes;
            - notice that seeding is only done in setting eval
            - Note that environments are not  
            
            TODO; also seed observation and actions spaces here?
            
            CONSIDER;
            if we only use seeding in eval setting (i.e. .evalaute())
            and that does not have multiple env's per worker, that means
            that the workers will be seeded correctly and that we dont
            have to worrry about multiple envs per worker
            (this is not as efficient as multiple env's per worker
             but does the job)
            '''
            
            # TODO; also seed observation and actions spaces here?
            if not isinstance(ForceSeed, type(None)): 
                # forcing seed
                self.seed_unique_saved = ForceSeed
                #self.set_seed(self.seed_unique_saved)
            else:
                # not forcing
                if not isinstance(self.seed_unique, type(None)):
                    # environment in worker
                    self.seed_unique_saved = int(self.seed_unique)
                    #self.set_seed(self.seed_unique_saved)
                    self.seed_unique += 1 # increment seed for next round
        # set the generators with selected seed
        self.set_seed(self.seed_unique_saved)

        ##
        if self.DO_NOISE:
            self.noise_dist = scs.multivariate_normal(self.noise_dist_mu,
                                                      self.noise_dist_sigma, # sig^2
                                                      #seed = # infers numpy seed
                                                      #allow_singular = True,
                                                      )
            # reset the rng for the noise distribution
            #https://stackoverflow.com/questions/16016959/scipy-stats-seed
            self.noise_dist.random_state = np.random.RandomState(self.seed_unique_saved)
            
        ## random velocity control
        if self.SAcode_num in [21,32,33]:
            if self.V_REL_RANDOM: 
                self.agent_specs['e0']['v_rel'] = np.random.uniform(
                                        max(self.agent_specs['e0']['v_rel_mean']-self.agent_specs['e0']['v_rel_range'],0),
                                        self.agent_specs['e0']['v_rel_mean']+self.agent_specs['e0']['v_rel_range'],
                                        1)[0]
        if self.SAcode_num in [34]:
            if self.V_REL_CONTROL:
                self.agent_specs['p0']['v_rel'] = float(self.agent_specs['p0']['v_rel_max'])
                self.agent_specs['e0']['v_rel'] = float(self.agent_specs['e0']['v_rel_max'])
        ##
        self.reset_state_space()
        self.min_distance = float(self.r0_e0) # initialize minimum distance, TODO MOVE?
        
        scores, outcomes = self.evaluate_state_space()
        REWARD_P, REWARD_E, reward_parts = self.determine_state_space_reward(scores, outcomes)
        # determine initial score
        
        ## Gather initial obs & info
        obs, infos, actions = {}, {}, {}
        for aid in self.agent_id:
            if 'p' in aid:
                
                obs[aid] = self.observation_pursuer(self.agent_specs[aid]['idx'], abs(1-self.agent_specs[aid]['idx']),
                                                    Tdelay_ego = self.agent_specs[aid]['Tdelay'],
                                                    Tdelay_tu = self.agent_specs[aid]['Tdelay'], 
                                                    )  # delays are the same initially
                # if you want to expand obs for every adversary then internal loop is needed
            elif 'e' in aid:
                obs[aid] = self.observation_evader(self.agent_specs[aid]['idx'], abs(1-self.agent_specs[aid]['idx']),
                                                    Tdelay_ego = self.agent_specs[aid]['Tdelay'],
                                                    Tdelay_tu = self.agent_specs[aid]['Tdelay'], 
                                                    )  # delays are the same initially
            else:
                raise Exception('Unknown agent')
                
            info_game = scores['game'].copy()
            if self.setting_eval:
                ## append identifier
                info_game['seed_unique'] = self.seed_unique_saved #(-1 as seed unique was already incremented for next run)
                
            infos[aid] = {'game':info_game, # we cannot define an info['game'] as ray thinks its an agent
                          'agent':{**scores[aid],
                                   **reward_parts[aid],
                              'a_total':0., # TODO replace with a_1DOF, a_2DOF
                          },
                      }
            if self.setting_eval:
                self.observation_space_history[aid] = \
                    np.full((self.t_idx_limit+1,self.agent_dims[aid][1]), 
                             np.nan,
                             np.float64) # (T,obs_size)
                self.observation_space_history[aid][0,:] = \
                    obs[aid].copy()
                            
                ##
                self.action_space_history[aid] = \
                    np.full((self.t_idx_limit+1,self.agent_dims[aid][2]), 
                             np.nan,
                             np.float64) # (T,action_size)
                self.action_space_history[aid][0,:] = 0.
                
                if aid in self.A_gain:
                    self.action_space_history[aid+'_gain'] = \
                        np.full((self.t_idx_limit+1,len(self.A_gain[aid])), 
                                 np.nan,
                                 np.float64) # (T,action_size)
                    self.action_space_history[aid+'_gain'][0,:] = 0.
                ##
                self.reward_space_history[aid] = \
                    np.full((self.t_idx_limit+1,1+len(reward_parts[aid])), np.nan,
                             np.float64) # (T,reward_size)
                self.reward_space_history[aid][0,:] = 0.
                ##
                # TODO; INFER! FROM INFO
                self.info_space_history[aid] = \
                    np.full((self.t_idx_limit+1,len(scores[self.agent_id[0]])+len(reward_parts[aid])+1), # +1 FOR ACTION_TOTAL TODO IMPROVE! 
                            np.nan,
                             np.float64) # (T,reward_size)
                self.info_space_history[aid][0,:] = list(infos[aid]['agent'].copy().values())
                    
        if self.setting_eval:
            ##
            self.reward_space_history['game'] = \
                np.full((self.t_idx_limit+1,1+len(reward_parts[aid])), np.nan,
                         np.float64) # (T,reward_size)
            self.reward_space_history['game'][0,:] = 0.
            ##
            self.info_space_history['game'] = \
                np.full((self.t_idx_limit+1,len(infos[self.agent_id[0]]['game'])), np.nan,
                         np.float64) # (T,info_size)
            self.info_space_history['game'][0,:] = list(infos[self.agent_id[0]]['game'].copy().values())
            
        ## finalize & return
        self.info_statics = {
            'env':{# meta info
                    'SAcode_num':self.SAcode_num,
                    'setting_eval':self.setting_eval,
                    'offline':self.offline,
                    'seed_unique':self.seed_unique_saved,
                    'remote':self.remote,
                    'worker_idx':self.worker_idx,
                    'vector_env_index':self.vector_env_index,
                    'num_workers':self.num_workers,
                    'time_initialized':self.time_initialized,
                    'time_reset':self.time_reset.strftime("%m/%d/%Y, %H:%M:%S"),
                    'time_end':None,
                    'time_episode':None, # time in seconds
                    ##
                    'DO_BS':self.DO_BS,
                    'BS_CL_ACTIVE':self.BS_CL_ACTIVE,
                    'BS_gamma_thres':self.BS_gamma_thres,
                    ##
                    'DO_NOISE':self.DO_NOISE,
                    'NOISE_PARAMS':self.noise_dist_params, 
                    ##
                    'V_REL_CONTROL':self.V_REL_CONTROL,
                    'V_REL_CONTROL_CL':self.V_REL_CONTROL_CL,
                    'V_REL_RANDOM':self.V_REL_RANDOM,
                    'V_REL_RANDOM_CL':self.V_REL_RANDOM_CL,
                    'A_CONTROL_CL':self.A_CONTROL_CL,
                    ##
                    'SELFSPIEL':self.SELFSPIEL,
                    ##
                    'INTERCEPT_DISTANCE_CL':self.INTERCEPT_DISTANCE_CL,
                    'CL_START_DIS_REWARD':self.CL_START_DIS_REWARD,
                    # TODO; SEPERATE EPISODE DICT?
                    #'episode_steps':, 'episode_T', ... 
                    
                    },
            'start':{# run settings
                    'CL':self.CL_taskLevel,
                    'r0':self.r0_e0, # TODO I think this should be in the agent dict! WITH FOR P0 R0 = 0
                    'cone0_e_range':self.cone0_e_range,
                    't_delta':self.t_delta,
                    't_limit':self.t_limit,
                    # relative velocity; in agent specs
                    # TODO NOISE &  value dims etc
                    },
            'agent':{aid: self.agent_specs[aid].copy() for aid in self.agent_id}, # TODO NOISE!
            'reward':{
                'escape_factor':self.escape_factor,
                'intercept_distance':self.intercept_distance,
            }, # todo save reward settings
            'end':{},
            }
        if self.offline:
            for aid in set(self.agent_id)-set(self.agent_id_stepped):
                # for agents offline i.e. w/o integration/update step
                self.info_statics['agent'][aid] = \
                    dict.fromkeys(self.info_statics['agent'][aid].keys(), np.nan) 
                # does not handle nested dicts, but its fine as a signal
                # override agent specs with nan (since offline)
            
        self.resetted = True
        self.reward_MaxStepScaler = float(1./self.t_idx_limit)
        return obs, infos

#%% GAIN action space implementation

    def Again_implement_gains(self, A_gain_aid, obs_aid):
        
        ## compute actions implied by gains
        
        gain_los, gain_dlos = A_gain_aid
        
        action_aid = [gain_los*obs_aid[0] \
                 + gain_dlos*obs_aid[1], # phi
                 gain_los*obs_aid[2] \
                 + gain_dlos*obs_aid[3], # theta
                 gain_los*obs_aid[4] \
                 + gain_dlos*obs_aid[5], # psi
                 ]
        
        return action_aid

    
    def Again_implement_GainGauge(self, A_gain_aid, obs_aid):
        
        ## unpack
        K1, K2 = A_gain_aid
        
        '''
        I, II, III = -np.array([1./np.sqrt(3)]*3, dtype = np.float64), \
            obs_aid[6:9], \
                obs_aid[9:12] # 
        '''
        I, II, III = obs_aid[6:9], \
            obs_aid[9:12], \
                obs_aid[12:15] # 
        #'''
        los, dlos = -1.*obs_aid[[0,2,4]], obs_aid[[1,3,5]] # notice minus!
        
        ## I,R,V selection;
        # K1 [0, 2pi-1/6*np.pi]
        #K1 = np.minimum(K1, 11/6*np.pi) # cutoff to remain uniqueness
        # setup weights (overlap, flows from ->I->R->V-> as a circle with peaks at k*2/3*pi)
        I_w = np.maximum(np.cos(K1), 0)
        II_w = np.maximum(np.sin(K1-(2/12*np.pi)), 0) # peak@2/3pi, halfV@pi, halfI@1/3pi
        III_w = -1.*np.minimum(np.sin(K1+(2/12*np.pi)), 0) # peak@4/3pi, halfR@pi, halfI@5/3pi
        
        # implement weights to get the scaler 'S'
        S = I_w*I + II_w*II + III_w*III # (3,); elementwise mulp
        dlos = -np.cross(II,S)
        
        ## los, dlos selection;
        # setup weights (no overlap; we want the system to choose!)
        mag = 10
        #''' 
        los_w = (np.minimum(K2,0)**2)*mag 
        dlos_w = (np.maximum(K2,0)**2)*mag
        ''' 
        los_w = np.minimum(K2,0)*mag 
        dlos_w = np.maximum(K2,0)*mag
        #'''
        # implement weights
        a_los = los_w * los + dlos_w * dlos # (3,); elementwise mulp
        
        ## gather & return
        #action_aid = S*a_los # (3,); elementwise mulp
        action_aid = a_los # (3,); elementwise mulp
        return action_aid 

#%% ENVIRONMENT STEP 

    def step(self, action_dict):
        self.resetted = False
        '''
        TODO CHECK CARTPOLE CONTINUOUS ANGLULAR RATE ACTION
        
        NOTHING BESIDES (PREPARING ACTION DICT) SHOULD BE DONE
        BEFORE UPDATING STATE!
        ''' 

        ## PREPARATION

        ## ACTION IMPLEMENTATION & TIME STEP PROGRESSION
        # (everything before this is at step t-1)
        # TODO improve efficiency; a lot of packing and unpacking
        # e.g. np.array(action_dict.values).flatten()?
        #u_p = action_dict["p0"][0]
        #u_e = 0. 
        #u_e = action_dict["e0"][0]
        '''
        # TODO incorproate & check
        # especially check the order
        u = []
        for aid in self.agent_id:
            try:
                u_aid = action_dict["e0"][0]
            except KeyError:
                u_aid = 0.
            u.append(u_aid)
        
        IDEALLY WE CHANGE THE ZOH FUNCTION TO BE DYNAMIC WRT TO 
        NUMBER OF AGENTS
        '''
        
        #u = [u_p, u_e]
        #print(f'action: {action_dict}')
        if self.OVERRIDE_ACTION_E0:
            A_gain_aid = action_dict.pop('e0').copy()
            A_gain_aid[0] += self.acc_g
            action_dict['e0'] = A_gain_aid
        
        
        if self.A_gain:
            A_gain_dict = {}
            for aid in self.A_gain:
                A_gain_aid = action_dict.pop(aid).copy() # remove from action dict
                A_gain_aid += self.A_gain[aid] # add bias
                
                action_aid = self.Again_implement_gains(A_gain_aid, self.obs_last[aid])[-self.dim_a:]
                #action_aid = self.Again_implement_GainGauge(A_gain_aid, self.obs_last[aid])[-self.dim_a:]
                '''
                print(A_gain_aid)
                print(action_aid)
                print(self.obs_last[aid])
                raise
                #''' 
                ## cache
                A_gain_dict[aid] = A_gain_aid # gains
                action_dict[aid] = action_aid # actual actions
        
        ## State space update & evaluation
        self.update_state(action_dict) # system solver, also updates timestep! 
        # requires 1d input 
        scores, outcomes = self.evaluate_state_space()
        
        
        ## Reward 
        REWARD_P, REWARD_E, reward_parts = self.determine_state_space_reward(scores, outcomes)
        
        ## SETUP OUTPUTS
        obs, rews, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        for aid in self.agent_id:
            
            ##
            aidx_ego = self.agent_specs[aid]['idx']
            aidx_tu = abs(1-aidx_ego)
            #
            Tdelay_ego = int(self.agent_specs[aid]['Tdelay']) #+ self.BS_idxLag[aid]
            Tdelay_tu = int(self.agent_specs[aid]['Tdelay']) + self.BS_idxLag[aid]*self.DO_BS*self.BS_active[aid] # if bs on, if active for agent & only then the actual lag
            ''' 
            TO CONSIDER
            - I think _ego might still should not be lagged! this would mean
                that you know you yourself are moving, but believe that
                your adversary is not  moving
                    - doing this implies that the observation does change though!
            - 
                
            ''' 
            ## agent specifics
            if 'p' in aid:
                ##
                obs[aid] = self.observation_pursuer(aidx_ego, aidx_tu,
                                                   Tdelay_ego = Tdelay_ego,
                                                   Tdelay_tu = Tdelay_tu,
                                                   ) # notice observation_pursuer
                # if you want to expand obs for every adversary then internal loop is needed
                rews[aid] = REWARD_P
                terminateds[aid] = outcomes['I']
            elif 'e' in aid:
                ##
                obs[aid] = self.observation_evader(aidx_ego, aidx_tu,
                                                   Tdelay_ego = Tdelay_ego,
                                                   Tdelay_tu = Tdelay_tu,
                                                   ) # notice observation_evader
                # if you want to expand obs for every adversary then internal loop is needed
                rews[aid] = REWARD_E
                terminateds[aid] = outcomes['E']
            else:
                raise Exception('Unknown agent')

            truncateds[aid] = outcomes['T'] or outcomes['M'] # exceeded time limit or missed
            
            info_game = scores['game'].copy()
            if self.setting_eval:
                ## append identifier
                info_game['seed_unique'] = self.seed_unique_saved #(-1 as seed unique was already incremented for next run)
                
            infos[aid] = {'game':info_game, # we cannot define an info['game'] as ray thinks its an agent
                          'agent':{**scores[aid],
                                   **reward_parts[aid],
                                   'a_total':np.sum(np.abs(action_dict[aid])), # TODO replace with a_1DOF, a_2DOF
                          },
                      }

                
            # TODO; INFOS FUNCTION AND SAVE FIRST ONE (RESET) TO INFO_SPACE
            # ONLY SAVE SETTINGS E.G. BS IF ITS ON
            '''
            Info can contain distance, CCL divergence, etc.
            
            TODO MAKE ON INFOS['GENERAL'] WITH NON-AGENT SPECIFIC INFO?
            '''
            
            if self.setting_eval:
                self.observation_space_history[aid][self.t_idx, :] = obs[aid]
                self.reward_space_history[aid][self.t_idx, :] = [rews[aid]] + list(reward_parts[aid].values())
                self.action_space_history[aid][self.t_idx, :] = action_dict[aid]
                self.info_space_history[aid][self.t_idx, :] = list(infos[aid]['agent'].values())
                
                if aid in self.A_gain:
                    self.action_space_history[aid+'_gain'][self.t_idx, :] = A_gain_dict[aid]
                #self.action_space_history[aid][self.t_idx, :]= np.append(action_dict[aid],[0.]) ## TODO EXTRA_ACTION_CODE
                    
        ## game state & info
        terminateds['__all__'] = any(terminateds.values()) # notice ANY not ALL
        truncateds['__all__'] = any(truncateds.values())
        # CURRENTLY ANY TERM/TRUNC MEANS EVERYTHING STOPS
        
        #terminateds['__any__'] = any(terminateds.values())
        #truncateds['__any__'] = any(truncateds.values())
        # these cannot be added because ray thinks this is additional agent
        
        if self.setting_eval:
            # save current state & game info
            self.state_space_history[self.t_idx,:,:,:] = \
                self.state_space[self.t_idx,:,:,:].copy() # fill in current state
            self.reward_space_history['game'][self.t_idx, :] = \
                self.reward_space_history['p0'][self.t_idx, :] + self.reward_space_history['e0'][self.t_idx, :]
            self.info_space_history['game'][self.t_idx, :] = list(infos[self.agent_id[0]]['game'].values())
            
            
            if (terminateds['__all__'] or truncateds['__all__']):
                # episode done: cutoff remaining timesteps
                self.state_space_history = self.state_space_history[:(self.t_idx+1),:,:,:]
                for aid in self.agent_id:
                    self.observation_space_history[aid] = self.observation_space_history[aid][:(self.t_idx+1), :]
                    self.reward_space_history[aid] = self.reward_space_history[aid][:(self.t_idx+1), :]
                    self.action_space_history[aid] = self.action_space_history[aid][:(self.t_idx+1), :]
                    self.info_space_history[aid] = self.info_space_history[aid][:(self.t_idx+1), :]
                    
                    if aid in self.A_gain:
                        self.action_space_history[aid+'_gain'] = self.action_space_history[aid+'_gain'][:(self.t_idx+1), :]
                
                self.reward_space_history['game'] = self.reward_space_history['game'][:(self.t_idx+1), :]    
                self.info_space_history['game'] = self.info_space_history['game'][:(self.t_idx+1), :]    
                # TODO ALSO CUTOFF t = 0? as it was the initial state?
                    
                ## update statics (game outcome)
                self.info_statics['end'] = outcomes 
                # meta info
                time_end = datetime.now()
                self.info_statics['env']['time_end'] = time_end.strftime("%m/%d/%Y, %H:%M:%S") 
                self.info_statics['env']['time_episode'] = (time_end-self.time_reset).total_seconds()

                n_outcomes = sum(self.info_statics['end'].values())
                assert n_outcomes == 1, f'Multiple or no game outcome(s) encountered! (N={n_outcomes})'
                ## update all history list
                self.history_all.append(self.get_histories())
        ## GATHER AND RETURN
        return obs, rews, terminateds, truncateds, infos
    
#%% SCORING & TERMINATING STATE SPACE

    def evaluate_state_space(self):
        
        scores = self.score_state_space()
        outcomes = self.determine_state_space_outcome(scores['game'])
        return scores, outcomes
    
    
    def score_state_space(self):
        '''
        Function that scores the current state space by setting up game and
        agent-specific metrics.
        '''
        ##
        #if self.dim_s == 2: 
        #    raise Exception('I DONT BELIEVE THIS WORKS AS MATRIXS ARE FLIPPED')
        
        scores = {}
        ## game scores
        scores['game'] = self.score_game_state(self.t_idx)
        BS_bools = self.update_BlindSight_dynamics(scores['game'])
        
        for aid in self.agent_id:
            aidx_ego = self.agent_specs[aid]['idx']
            aidx_tu = abs(1-aidx_ego)
            
            ## agent-specific scores
            scores[aid] = self.score_agent_state(self.t_idx, aidx_ego, aidx_tu)
            ## add additional scores
            if self.DO_BS:
                # TODO IMPROVE THIS?
                scores_extra = {
                    'BS_lag':self.BS_idxLag[aid],
                    'BS_bool': BS_bools[aid],
                    'BS_active':self.BS_active[aid],
                    } # todo move these extras to the score_agent_state func?
                
                scores[aid] = {**scores[aid], **scores_extra}    
                
        ##
        
        return scores
    
    def score_game_state(self, t_idx):
        '''
        Scores the game state at a specific time step (t_idx)
        '''
        ## State variables
        state_xyz = self.state_space[t_idx,:,:,:3].copy()
        delta = state_xyz[1,:,:3]-state_xyz[0,:,:3] # (3,3), (D,xyz)
        # fixed agent idxes
        
        delta_norm = np.linalg.norm(delta, axis = -1, keepdims=False) # (D,)
        
        v_normP = np.linalg.norm(state_xyz[0,1,:3])
        v_normE = np.linalg.norm(state_xyz[1,1,:3])
        v_uV = state_xyz[1,1,:3]/(v_normE+1e-15) \
            - state_xyz[0,1,:3]/(v_normP+1e-15) # unit-velocity diff
        v_uV_norm = np.linalg.norm(v_uV) #  norm of unit-velocity diff, ranges in [0,2]
        v_uV_unit = v_uV/(v_uV_norm+1e-12)
        
        ## Extra scores
        gamma = np.dot(delta[0,:],delta[1,:])\
            /(delta_norm[[0,1]].prod() + 1e-15) # CATD == -1! conventional gamma
        
        gamma_uV = np.dot(delta[0,:]/(delta_norm[0]+1e-15),v_uV_unit) # unit vector gamma
           
        tprev_idx = max(0,(self.t_idx-1))
        dis = delta_norm[0]
            
        dis_prev = np.linalg.norm(self.state_space[tprev_idx,1,0,:3] \
                                    - self.state_space[tprev_idx,0,0,:3])
        
        disDelta = (dis - dis_prev)
        disDeltaPos = int(disDelta > 0)
        disNearMiss = int(dis < (3*self.intercept_distance))
        ## Misc scores
        dur = float(self.t_idx/self.t_idx_limit)
        
        ## Gather and return
        #''' 
        # NOT WORKING!
        self.min_distance = min(self.min_distance, dis)
        #''' 
        # update min distance, if off, then min = self.r0
        
        game_score = {
            'Gamma':gamma,
            'Gamma_uV':gamma_uV,
            'dur':dur,
            'dis':dis,
            'disDelta':disDelta,
            'disDeltaPos':disDeltaPos,
            'disNearMiss':disNearMiss,
            #'dis/dis0':dis/self.r0_e0,
            'Vratio':v_normP/(v_normE+1e-4), 
            ##
            }
        return game_score
       
        
    def score_agent_state(self, t_idx, aidx_ego, aidx_tu):
        '''
        Scores an agent's state at a specific time step (t_idx)
        '''
        ## State variables (inertial frame!)
        state_xyz = self.state_space[t_idx,:,:,:].copy()
        state_ego = state_xyz[aidx_ego,:,:]
        delta = state_xyz[aidx_tu,:,:3]-state_xyz[aidx_ego,:,:3] # (3,3), (D,xyz)
        # ego-centered coordinates (not frame!)
        delta_norm = np.linalg.norm(delta, axis = -1, keepdims=False)

        rotmat_BI = Rotations.body2inertial(state_ego[0,3:])
        
        
        ## Misc scores
        v_norm = np.linalg.norm(state_ego[1,:3])
        v_downward = -1*np.dot(state_ego[1,:3]/(v_norm+1e-8),[0.,0.,-1.])# 
        
        z_downward = -1*np.dot(
                            rotmat_BI.apply([0.,0.,-1.]), # body to inertial
                            [0.,0.,-1.], # inertial
                        )# 
        # -1* because easier to understand
            
        ## Extra scores
        gamma_pp = np.dot(delta[0,:], state_ego[1,:3])/\
            (delta_norm[0]*v_norm+1e-15)
            
        thrust_I = rotmat_BI.apply(self.thrust_orientation.copy())    
        thrust_pp = np.dot(delta[0,:], thrust_I)/(delta_norm[0]+1e-15)    
        
        
        tprev_idx = max(0,(self.t_idx-1))
        # dis_t is already determined and trimpoint
        
        dis_tprev_aid = np.linalg.norm(self.state_space[self.t_idx,aidx_tu,0,:3] \
                                    - self.state_space[tprev_idx,aidx_ego,0,:3]) # current tu, previous ego position
        dis_delta_aid = delta_norm[0] - dis_tprev_aid # agent specific change in distance 
        dis_deltaPos_aid =  int(dis_delta_aid  > 0)  # positive  
        
        ## Gather and return
        agent_score = {
            'Gamma_pp':gamma_pp,
            'thrust_pp':thrust_pp,
            ##
            'stable_Vdownward':v_downward,
            'stable_Zdownward':z_downward,
            
            'Vnorm':v_norm,
            'disDelta':dis_delta_aid,
            'disDeltaPos':dis_deltaPos_aid,
            
            }
        return agent_score
       
    def determine_state_space_outcome(self, game_score: dict):
        
        dis = game_score['dis']
        
        truncated = (self.t_idx >= (self.t_idx_limit-1))
        
        missed = False
        DO_MISS = False # to config
        if DO_MISS:
            raise Exception('RECONSIDER THIS DYNAMIC')
            if dis < self.intercept_distance*2:
                # enter final pursuit stage
                self.miss_final_stage = True
                self.miss_min_distance = min(dis, self.miss_min_distance) # recursively 
                # TODO THIS REQUIRES ANOTHER VARIABLE TO DETERMINE IF WE ARE IN THE FINAL STAGE
            missed = ((self.miss_min_distance*1.5) < dis) and self.miss_final_stage 
            # final stage entered and distance increases over threshold
            '''
            TODO POTENTIALLY RELATE THIS TO THE RELATIVE VELOCITY AND/OR
            TURNING SPEED, I.E. IF YOU CANNOT TURN QUICKLY ENOUGH AND 
            ACHIEVE INTERCEPTION THEN ITS A MISS
            '''
  
        intercepted = (dis < self.intercept_distance)
        #escaped = (dis > (self.r0_e0*self.escape_factor)) # if evader reaches c times initial distance, hes escaped
        escaped = (dis > (self.min_distance*self.escape_factor)) # dynamic variant of previous

        '''
        TODO POTENTIALLY RELATE INTERCEPTION/ESCAPE TO THE RELATIVE VELOCITY?
        '''
        ##
        outcomes = {
            'I':intercepted,
            'E':escaped,
            'M':missed,
            'T':(truncated and not (intercepted or escaped or missed)),
            }
        game_score['INToutcome'] = int(outcomes['I']) # track number of intercepts
        return outcomes
    
    
    def update_BlindSight_dynamics(self, game_score: dict):
        '''
        DESCRIPTION
        
        TODO;
        -  CHECK IF GAMMA DOES NOT START AT -1, 
            - no it starts at PP which can align somewhat though based on 
            adversary attitude 
        - TODO BLINDSIGHTING CAN ALSO BE DONE BASED ON ANGLE/CCL DIVERGENCE,
            - This is nice THEN ITS RANGE DEPENDENT!
        - TODO IF BS; THEN ONLY PROVIDE ERROR ANGLES, AND NO VELOCITY RELATED 
            INFO I.E. NO OPTIC FLOW AVAILABLE
        
        Parameters
        ----------
        game_score : dict
            dictionary containing 'Gamma' score. Note that the game
            score provides governs at which timestep we do this

        Returns
        -------
        BS_bool : dict 
            dictionary containing whether agents are blindsighted (=1) or
            not (=0).


        = PSUEDO CODE =
        
        ...
        for agent in agents:
            r, v, a = s_{agent}
            
            if agent is blindsighted:
                 v, a = 0. # mask derivative states
                 r_prev, _ , _ = s_{history}[t_noBs] # get last known adversary position
                 # t_noBs: last timestep index not blind-sighted
                 
                 r = r_prev # override current position with historical one 
                                  
                 obs_{agent} = (r,v,a)
            
        '''
        BS_bool = {aid: 0 for aid in self.agent_id} 
        #if False:
        #if self.DO_BS and (self.CL_taskLevel >= self.BS_CL_ACTIVE or self.setting_eval):
        #if self.DO_BS and (self.CL_taskLevel >= self.BS_CL_ACTIVE):
        self.BS_active = {'e0':False,'p0':False} # not active for either agent
        if self.DO_BS:
            if (self.CL_taskLevel >= self.BS_CL_ACTIVE):
                self.BS_active = {'e0':True,'p0':False} # active for agent is condition is met
            # DO_BS should be on and CL level high enough (unless setting_eval = forced)
            #raise Exception('HAVE YOU CHECKED THIS FUNCION')
            gamma = game_score['Gamma']
            if (gamma < (-1*self.BS_gamma_thres)):
                # irrotational CCL and converging distance
                self.BS_idxLag['e0'] += 1 # evader blindsighted
                BS_bool['e0'] = 1
            else:
                self.BS_idxLag['e0'] = 0 # evader blindsight removed
                
            #'''
            if (gamma > self.BS_gamma_thres):
                # irrotational CCL and diverging distance
                self.BS_idxLag['p0'] += 1 # pursuer blindsighted
                BS_bool['p0'] = 1
            else:
                self.BS_idxLag['p0'] = 0 # pursuer blindsight removed
            #'''
            # NOTE YOU CANNOT DO this with BS_idxLag[aid] because its cross-agent
        return BS_bool
    
    
#%% UTILITIES 
    def get_next_input(self, timesteps_rnn= None, torch_out = True):
        '''
        Compute the input for a (RNN) model according to the desired dimension
        format i.e. (T,F), where T is defined at initialization for a 
        pursuer specifically
        
        
        NOTE; this function is purposefully named input_rnn & not obs_rnn
        as previous rewards, actions etc might be available as inputs as well,
        yet these are not considered observations.
        '''
        assert self.setting_eval, 'Function only useable in eval setting'
        input_rnn = {}
        for aid in self.agent_id:
            input_rnn[aid] = {}
            # get time indices
            if timesteps_rnn is not None:
                t_rnn = timesteps_rnn
            else:
                t_rnn = self.agent_dims[aid][3] # use default
            t_range_obs = list(range(max(0,self.t_idx+1-t_rnn),self.t_idx+1))
            if len(t_range_obs) < t_rnn:
                pad_len = t_rnn-len(t_range_obs)
                t_range_obs = np.pad(t_range_obs, (pad_len,0), 
                                 mode = 'constant')
                
            t_range_action = list(np.maximum((np.array(t_range_obs.copy())-1), 0)) # shift one back 
            
            # setup (T,F) format  
                       
            prev_obs_aid = \
                self.observation_space_history[aid][t_range_obs,:] # out (T,F), not [np.newaxis,t_range,:]
            obs_aid = prev_obs_aid[-1,:].copy()
            prev_actions_aid = \
                self.action_space_history[aid][t_range_action,:] # out (T,F), not [np.newaxis,t_range,:]
            # TODO MAYBE ADD PREVIOUS(OWN) REWARDS/ACTIONS
            if torch_out:
               obs_aid = torch.tensor(obs_aid, dtype = torch.float32)
               prev_obs_aid = torch.tensor(prev_obs_aid, dtype = torch.float32)#.unsqueeze(dim = 0)
               prev_actions_aid = torch.tensor(prev_actions_aid, dtype = torch.float32)#.unsqueeze(dim = 0)
               
               '''
               prev_obs_aid = prev_obs_aid.cuda()
               prev_actions_aid = prev_actions_aid.cuda()
               #'''
               # note that device = CPU 
               
            ##
            input_rnn[aid]['obs'] = obs_aid
            input_rnn[aid]['prev_n_obs'] = prev_obs_aid
            input_rnn[aid]['prev_actions'] = prev_actions_aid
            
        return input_rnn 
    
        
    def render(self, mode = 'human'):
        '''
        TODO UPDATE OR REMOVE
        '''
        pursuer_color = np.array([1.,0.,0.])
        evader_color = np.array([0.,0.,1.])
        
        grid_size = 60
        grid = np.ones((grid_size,grid_size,3))

        pursuer_pos = int(grid_size/2)+self.state_space[self.t_idx,0,0,:2].astype(np.int32)    
        evader_pos = int(grid_size/2)+self.state_space[self.t_idx,1,0,:2].astype(np.int32)
        
        grid[pursuer_pos[0],pursuer_pos[1],:] = pursuer_color
        grid[evader_pos[0],evader_pos[1],:] = evader_color
        #grid *= 255. # rgb image
        return grid

    def get_histories(self):
        assert self.setting_eval, 'Function only useable in eval setting'
        if self.verbose:
            print(f'Histories up to {self.t_idx}/{self.t_idx_limit} e.g. act shape {self.action_space_history[self.agent_id[0]].shape}\n(note dim T=t_idx_limit+1 due to initial state)')
        return self.state_space_history, \
                    self.observation_space_history, \
                        self.reward_space_history, \
                            self.action_space_history, \
                                self.info_space_history, \
                                    self.info_statics # s, o, r, a, i, statics

    def getNwipe_histories_all(self):
        history_out = self.history_all.copy()
        self.history_all = [] # wipe
        
        ## reset seed
        #self.seed_unique = int(self.seed_unique0)
        return history_out
    
#%% INITIALIZATIONS

    def initialize_state_space_2D(self):
        '''
        Initialize state space for 2 dimensional case, regardless of the DOF
        
        Evader starts at a random location within the radius [r_e_min,r_e_max] 
        from the origin. In addition, he starts with certain heading (psi_e) 
        within range (-psi_e_max, psi_e_max), where drawn psi_e is
        subsequently adjusted based on the psi_p 
        
        NOTE the reason for this range & adjustment for psi is to ensure
        that the evader is not moving directly towards (or close to) the origin
        -> imagine two cones >< which are tangent to a circle centered around
            the origin (obtained by rotating >< by an angle) 
        
        TODO note that having (psi_e_min, psi_e_max) = (~,0.5)*np.pi
            means pursuer will always have some sort of heading towards pursuer
            albeit not directly
        '''
        idx_p0, idx_e0 = self.agent_specs['p0']['idx'], self.agent_specs['e0']['idx']
        # Set initial evader position (random)
        #self.state_space[self.t_idx,1,0,:2] =  np.random.uniform(low =np.array([[-5.,5.]]), 
        #                                high=np.array([[5.,10.]]), size = (1,2))
        #r0_e_range[0], r0_e_range[1], psi0_e_min, psi0_e_max = 3., 6., 0.25*np.pi, 0.75*np.pi # ensure > 0 all
        r0_e, angle0_e, psi0_e =  np.random.uniform(low =
                                        np.array([self.r0_e_range[0],-np.pi, 0.5*np.pi-self.cone0_e_range]), 
                                        high=
                                        np.array([self.r0_e_range[1],np.pi, 0.5*np.pi+self.cone0_e_range]),
                                        size = (3,))
        # cone definition for consistency with polar cone setup in 3D
        self.r0_e0 = r0_e # cache initial distance for evader e0, TODO MAKE DYNAMIC FOR EVADERS
        psi0_e *= np.random.choice([-1,1]) # allow range (-psi_e_max, -psi_e_max) as well
        x0_e, y0_e = r0_e*np.cos(angle0_e), r0_e*np.sin(angle0_e) 
        
        
        # initial heading pursuer aligns with initial position
        #psi_p = 0. # along x-axis
        psi0_p = np.arctan2(y0_e, x0_e) # implies zero relative heading evader (ADD NOISE AFTER ***)
        # !be aware that if psi_p = np.arctan2(y_e, x_e) & rel_velo = 0, the controller will have to do nothing!
        
        # adjust psi_e to ensure movement away from the origin
        psi0_e += psi0_p # ccw rotation applied (***)
        
        # clip angles
        psi0_e = ((psi0_e +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        psi0_p = ((psi0_p +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        
        # save sampled initial ground states
        self.state_space[self.t_idx,idx_p0,0,5] = psi0_p # update pursuer state
        self.state_space[self.t_idx,idx_e0,0,[0,1,5]] = np.array([x0_e, y0_e, psi0_e])  # update evader state
        
        # Set initial evader position (random)

        # initial velocities consistent with initial \psi settings
        e_v_rel = self.agent_specs['e0']['v_rel']
        self.state_space[self.t_idx,idx_p0,1,:2] = [np.cos(psi0_p), np.sin(psi0_p)]
        self.state_space[self.t_idx,idx_e0,1,:2] = [np.cos(psi0_e)*e_v_rel,np.sin(psi0_e)*e_v_rel]
        # body to inertial SEE self.ANGLES_SIGN
        
        return 
    
    def initialize_state_space_3D(self):
        '''
        initializes state space for 3D configurations
        
        steps taken:
        1. sample location for evader
        2. initialize orientation of pursuer to align (velocity vec) with
            initial position of evader (also means we infer the orientation)
        3. initialize orientation evader (indepedent from pursuer) by 
            randomly sampling a trajectory in the 'doublesided cone'
            which does not move too directly towards/away from pursuer. This
            can be visualized (in 2d) as: 
                >e<          (>< indicates heading)
                 \ <- = r0_e (pursuer heading aligned with evaders ini pos)
                 p           (pursuer starts at origin always) 

        '''
        idx_p0, idx_e0 = self.agent_specs['p0']['idx'], self.agent_specs['e0']['idx']
        ## P1: initialize evader position
        # sample initialization parameters
        r0_e, angleP0_e, angleA0_e =  np.random.uniform(low =
                                        np.array([self.r0_e_range[0],0.25*np.pi, 0.*np.pi]), 
                                        high=
                                        np.array([self.r0_e_range[1],0.75*np.pi,2.*np.pi]), 
                                        size = (3,)) 
        # initial; radius/distance, polar-, azimuth-angle and initial heading
        self.r0_e0 = r0_e 
        
        ## initialize position
        # evader initial position randomly sampled
        self.state_space[self.t_idx,idx_e0,0,[0,1,2]] = [r0_e*np.sin(angleP0_e)*np.cos(angleA0_e), 
                                         r0_e*np.sin(angleP0_e)*np.sin(angleA0_e),
                                         r0_e*np.cos(angleP0_e),
                                         ] # polar/spherical coordinates, physics notation 
        
        # pursuer always starts at origin
        
        
        ## P2: initialize pursuer orientation
        # pursuer orientation
        # align pursuer's x-axis with the initial starting point of the evader
        v_p = self.state_space[self.t_idx,idx_e0,0,[0,1,2]].copy()
        v_p /= np.linalg.norm(v_p)

        # infer attitude from inertial frame attitude (v_p is in inertial frame!)
        psi_p = np.arctan2(-v_p[1], np.sign(v_p[0])*np.sqrt(max(0,1-v_p[1]**2)))
        theta_p = np.arctan2(np.sign(v_p[0])*v_p[2], np.sqrt(max(0,(1-v_p[1]**2)-v_p[2]**2)))
        
        # clip angles
        psi_p = ((psi_p +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        #theta_p = ((theta_p +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        theta_p = max(min(theta_p,0.5*np.pi),-0.5*np.pi) 
        # ^ this is possible since notice that atan2 only has positive x!
         
        RotMat_p = Rotations.body2inertial([0., theta_p, psi_p])
        
        p_v_rel = self.agent_specs['p0']['v_rel']
        self.state_space[self.t_idx,idx_p0,1,[0,1,2]] = RotMat_p.apply([p_v_rel, 0., 0.]) # constant velocity 
        self.state_space[self.t_idx,idx_p0,0,[3,4,5]] = [0., theta_p,psi_p] # angles
        
        ## P3: initialize evader orientation
        # random heading sampled and then used to translate velocity vector
        angleP1_e, angleA1_e =  np.random.uniform(low =
                                        np.array([0.*np.pi,    0.*np.pi]), 
                                        high=
                                        np.array([self.cone0_e_range,   2.0*np.pi]), 
                                        size = (2,))# polar coordinates
        # notice cone open at the top (=z) 
        # angleP1_e already has correct sign 

        # define velocity vector (in body frame)
        e_v_rel = self.agent_specs['e0']['v_rel']
        v_e = np.array([e_v_rel*np.sin(angleP1_e)*np.cos(angleA1_e), 
                e_v_rel*np.sin(angleP1_e)*np.sin(angleA1_e),
                e_v_rel*np.cos(angleP1_e)*np.random.choice([-1,1]), # -v_z/v_z, also get negative z heading
                ]) # polar/spherical coordinates, physics notation in BODY frame
        # turn the cone 90 to be open ended in yz (instead of xy plane i.e. top)
        RotMat_x = Rotations.body2inertial([0.5*np.pi-angleA0_e, 0., 0.])
        # theta; xy -> xz | phi; xz -> yz  (+/-) doesnt matter
        v_e = RotMat_x.apply(v_e)
        
        ## apply rotations to evader velocity and cache states
        # from body to inertial
        RotMat_e = Rotations.body2inertial([0, theta_p, psi_p]) 
        # body frame is along the origin-ini_pos_e vector which we aligned with pursuer heading 
        v_e_I = RotMat_e.apply(v_e)
        self.state_space[self.t_idx,idx_e0,1,[0,1,2]] = v_e_I # constant velocity 
        
        # infer attitude from inertial frame velocity vector
        psi_e = np.arctan2(-v_e_I[1], np.sign(v_e_I[0])*np.sqrt(max(0,1-v_e_I[1]**2)))
        theta_e = np.arctan2(np.sign(v_e_I[0])*v_e_I[2], np.sqrt(max(0,(1-v_e_I[1]**2)-v_e_I[2]**2)))
        
        # clip angles
        psi_e = ((psi_e +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        #theta_e = ((theta_e +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        theta_e = max(min(theta_e,0.5*np.pi),-0.5*np.pi) 
        # ^ this is possible since notice that atan2 only has positive x!
        
        self.state_space[self.t_idx,idx_e0,0,[3,4,5]] = [0.,theta_e,psi_e] # angles
        self.state_space[self.t_idx,...] = np.nan_to_num(self.state_space[self.t_idx,...],
                                                         nan=0.,
                                                         neginf=0.,
                                                         posinf=0.)
        '''
        # align with x-axis
        self.state_space[self.t_idx,1,1,[0,1,2]] = [e_v_rel, 0., 0.] # constant velocity
        #self.state_space[self.t_idx,1,0,[3,4,5]] = [0, 0., 0.] # angles
        '''
        
        return 


    def initialize_state_space_3D4DOF(self):
        ''' 
        Rationale: for 4DOF we need to be aware that initialization considers 
        all relevant states and circumstances. Compared to other DOF cases 
        (non-acceleration) especially the initialization velocity is important!
        
        Note that in the acceleration cases the orientation (i.e. euler angles)
        has been disconnected from velocity vector (unlike e.g. 2D1DOF, 
        3D2/3DOF), thus we can initialize it far more randomly than in cases 
        where there is a hard constraint/relationship between velocity & euler 
        orientation of the body frame!
        
        Given the rationale outlined aboved, we acknowlegde that
        initialize_state_space_3D() has initialized the following:
        1. evader position, attitude* and velocity (based on e_v_rel)
        2. pursuer attitude* and velocity (p_v_rel = 1, always)
            (note not position as == origin)
        * both attudes only consider theta and psi (no phi) though!
    
        Consequently, we also make the follow adjustments to ensure all states
        are properly initialized. Adjustment steps;
        1. introduce phi orientation; ~N(0,1) truncated
        2. Introduce random velocity vector with magnitude [0,max_speed] 
        
        * wipe z-velocity compoentn for both agents (since it wont decay for null input!)
            (Specifically, we do this to be able to use the drag model)
        * note that all rng's are controlled by the set seed functionality in
            the .reset() function
        '''
        self.initialize_state_space_3D()


        #phi = [0.,0.]#((np.random.randn(2) +np.pi) % (2.*np.pi)) - np.pi # sample & clip
        
        '''
        v_unit = np.random.uniform(-1,1,(2,3))
        v_unit /= np.linalg.norm(v_unit) 
        
        bounds =  np.array([1, 0.5, 1])
        angles = np.random.uniform(-1*bounds*np.pi,
                                   bounds*np.pi,(2,3))
        '''
        if self.A_CONTROL_3D3DOF:
            # A_CONTROL_3D3DOF BODY FRAME
            self.state_space[self.t_idx,:,0,3:] = 0. # angle
            self.state_space[self.t_idx,:,2,:3] = self.state_space[self.t_idx,:,1,:3].copy() # acc
            
        elif not self.V_REL_CONTROL: # notice ELIF!
            # applies for all other 3D4DOF cases & A_CONTROL_3D3DOF
            
            #'''
            if not self.A_CONTROL_noG:
                # no negative Z-coordinate
                self.state_space[self.t_idx,1,0,2] = abs(self.state_space[self.t_idx,1,0,2]) #+ 1. # z
                self.r0_e0 = np.linalg.norm(self.state_space[self.t_idx,1,0,:3])
                self.state_space[self.t_idx,0,1,:3] = \
                    (self.state_space[self.t_idx,1,0,:3].copy()/self.r0_e0)*self.agent_specs['p0']['v_rel'] # v_p
            #'''
            
            for aid in self.agent_id:
                aidx = self.agent_specs[aid]['idx']
                aid_specs = self.agent_specs[aid].copy()
                ## P1
                
                #self.state_space[self.t_idx,aidx,0,3] = phi[aidx]
                if self.DUMMY_EVADER_3D4DOF and aid =='e0':
                    # skip reinitialization for evader
                    continue
                '''
                ## P2
                v_norm_aid = \
                    np.random.uniform(0.001, 
                                      2., #self.agent_specs[aid]['v_norm_max']
                                      )
                self.state_space[self.t_idx,aidx,1,:3] = (v_unit[aidx,:]*v_norm_aid)
                #'''
                
                '''
                ## P3* random orientation not aligned initially 
                self.state_space[self.t_idx,aidx,0,3:] = angles[aidx,:]
                
                #''' 


                '''
                In case we opt for acceleration control we intend to 
                align to original velocity vector with the body z-axis (z_B) 
                which holds the thrust vector and thus have to find* the
                correspoding attitude (phi, theta & psi, where phi != 0!)
                
                This new allignment uses the previous velocity vectors
                (retains them), but finds the for body z-axis
                
                (V_REL_CONTROL has velocity aligned with body x-axis, x_B)
                
                *Note that we estimate the angle, this might not work
                every time!
                '''
                v_orient = self.thrust_orientation.copy() # already unit vector
                #v_orient = np.array([1.,0., 0.])
                '''
                # random initialization
                v_aid = self.state_space[self.t_idx,aidx,1,:3].copy()
                v_aid_norm = np.linalg.norm(v_aid) 
                
                v_aid_unit = np.random.randn(3)
                v_aid_unit = v_aid_unit / np.linalg.norm(v_aid_unit)
                
                '''
                # DDDDDDD
                
                
                # initialization with velocity
                v_aid = self.state_space[self.t_idx,aidx,1,:3].copy()
                v_aid += np.random.randn(3)*0.1 # add noise
                v_aid_norm = np.linalg.norm(v_aid)
                v_aid_unit = v_aid/v_aid_norm 
                #''' 
                
                angles_aid = Rotations.body2inertial_align(v_orient, v_aid_unit)
                # note that regardless of which orientation if has found
                # that orientation will be used for the initial velocity
                # from this point on
                
                
                RotMat_aid = Rotations.body2inertial(angles_aid)
                v_aid_new = RotMat_aid.apply(v_orient*v_aid_norm) # retain norm!
                
                #angles_aid =  R.random().as_euler('xyz', degrees=False) ## ALEX
                
                self.state_space[self.t_idx,aidx,1,:3] = v_aid_new.copy() # velocity
                self.state_space[self.t_idx,aidx,0,3:] = angles_aid # angle
                
                ''' 
                # LEARN2FLY
                angles_aid =  R.random().as_euler('xyz', degrees=False)
                v_aid_unit = np.random.randn(3)*0.1 ## ALEX
                
                #if not self.A_CONTROL_noG:
                #    # no negative Z-coordinate
                #    v_aid_unit[2] = abs(v_aid_unit[2])
                #v_aid_unit = v_aid_unit / np.linalg.norm(v_aid_unit)
                
                self.state_space[self.t_idx,aidx,1,:3] = v_aid_unit.copy() # velocity
                self.state_space[self.t_idx,aidx,0,3:] = angles_aid # angle
                #'''
                
                a_T_range, a_pos = aid_specs['a_T_range'], aid_specs['a_pos']
                pqr_lim = aid_specs['rate_limit']
                
                a_limT = a_pos*a_T_range[0] + a_T_range[1] 
                a_limB = -1*(1-a_pos)*a_T_range[0] 
                a_factor = 0.5 + 0.25*a_pos
                #'''
                #  P,Q,R & T states
                
                self.state_space[self.t_idx,aidx,1,3:] = \
                    np.random.uniform(low= -0.5*pqr_lim, high= 0.5*pqr_lim, size = 3) # p,q,r
                self.state_space[self.t_idx,aidx,2,3] = \
                    np.random.uniform(low= (1-a_factor)*a_limB, high= a_factor*a_limT, size = 1) # T
                '''
                self.state_space[self.t_idx,aidx,2,3] = a_limT
                #'''
                
                #''' 
                # IMU states
                RotMat_BI = Rotations.body2inertial(self.state_space[self.t_idx,aidx,0,3:])
                RotMat_IB = Rotations.inertial2body(self.state_space[self.t_idx,aidx,0,3:])
                
                dx_B, dy_B, dz_B = RotMat_IB.apply(self.state_space[self.t_idx,aidx,1,:3]) # v
                specs = self.agent_specs[aid].copy()
                a_g, drag_C = self.agent_specs[aid]['a_g'], self.agent_specs[aid]['drag_C']
                a_T = self.state_space[self.t_idx,aidx,2,3]
                
                F_B = [-drag_C[0]*dx_B - drag_C[3]*abs(dx_B)*dx_B       - drag_C[6]*(dx_B)/(self.t_delta*self.null_delay),  # Fx_B  ## ALEX
                       -drag_C[1]*dy_B - drag_C[4]*abs(dy_B)*dy_B       - drag_C[7]*(dy_B)/(self.t_delta*self.null_delay),  # Fy_B
                       -drag_C[2]*dz_B - drag_C[5]*abs(dz_B)*dz_B + a_T - drag_C[8]*(dz_B)*(1/(1.+abs(a_T)**(0.25)))/(self.t_delta*self.null_delay),   # Fz_B
                       ]
                self.state_space[self.t_idx,aidx,2,[0,1,2]] = RotMat_BI.apply(F_B) - [0.,0.,a_g] # acc

                #'''
        
        ## Alternative adjustments (dummy evader)
        if self.DUMMY_EVADER_3D4DOF: # evader not learnt (= dummy), i.e. dynamic waypoint
            # wipe orientation
            self.state_space[self.t_idx,1,0,[3,4,5]] = 0. # attitude
            self.state_space[self.t_idx,1,1,[3,4,5]] = 0. # dot-attitude
            self.state_space[self.t_idx,1,2,:] = 0. # acceleration
            # wipe velocity
            ''' 
            velocity_E = [0.,0.,0.01] # 
            self.state_space[self.t_idx,1,1,:3] = velocity_E # should not be completely zero!
            #'#''
            
            '''
            # CL velocity
            v = self.state_space[self.t_idx,1,1,:3].copy()
            v_norm = np.linalg.norm(v)
            
            if self.DUMMY_POINT_3D4DOF:
                v_norm_new = 0.001
                #v_norm_new = v_norm
            else:
                low = 0.5
                cl_factor = (1+1.*max(0,self.CL_taskLevel-1))
                ''' 
                v_norm_new = np.random.uniform(low = low, #1.*cl_factor, 
                                               high = self.DUMMY_EVADER_3D4DOF_V_MAX*cl_factor,
                                               )
                '''
                v_norm_new = np.random.uniform(low = 0.5, #1.*cl_factor, 
                                               high = 3.,
                                               )
                #''' 
                v_norm_new = max(min(v_norm_new, 6.),low)
                #v_norm_new = 15 ## ALEX
            v = (v/v_norm)*v_norm_new
            #v = (v/v_norm)*3.
            self.state_space[self.t_idx,1,1,:3] = v
            self.state_space[self.t_idx,1,2,:3] = 0.
            #''' 
            
            ## MISC todo move
            # wipe drag coeff
            self.agent_specs['e0'].update({
                    ## velo control (dont override)
                    #'v_rel_max':0., # no velocity control
                    #'v_rel_max0':0., # no thrust
                    ## acc control
                    'drag_C':[0., 0., 0.]*3, # no drag
                    'a_T_range':[0.,0.], # no thrust
                    'a_T_range0':[0.,0.], # no thrust
                    'a_g':0. # no gravity
                })

        return
        
#%% OBSERVATION FUNCTIONS
    
    def observation_pursuer_3D_main(self, aidx_ego: int, aidx_tu: int, 
                                    Tdelay_ego: int = 0,
                                    Tdelay_tu: int = 0):
        '''
        Observation retrieval for 3 dimensional state.
        
        Essentially, following main steps are taken
        - transform state_space from inertial to body frame of aidx_ego 
        - in body frame compute LoS its rate as well as 
            distance, closing speed and gamma
        '''
        t_idx_clipped_ego = max((self.t_idx-Tdelay_ego),0)  # >= 0
        t_idx_clipped_tu = max((self.t_idx-Tdelay_tu),0)  # >= 0
        
        state_space_ego_T = self.state_space[self.t_idx,aidx_ego,:,:].copy() # current timestep 
        state_space_tu_T = self.state_space[self.t_idx,aidx_tu,:,:].copy() # current timestep 
        state_space_ego = self.state_space[t_idx_clipped_ego,aidx_ego,:,:].copy()
        state_space_tu = self.state_space[t_idx_clipped_tu,aidx_tu,:,:].copy() 
        # (T,N,D,6) -> 2 sets of (D,6)
        angle_ego_T = state_space_ego_T[0,[3,4,5]].copy()  # phi, theta , psi
        state_space_tu = state_space_tu[:,0:3]
        state_space_ego = state_space_ego[:,0:3]
        '''
        ADD NOISE HERE! such that it flows through the system later stage
        
        CONSIDER;
        - ONLY NOISE FOR TU? or assymetric noise i.e. we know
            outselves better than the other
            - alternatively you could say we have an estimate of
            relative velocity and dont even know ego or tu (thus we only
            have realtive info; but i dont think that the case for us)
        - noise different for the different derivatives, p, v & a
        - noise proportional to the game state; draw an innovation
            and scale it e.g. by the magnitude (e.g. distance)
        '''
        
        #'''
        # EXAMPLE IMPLMENTATION 'TRICLE DOWN NOISE'
        if self.DO_NOISE:
            noise_ego = self.noise_dist.rvs().astype(np.float64).reshape((3,3))/self.noise_scalar # (9,) -> (3,3)
            state_space_ego += noise_ego

            noise_ego_T = self.noise_dist.rvs().astype(np.float64).reshape((3,3))/self.noise_scalar # (9,) -> (3,3)
            state_space_ego_T[:,:3] += noise_ego_T # NOTE NO NOISE FOR EGO ATTITUDE I.E. STATES [3,4,5]
            
            noise_tu = self.noise_dist.rvs().astype(np.float64).reshape((3,3))/self.noise_scalar # (9,) -> (3,3)
            state_space_tu += noise_tu
            


        #'''
        
        ## setup
        delta = state_space_tu - state_space_ego # (3,3), [r_{e-p},v_{e-p},a_{e-p}] 
        delta_T = state_space_tu_T[:,:3] - state_space_ego_T[:,:3]
        
        delta_norm = np.linalg.norm(delta,
                           axis = -1, keepdims = False) # (3,)
        delta_T_norm = np.linalg.norm(delta_T,
                           axis = -1, keepdims = False) # (3,)
        ## rotate from intertial to body frame
        # setup rotation matrix
        #angle_ego_T[2] *= -1
        angle_ego_T *= self.ANGLES_SIGN # in 2d everything *-1
        RotMat_IB = Rotations.inertial2body(angle_ego_T*1) # INERT
        RotMat_BI = Rotations.body2inertial(angle_ego_T*1) # INERT
        
        thrust_orient_I_B = self.thrust_orientation.copy() # do not remove, overriden & used later
        '''
        # RANGE FRAME CENTERED
        thrust_orient_I = RotMat_BI.apply(self.thrust_orientation) # thrust for current attitude in I
        angle_ego_T = Rotations.body2inertial_align(self.thrust_orientation,
                                                  delta[0,:], 
                                                  normalize = 2)
        RotMat_IB = Rotations.inertial2body(angle_ego_T) # INERT
        # ^Note from this point on delta[0,:] is unuseable as it only holds [0,0,R]
        
        thrust_orient_I_B =  RotMat_IB.apply(thrust_orient_I)#*delta_norm[0] 
        # ^NOTE thrust attitude in I now transformed to 'range frame' B
        #'''
        # apply rotation
        
        # by hand = (R.dot(A.T)).T https://stackoverflow.com/questions/68476672/what-does-scipy-rotation-apply
        #delta_B = (RotMat_IB.as_matrix().T @ delta.T).T
        state_space_ego_B = RotMat_IB.apply(state_space_ego) # (D,3)
        state_space_tu_B = RotMat_IB.apply(state_space_tu) # (D,3)
        delta_B = RotMat_IB.apply(delta) # (D,3)
        delta_T_B = RotMat_IB.apply(delta_T) # (D,3)
    
        state_space_ego_T_B = state_space_ego_T.copy()
        state_space_ego_T_B[:,:3] = RotMat_IB.apply(state_space_ego_T[:,:3])
        
        state_space_tu_T_B = state_space_tu_T.copy()
        state_space_tu_T_B[:,:3] = RotMat_IB.apply(state_space_tu_T[:,:3])
        
        
        if False:#(self.SAcode_num in [34]) and not self.V_REL_CONTROL:                       
            # rotate 0.5*pi z-axis downward i.e. Z vector becomes X
            delta_B = (self.rotmat_Z2X@(delta_B.T)).T
            state_space_ego_B = (self.rotmat_Z2X@(state_space_ego_B.T)).T
            state_space_tu_B = (self.rotmat_Z2X@(state_space_tu_B.T)).T
            state_space_ego_T_B[:,:3] = (self.rotmat_Z2X@(state_space_ego_T_B[:,:3].T)).T
            
        # 
        state_space_ego_T_B_norm = np.linalg.norm(state_space_ego_T[:,:3], axis = -1, keepdims = False) # (D,)
        state_space_ego_T_B_unit = state_space_ego_T[:,:3]/(state_space_ego_T_B_norm[:, np.newaxis]+1e-12) # (3,3)/(3,1) = (3,3)
        
        state_space_tu_T_B_norm = np.linalg.norm(state_space_tu_T_B[:,:3], axis = -1, keepdims = False) # (D,)
        state_space_tu_T_B_unit = state_space_tu_T_B[:,:3]/(state_space_tu_T_B_norm[:, np.newaxis]+1e-12) # (3,3)/(3,1) = (3,3)
        
        
        
        state_space_ego_B_norm = np.linalg.norm(state_space_ego_B, axis = -1, keepdims = False) # (D,)
        state_space_ego_B_unit = state_space_ego_B/(state_space_ego_B_norm[:, np.newaxis]+1e-12) # (3,3)/(3,1) = (3,3)
        # note velo_ego_B is [1,0,0] for SA cases with constant velocity, since own body frame
        
        state_space_tu_B_norm = np.linalg.norm(state_space_tu_B, axis = -1, keepdims = False) # (D,)
        state_space_tu_B_unit = state_space_tu_B/(state_space_tu_B_norm[:, np.newaxis]+1e-12) # (3,3)/(3,1) = (3,3)

        delta_B_norm = np.linalg.norm(delta_B,
                           axis = -1, keepdims = False) # (D,)
        delta_B_unit = delta_B/(delta_B_norm[:, np.newaxis]+1e-12) # (3,3)/(3,1) = (3,3)
        
        delta_T_B_norm = np.linalg.norm(delta_T_B,
                           axis = -1, keepdims = False) # (D,)
        delta_T_B_unit = delta_T_B/(delta_T_B_norm[:, np.newaxis]+1e-12) # (3,3)/(3,1) = (3,3)
        # normalized = unit vector delta
        
        ## 
        '''
        MAIN EXPLANATION;
        - 2D cases have rotations applied wrt inertial frame (i.e. is main frame)
        - 3D cases have rotations applied wrt body frame (i.e. is main frame)
        - all angles are in the body frame
        
        Here we compute dLoS in the body frame, i.e. the angular rates desired.
        however, all angles are in inertial frame, so whatever we compute
        has to be multiplied with -1 (for the 3d cases). Example; you 
        compute in body frame that you need to turn +1 rad wrt to current orientation
        that means -1*(+1 rad) in inertial frame or at least that is what 
            Rotations.body2inertial(angles_change) will do for you
        
        THUS having -1 here for LoS is completely fine, you just have to 
        know why!
        
        -> on the other hand having an inconsistent frame for 2d vs 3d case
        is annoying!
        ''' 
        BODY_FRAME_SIGN = -1 # THIS IS REQUIRED SINCE ITS IN THE BODY FRAME, IMPLIES omega = (v X r)/r^2 but it works | I DONT NOW WHICH VERSION IS CONSISTENT WITH THE ANGLES (NOT-RATE)        
        #'''

        
        dLoS = BODY_FRAME_SIGN*np.cross(delta_B[0,:], delta_B[1,:])/(np.dot(delta_B[0,:],delta_B[0,:])+1e-6) # INCORRECT_SIGN*(r x v)/|r|^2 
        dLoS *= self.ANGLES_SIGN
        
        #Vc = np.dot(delta_B[0,:], delta_B[1,:]) * delta_B_unit[0,:] / delta_B_norm[0]
        #dLoS *= Vc ## ALEX
        
        #dLoS = np.clip(dLoS, -10., 10.)
        #dLoS *= delta_B_norm[0] ## ALEX
        #dLoS = -np.cross((state_space_ego_B_unit[1,:] + 0.01), dLoS)*(delta_B_norm[1])
        #dLoS = np.cross(state_space_ego_B_unit[1,:], dLoS)*delta_B_norm[1]*delta_B_norm[0] # MCPG
        #dLoS = np.cross(state_space_ego_B_unit[0,:], dLoS)*delta_B_norm[1]*delta_B_norm[0] # MCPG
        
        #dLoS = np.cross([1.,0.,0.], dLoS)*delta_B_norm[1]*delta_B_norm[0] # MCPG
        
        #dLoS = np.cross(delta_B[1,:], dLoS)#*delta_B_norm[1]
        #dLoS = np.cross(delta_B[1,:], dLoS)#*delta_B_norm[1]
        #dLoS = 1*delta_B_unit[0,:]*dLoS*state_space_ego_B_norm[1]
        #dLoS = np.cross(delta_B[1,:], dLoS)#*delta_B_norm[1] # PN
        '''
        dLoS = BODY_FRAME_SIGN*np.cross(delta_B[0,:], delta_B[1,:])/np.dot(delta_B[0,:],delta_B[0,:]) # INCORRECT_SIGN*(r x v)/|r|^2 
        #dLoS = BODY_FRAME_SIGN*np.cross([1.,0.,0.], delta_B[1,:])#/(delta_B_norm[0]+1e-8)
        #dLoS = BODY_FRAME_SIGN*np.cross(state_space_ego_B_unit[1,:], delta_B[1,:])/(delta_B_norm[0]+1e-8)
        #dLoS = BODY_FRAME_SIGN*(delta_B[1,:])/(delta_B_norm[0]+1e-8)
        
        angles_R2T = Rotations.body2inertial_align(delta_B_unit[0,:], 
                                                   np.array([1., 0., 0.]))
        RotMat_R2T = Rotations.body2inertial(angles_R2T)
        dLoS = RotMat_R2T.apply(dLoS)*delta_B_norm[0]
        #'''
        
        ''' 
        TODO I HATE THIS INCORRECT ORDER FOR OMEGA wrt to 2d/3d discrepancy
        ''' 
        #dLoS_invV = dLoS*(1/delta_B_norm[1])# this seems important information, looking at PN laws
        

        ''' 
        # (A) acceleration normal to instantaneous line of sight (in body frame)
        # https://en.wikipedia.org/wiki/Proportional_navigation -> this is in intertial frame
        dLoS_los = (BODY_FRAME_SIGN*-1)*np.cross(delta_B_unit[0,:],dLoS)  # -1*(r_unit x omega)
        # if sign is reversed (i.e. BODY_FRAME_SIGN = 1 != -1), then we need the minus here!!!

        acc_los = delta_B_norm[1]*dLoS_los # NOTE no gain yet & I think the norm makes it acceleration -> NOT TRUE?
        #'''  
        
        ''' 
        # B) acceleration normal to ego velocity orientation (in body frame)
        # https://en.wikipedia.org/wiki/Proportional_navigation -> this is in intertial frame
        dLoS_Vego = (BODY_FRAME_SIGN*-1)*np.cross(state_space_ego_B_unit[1,:],dLoS) # -1*(v_ego_unit x omega)
        # if sign is reversed (i.e. BODY_FRAME_SIGN = 1 != -1), then we need the minus here!!!

        acc_Vego = delta_B_norm[1]*dLoS_Vego  # NOTE no gain yet & I think the norm makes it acceleration -> NOT TRUE?
        
        #''' 
        
        # OLD COMMENTS STARTING FROM THIS LINE
        # not  r x v as that seems to be escape ones? 
        # also PN for acc often has -1* so that also v x r (since = r x v *-1)
        
        #dLoS = np.cross(delta_B[1,:],dLoS)
        #dLoS = 1.*np.cross(delta_B_unit[0,:],dLoS)*delta_B_norm[1]
        
        dLoS_dphi, dLoS_dtheta, dLoS_dpsi = dLoS
        
        ''' 
        if aidx_ego == 0:
            print(f'=\n{delta_B[0,:2]}')
            print(f'NEW={delta_B[1,0]*delta_B[0,1]-delta_B[1,1]*delta_B[0,0]} === {dLoS}')
        ''' 
        # NOTE THAT CROSSPRODUCT r x v = ANGULAR MOMENTUM = L IN 3 AXIS AND 
        # omega = L/I where I is moment of inertia i.e. np.dot(r,r)*Mass 

        #if aidx_ego == 0:
        #   print(delta_B[0,:])
        #'''
        #TODO THETA

        #LoS_vector = delta_B[0,:] # R
        LoS_vector = ((self.rotmat_Z2X)@(delta_B[[0],:].T))[:,0] # R
        #LoS_vector = delta_B_unit[0,:] - state_space_ego_B_unit[1,:] # R-V_ego, both unit

        #LoS_phi = np.arctan2(LoS_vector[2],-LoS_vector[1])   # yz, roll angle
        LoS_phi = np.arctan2(LoS_vector[1],LoS_vector[2])   # zy, roll angle (aims for alignment with positive z-axis i.e. y is reduced)
        LoS_theta = np.arctan2(LoS_vector[2],LoS_vector[0])   # xz, pitch angle TODO [-0.5pi, 0.5pi] ABS? (aims for alignment with positive x-axis i.e. z is reduced)
        LoS_psi = np.arctan2(LoS_vector[1],LoS_vector[0])   # xy, yaw angle   (aims for alignment with positive x-axis i.e. y is reduced)
            
        LoS_phi = ((LoS_phi +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        #LoS_phi *= self.ANGLES_SIGN
        LoS_theta = ((LoS_theta +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        #LoS_theta = max(min(LoS_theta,0.5*np.pi), - 0.5*np.pi)
        #LoS_theta = max(min(LoS_theta,np.pi), 0.)
        
        #LoS_phi *= self.ANGLES_SIGN
        LoS_psi = ((LoS_psi +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
        ''' 
        # in body frame
        #LoS = BODY_FRAME_SIGN*np.cross(delta_B_unit[0,:],self.thrust_orientation)
        #LoS = BODY_FRAME_SIGN*np.cross(delta_B[0,:],self.thrust_orientation)/np.dot(delta_B[0,:],delta_B[0,:])
        #delta_B[0,:],  #delta_B_unit
        LoS = delta_B_unit[0,:]
        
        LoS_phi, LoS_theta, LoS_psi = LoS
        #'''
        #self.errors_last[self.agent_id[aidx_ego]] = np.array([LoS_phi, LoS_theta, LoS_psi])
        #LoS_psi *= self.ANGLES_SIGN
        #LoS_phi = np.arcsin(delta_B_unit[0,0])   # yz, roll angle
        #LoS_theta = -np.arcsin(delta_B_unit[0,2])   # xz, pitch angle TODO [-0.5pi, 0.5pi] ABS?
        #LoS_psi = np.arcsin(delta_B_unit[0,1])
        
        # np.arctan2 choses correct quadrant based on the sign
        
        '''
        def quaternion_error(current_quat, desired_quat):
            current_rot = R.from_quat(current_quat)
            desired_rot = R.from_quat(desired_quat)
            error_rot = desired_rot * current_rot.inv()
            return error_rot.as_quat(canonical = True)

        # Function to convert a direction to a quaternion
        def direction_to_quat(direction):
            direction = direction / np.linalg.norm(direction)
            z_axis = np.array([0, 0, 1])
            v = np.cross(z_axis, direction)
            s = np.sqrt((1 + np.dot(z_axis, direction)) * 2)
            q = np.array([v[0], v[1], v[2], s * 0.5])
            q /= np.linalg.norm(q)
            return q
        
        RotMat_IB

        delta_B_unit[0,:] 
        
        error_angles_q = 
        desired_quat = direction_to_quat(direction_to_target)

        error_quats = quaternion_error(interceptor_quat, desired_quat)
        error_axis_angles = R.from_quat(error_quat).as_rotvec()
        '''
        error_angles = [0]*3
        
        LoS_dLoS_ordered = [LoS_phi, dLoS_dphi, LoS_theta, dLoS_dtheta, LoS_psi, dLoS_dpsi]
        '''
        ## Angular error for the velocity and thrust vector
        # this is only relevant in case v & thrust vector are disentangled
        LoS_v_vector = state_space_ego_B_unit[1,:]
        LoS_v_phi = ((np.arctan2(LoS_v_vector[1],LoS_v_vector[2])+np.pi) % (2.*np.pi)) - np.pi   
        # zy, roll angle (aims for alignment with positive z-axis i.e. y is reduced)
        LoS_v_theta = ((np.arctan2(LoS_v_vector[2],LoS_v_vector[0])+np.pi) % (2.*np.pi)) - np.pi  
        # xz, pitch angle TODO [-0.5pi, 0.5pi] ABS? (aims for alignment with positive x-axis i.e. z is reduced)
        LoS_v_psi = ((np.arctan2(LoS_v_vector[1],LoS_v_vector[0])+np.pi) % (2.*np.pi)) - np.pi
        # xy, yaw angle   (aims for alignment with positive x-axis i.e. y is reduced)
        
        #LoS_v_norms2d = [np.linalg.norm(state_space_ego_B[1,[1,2]]),
        #                 np.linalg.norm(state_space_ego_B[1,[0,2]]),
        #                 np.linalg.norm(state_space_ego_B[1,[0,1]])]
        LoS_v = [LoS_v_phi, LoS_v_theta, LoS_v_psi, 
                 #*LoS_v_norms2d,
                 np.linalg.norm(state_space_ego_B[1,:]),
                 ]
        
        
        ## Closing speeds
        Vc_xy = save_divide_scalar((delta_B[0,0]*delta_B[1,0]+delta_B[0,1]*delta_B[1,1]),
            ((delta_B[0,0]**2+delta_B[0,1]**2)**(1/2.)))
            
        Vc_xz = save_divide_scalar((delta_B[0,0]*delta_B[1,0]+delta_B[0,2]*delta_B[1,2]),
            ((delta_B[0,0]**2+delta_B[0,2]**2)**(1/2.)))
            
        Vc_yz = save_divide_scalar((delta_B[0,1]*delta_B[1,1]+delta_B[0,2]*delta_B[1,2]),
            ((delta_B[0,1]**2+delta_B[0,2]**2)**(1/2.))) 
        # this is zero if gamma is 1 and v aligns with x-axis
        Vc = [Vc_xy, Vc_xz, Vc_yz]
        '''
        
        #Vc = np.dot(delta_B[0,:], delta_B[1,:]) * delta_B_unit[0,:] / delta_B_norm[0]
        #'''
        ##
        '''
        r_norm = delta_B_norm[0]
        #v_norm = delta_B_norm[1] # see below
        a_norm = delta_B_norm[2]
        dd = np.dot(-1.*delta_B_unit[0,:], delta_B[1,:]) # (r/|r|)*v, not normalized wrt to v, *-1 for rano consistency
        
        # TODO component specific closing speeds?
        
        
        ## 
        # uV = unitVelocity, these metrics are considered to consider game state invariant of velocity differences
        v_uV = state_space_tu_B_unit[1,:] - state_space_ego_B_unit[1,:] # unit-velocity diff
        v_uV_norm = np.linalg.norm(v_uV) #  norm of unit-velocity diff, ranges in [0,2]
        v_uV_unit = v_uV/(v_uV_norm+1e-12)
        
        v_norm = delta_B_norm[1] # norm of velocity diff
        v_norm_ranged = state_space_ego_B_norm[1]*(self.agent_specs[self.agent_id[aidx_ego]]['v_rel']/2)-1. #[-1,1]
        v_norm_ratio = state_space_tu_B_norm[1]/(state_space_ego_B_norm[1]+1e-15) # ratio of velocities, in come SA cases this is constnat
        #v_norm_uV_ratio = 1. # ALWAYS
        v_uV_dot = np.dot(state_space_tu_B_unit[1,:3], state_space_ego_B_unit[1,:3]) # alignment of unit-velocities
        # TODO is v_uV_dot not similar to /the same as v_norm_uV; CURRENTLY v_uV_norm IS NOT USED AS WE THINK ITS OVERKILL
        
        ##
        gamma = np.dot(delta_B_unit[0,:], delta_B_unit[1,:]) # (1,) 'range vector correlation', -1 for r_{p-e} previously; this is relative
        gamma_uV = np.dot(delta_B_unit[0,:], v_uV_unit)
        # gamma but normalized wrt to relative velocities
        gamma_pp_ego = np.dot(delta_B_unit[0,:], state_space_ego_B_unit[1,:]) # pure-pursuit 'score' of yourself (not relative), 
        gamma_pp_tu = np.dot(delta_B_unit[0,:], state_space_tu_B_unit[1,:]) # pure-pursuit 'score' of other (not relative)
        # note that _pp are uV technically
        '''
        r_norm = delta_T_B_norm[0]
        #v_norm = delta_T_B_norm[1] # see below
        a_norm = delta_T_B_norm[2]
        dd = np.dot(-1.*delta_T_B_unit[0,:], delta_T_B[1,:]) # (r/|r|)*v, not normalized wrt to v, *-1 for rano consistency
        
        # TODO component specific closing speeds?
        
        
        ## 
        # uV = unitVelocity, these metrics are considered to consider game state invariant of velocity differences
        v_uV = state_space_tu_T_B_unit[1,:] - state_space_ego_T_B_unit[1,:] # unit-velocity diff
        v_uV_norm = np.linalg.norm(v_uV) #  norm of unit-velocity diff, ranges in [0,2]
        v_uV_unit = v_uV/(v_uV_norm+1e-12)
        
        v_norm = delta_T_B_norm[1] # norm of velocity diff
        v_norm_ranged = state_space_ego_T_B_norm[1]*(self.agent_specs[self.agent_id[aidx_ego]]['v_rel']/2)-1. #[-1,1]
        v_norm_ratio = state_space_tu_T_B_norm[1]/(state_space_ego_T_B_norm[1]+1e-15) # ratio of velocities, in come SA cases this is constnat
        #v_norm_uV_ratio = 1. # ALWAYS
        v_uV_dot = np.dot(state_space_tu_T_B_unit[1,:3], state_space_ego_T_B_unit[1,:3]) # alignment of unit-velocities
        # TODO is v_uV_dot not similar to /the same as v_norm_uV; CURRENTLY v_uV_norm IS NOT USED AS WE THINK ITS OVERKILL
        
        ##
        gamma = np.dot(delta_T_B_unit[0,:], delta_T_B_unit[1,:]) # (1,) 'range vector correlation', -1 for r_{p-e} previously; this is relative
        gamma_uV = np.dot(delta_T_B_unit[0,:], v_uV_unit)
        # gamma but normalized wrt to relative velocities
        gamma_pp_ego = np.dot(delta_T_B_unit[0,:], state_space_ego_T_B_unit[1,:]) # pure-pursuit 'score' of yourself (not relative), 
        gamma_pp_tu = np.dot(delta_T_B_unit[0,:], state_space_tu_T_B_unit[1,:]) # pure-pursuit 'score' of other (not relative)
        # note that _pp are uV technically
        
        #'''
        # already equal to gamma_catd in case v_norm == 1, but this is not likely 
        
        #obs = np.array([r_norm, gamma, *Vc, *LoS_dLoS_ordered], dtype = np.float64) # (9,)
        
        #orientation_ego = state_space[aidx_ego,0,[3,5]] # theta, psi
        
        ## COLLECT AND RETURN
        '''
        distance = [] #[r_norm, dd]
        Vc = []
        #'''
        # EXTRA_ACTION_CODE
        distance = [r_norm, #*delta_B[0,:],
                    ] # 4
        Vc = [v_norm, v_norm_ratio, v_uV_dot, #v_norm_ranged, 
              gamma_pp_ego, #*delta_B[1,:], *Vc,
              ] # 3*3
        #'''
        
        ''' 
        orientation_ego = []
        '''
        ##
        v_I_ego = state_space_ego[1,:]
        v_I_ego_norm = np.linalg.norm(state_space_ego[1,:])
        v_I_ego_unit = v_I_ego/(v_I_ego_norm+1e-8)
        v_I_downward = -1*np.dot(v_I_ego_unit,[0.,0.,-1.]) # determine alignment v vector
        ##
        
        z_I_downward = -1*np.dot(
                            RotMat_BI.apply([0.,0.,-1.]), # body to inertial
                            [0.,0.,-1.], # inertial
                        ) # determine alignment current orientation (e_z vector)
        
        '''
        angles_BrG_ego = BODY_FRAME_SIGN*np.cross(RotMat_IB.apply([0.,0.,1.]),
                                  [0.,0., -1.]) # gavity_B!_unit x T_B_unit
        ''' 
        #angles_BrG_ego = BODY_FRAME_SIGN*np.cross(state_space_ego_B_unit[1,:],
        #                          [0.,0., -1.]) # gavity_B!_unit x T_B_unit
        #'''
        Q = R.from_euler('xyz', angle_ego_T, degrees = False).as_quat(canonical = True)
        
        Z_I_T_B = RotMat_IB.apply([0., 0., -1.]) # inertial Z axis in body frame 
        Z_B_T_I = RotMat_BI.apply([0., 0., 1.]) # body Z axis in inertial frame 
        # CCCCCCCCCCCCCCCC
        #'''
        state_ego = [#*angle_ego_T,  # angles (inertial)
                     #*Z_I_T_B, # Z-axis in body frame
                     *Z_B_T_I, # Z-axis in INERT frame
                     
                     *state_space_ego_T_B[1,:3],
                     *state_space_ego_T_B[2,:3],
                                       
                    #*angles_BrG_ego,  # angles relative to gravity (body)
                    #*v_I_ego_unit,  # v 
                    #v_I_ego_norm,
                    #v_I_downward,
                    #z_I_downward,
                    #*v_I_ego,
                    #*delta[0,:], # inertial frame
                    
                    *delta_B[0,:],
                    *delta_B[1,:],
                    
                    *state_space_ego_B[1,:],
                    *state_space_tu_B[1,:],
                    
                    *Q,
                    *state_space_ego_T_B[1,3:],state_space_ego_T_B[2,3],
                    
                    *thrust_orient_I_B,
                    
                    #*error_angles, 
                    ]
        '''
        state_ego = [*angle_ego_T,  # angles (inertial)
                     #*Z_I_T_B, # Z-axis in body frame
                     
                     *state_state_ego_T_B[1,:3],
                     *state_state_ego_T_B[2,:3],
                                       
                    #*angles_BrG_ego,  # angles relative to gravity (body)
                    #*v_I_ego_unit,  # v 
                    #v_I_ego_norm,
                    #v_I_downward,
                    #z_I_downward,
                    #*v_I_ego,
                    #*delta[0,:], # inertial frame
                    
                    *delta_B_unit[0,:],
                    *delta_B_unit[1,:],
                    
                    *state_space_ego_B_unit[1,:],
                    *state_space_tu_B_unit[1,:],
                    
                    *Q,
                    *state_state_ego_T_B[1,3:],state_state_ego_T_B[2,0],
                    
                    *thrust_orient_I_B,
                    ]
        
        
        #'''
        '''
        dLoS_others = [
                *LoS_v,
                *np.cross(delta_B_unit[0,:], dLoS),    
                *np.cross(delta_B_unit[1,:], dLoS),
                *np.cross(state_space_ego_B_unit[1,:], dLoS),
            ]
        '''
        obs = np.array([*LoS_dLoS_ordered, *distance, *Vc, *state_ego,
                        #*dLoS_others,
                        ], 
                       dtype = np.float64) # (6,)
        
        obs = np.nan_to_num(obs, copy = False, 
                            nan = 0., posinf=0., neginf=0.) # handle invalid data
        obs_N = len(obs)
        
        ##
        x_targets = obs[[1,3,5]].copy()[-self.AL_DIM:] # if zero, does not pass next gate at **
        
        ## add noise
        ''' 
        # NOISE ADDED ABOVE!
        if self.DO_NOISE:
            noise = self.noise_dist.rvs().astype(np.float64) # sample from noise distribution       
            
            obs += noise
        ''' 
        ## blindsight
        '''
        # NAIVE BS IMPLEMENTATION; MASK EVERYTHING
        if self.DO_BS:
            # blindsight entire observation vector, like guido wanted
            BSed = (1-bool(self.BS_idxLag[self.agent_id[aidx_ego]])) # BSed = 'blind-sighted'
            # if lag > 0, multiply by 0.
            obs *= BSed
            x_targets *= BSed # if you cannot see, dont get a loss for not being able to predict
        #'''
        
        #'''
        if self.DO_BS:
            # CURATED BS IMPLEMENTATION; TRICLE DOWN 
            #IDEA (two compoentns); 
            #1. mask everything in with the argument that its optic flow we dont have s[t,aidx_masked,1:,:]
            #2. use delay on previous location
            #-> these together imply that the adversary is not moving and in the same location as previously
            
            non_ground_idx = [1,3,5, # dLoS obs
                              7,8,9,10, #11, # v-related obs
                              ] # non-ground state indices (D>0, e.g. velocity/acceleration)
            extra_idx_blind = [
                             11,12,13,
                             
                             #14,15,16,
                             #17,18,19,
                             
                             #20,21,22,
                             23,24,25,
                             
                             #26,27,28,
                             29,30,31,
                               ]
            #extra_idx_blind = []
            blind_idx = [*non_ground_idx, *extra_idx_blind]
            # simple implementation is doing this at the end (but its more robust if we do this at the start)
            
            # NOTE THAT THIS IS STILL WRONG, BECAUSE THIS IMPLIES YOU YOURSELF HAVE ALSO STOPPED MOVING
            # WHY? BECAUSE EVERYTHING IS RELATIVE THUS IF '= 0'  ANYWHERE IT MEANS THAT NEITHER OBJECT IS MOVING
            #   OR THAT BOTH OBJECT ARE MOVING in EXACTLY THE SAME MANNER
            obs[blind_idx] *= (1-bool(self.BS_idxLag[self.agent_id[aidx_ego]])*self.BS_active[self.agent_id[aidx_ego]])
        #''' 
        ## additional inputs for value network and losses
        self.value_idx = list(obs_N+np.arange(0,self.VALUE_DIM, dtype = np.int32)) 
        # initialize outside since always used
        if bool(self.VALUE_DIM):#bool(self.Value_dim):
            # remember that value func looks into the future,
            t_left = (1.-self.t_idx/self.t_idx_limit) # length of episode left
            value_inputs = [
                            #*LoS_dLoS_ordered,
                            #*state_space_ego[0,3:], # angles
                            #*state_space_ego[1,3:], # velocity vector
                            t_left, 
                            r_norm, #dd, 
                            v_norm, 
                            v_norm_ratio, v_uV_dot,  #v_uV_norm NOT USED AS OVERKILL WITH v_uV_dot
                            gamma, 
                            gamma_uV,# choose either gamma or gamma_uV? no because BS uses gamma!
                            gamma_pp_ego, gamma_pp_tu, 
                            ]
            if self.DO_BS:
                BS_lag_tu = self.BS_idxLag[self.agent_id[aidx_tu]]*self.BS_active[self.agent_id[aidx_tu]]
                BS_lag_ego = self.BS_idxLag[self.agent_id[aidx_ego]]*self.BS_active[self.agent_id[aidx_ego]]
                value_inputs_BS = [#BS_lag_ego, #/self.t_idx_limit,
                                   bool(BS_lag_ego)*1.,
                                   #BS_lag_tu, #/self.t_idx_limit,
                                   bool(BS_lag_tu)*1.,
                                   ] # the blinding lag and whether its blinded currently for both agents
                # TODO BLINDSIGHT NORMALIZE E.G. /self.t_idx_limit,
                # while blinding cannot happen simultaneously, we reckon it important to know if you were blinded
                value_inputs += value_inputs_BS
            ##
            value_inputs = np.array(value_inputs,dtype = np.float64)
            obs = np.hstack((obs,value_inputs))
        ## add supervised targets, **
        if bool(self.AL_DIM):
            obs = np.hstack((obs,x_targets))

        if self.A_gain: # save last observations
            #vecs = delta_B_unit.copy().flatten() # (9,) r, v, a (delta & unit)
            ##
            '''
            II = np.array([LoS_phi, LoS_theta, LoS_psi])
            II /= np.linalg.norm(II)
            
            ##    
            # notice v dim is used in the following
            III_phi = -np.arctan2(delta_B[1,2],delta_B[1,1])   # yz, roll angle
            III_theta = np.arctan2(delta_B[1,2],delta_B[1,0])   # xz, pitch angle TODO [-0.5pi, 0.5pi] ABS?
            III_psi = np.arctan2(delta_B[1,1],delta_B[1,0])   # xy, yaw angle  
            
            III_phi = ((III_phi +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
            III_theta = ((III_theta +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
            III_psi = ((III_psi +np.pi) % (2.*np.pi)) - np.pi # set to [-pi,pi] range
            
            III = np.array([III_phi, III_theta, III_psi])
            III /= np.linalg.norm(III)
            #'''
            #I, II, III = np.array([1./np.sqrt(3)]*3)/r_norm, vecs[:3]/r_norm, vecs[3:6]/r_norm
            '''
            II = np.array([
                        -1*(II[2]-II[1]),
                         1*(II[0]-II[2]),
                           (II[1]-II[0]),
                           ])
            III = np.array([
                        1*(III[2]/III[1]),
                         1*(III[0]/III[2]),
                           (III[1]/III[0]),
                           ])

            ##
            self.obs_last[self.agent_id[aidx_ego]] = \
                np.hstack((obs[[0,1,2,3,4,5]].copy(), # (6,)
                           #I,
                           #II,  # (3,)
                           #III,  # (3,)
                           #vecs,
                           ))
            '''
            self.obs_last[self.agent_id[aidx_ego]] = \
                obs[[0,1,2,3,4,5]].copy()
        return obs
        
    
    def observation_pursuer_2D1DOF(self, aidx_ego: int, aidx_tu: int,
                                   Tdelay_ego: int = 0,
                                   Tdelay_tu: int = 0):
        ''' 
        Observation retrieval for 2 dimensional 1 DOF setting.
        
        Wrapper function with output adjustment to use 3D observation case for 2D
        
        Recall that in 2d, z-axis states (=column idx 3) are zero
        and rotations are considered around z-axis. Also no acceleration is 
        present.
        '''
        #''' 
        
        obs = self.observation_pursuer_3D_main(aidx_ego, aidx_tu, 
                                               Tdelay_ego, Tdelay_tu)
        #raise Exception("CHECK INPUTS")
        #obs = obs[[4,5,6,7]] # LoS_ & dLoS_ phi & d, dd, 
        idx_2d1dof = [4,5]
        obs = obs[[*idx_2d1dof, *self.value_idx, *self.AL_idx]] # LoS_ & dLoS_ phi
        
        obs = np.maximum(np.minimum(obs,self.observation_space_limitT),
                         self.observation_space_limitB)
        return obs
    
    
    def observation_pursuer_3D2DOF(self, aidx_ego: int, aidx_tu: int, 
                                   Tdelay_ego: int = 0,
                                   Tdelay_tu: int = 0):
        ''' 
        Observation retrieval for 3 dimensional 2 DOF setting.
        
        Wrapper function with output adjustment to use 3D
        
        Recall that for 3D-2DOF phi is constant at zero i.e. no roll.
        Also no acceleration is present.
        '''
        obs = self.observation_pursuer_3D_main(aidx_ego, aidx_tu, 
                                               Tdelay_ego, Tdelay_tu)
        #raise Exception("CHECK INPUTS")
        #obs = obs[[2,3,4,5,6,7]] # (Los dLos for theta & psi), d, dd, 
        idx_3d2dof = [2,3,4,5]
        idx_extra = list(range(6,6+5))
            
        obs = obs[[*idx_3d2dof,*idx_extra, *self.value_idx, *self.AL_idx]]
            
        obs = np.maximum(np.minimum(obs,self.observation_space_limitT),
                         self.observation_space_limitB)
        return obs


    def observation_pursuer_3D3DOF(self, aidx_ego: int, aidx_tu: int, 
                                   Tdelay_ego: int = 0,
                                   Tdelay_tu: int = 0):
        ''' 
        Observation retrieval for 3 dimensional 3 DOF setting.
        
        Wrapper function with output adjustment to use 3D
        
        Recall that for 3D-3DOF no acceleration is present.
        '''
        obs = self.observation_pursuer_3D_main(aidx_ego, aidx_tu, 
                                               Tdelay_ego, Tdelay_tu)
        #raise Exception("CHECK INPUTS")
        #obs = obs[[0,1,2,3,4,5,6,7, *self.AL_idx]] # Los_dLoS pairs,  d, dd,
        idx_3d3dof = [0,1,2,3,4,5]
        idx_extra = list(range(6,6+5))
            
        obs = obs[[*idx_3d3dof,*idx_extra, *self.value_idx, *self.AL_idx]]
        
        obs = np.maximum(np.minimum(obs,self.observation_space_limitT),
                         self.observation_space_limitB)
        return obs
    
    
    def observation_pursuer_3D4DOF(self, aidx_ego: int, aidx_tu: int, 
                                   Tdelay_ego: int = 0,
                                   Tdelay_tu: int = 0):
        ''' 
        Observation retrieval for 3 dimensional 3 DOF setting.
        
        Wrapper function with output adjustment to use 3D
        
        Recall that for 3D-3DOF no acceleration is present.
        '''
        obs = self.observation_pursuer_3D_main(aidx_ego, aidx_tu, 
                                               Tdelay_ego, Tdelay_tu)
        #raise Exception("CHECK INPUTS")
        #obs = obs[[0,1,2,3,4,5,6,7, *self.AL_idx]] # Los_dLoS pairs,  d, dd,
        '''
        idx_3d4dof = [0,1,2,3,4,5]
        idx_extra = [6,7] #list(range(6,6+5))
        idx_state = [14,15,16, 
                     17,18,19] #list(range(11,11+15+2))
        idx_extraDlos = []#list(range(20,20+4))
        '''
        '''
        idx_3d4dof = [0,1,2,3,4,5]
        idx_extra = [6,7] #list(range(6,6+5))
        idx_state = [11,12,13] #list(range(11,11+15+2))
        idx_extraDlos = []#list(range(20,20+4))
        
        #'''
        
        #'''
        idx_3d4dof = [0,1,2,3,4,5] ###
        idx_extra = []#[6,7] #list(range(6,6+5))
        idx_state = [#11,12,13, # angles_T_ego ###
                     
                     14,15,16, # v_T_ego
                     17,18,19, # a_T_ego ###
                     
                     #20,21,22, # delta_R
                     #23,24,25, # delta_V
                     
                     #26,27,28, # v_ego
                     #29,30,31, # v_tu
                     
                     32,33,34,35, # Q
                     36,37,38,39, #p,q,r,T
                     
                     #40,41,42, # thrust_orient_I_B (Range Frame)
                     ] #list(range(11,11+15+2))
        idx_extraDlos = []#list(range(20,20+4))
        #'''
        obs = obs[[*idx_3d4dof,*idx_extra,*idx_state,*idx_extraDlos,
                   *self.value_idx, *self.AL_idx]]
        
        obs = np.maximum(np.minimum(obs,self.observation_space_limitT),
                         self.observation_space_limitB)
        return obs
    
    def observation_evader(self, aidx_ego: int, aidx_tu: int,
                           Tdelay_ego: int = 0,
                           Tdelay_tu: int = 0):
        '''
        CURRENTLY WRAPPER FUNCTION
        
        TODO/IDEAS; 
        - optic flow is only possible through comparison to previous
            angular rates, thus its not available if CATD is upheld
            -> instead of providing true velo you can provide approximations!
        '''
        obs = self.observation_pursuer(aidx_ego, aidx_tu, 
                                        Tdelay_ego, Tdelay_tu) 
        return obs
        
#%% REWARD FUNCTIONS

    def func_d_scaler1(self, x):
        disT_Flogd, disT_Fd = 2., 2. # exponent factor for log(dis**) and dis**
        stable = 1e-4 
        
        y = np.log((x**disT_Flogd+stable)) + x**disT_Fd
        
        y += 10
        return y

    def func_d_scaler2(self, x):
        disT_Flogd, disT_Fd = 2., 2. # exponent factor for log(dis**) and dis**
        stable = 1e-4 
        
        a = 1*(10**(-1.5))
        b = 3
        
        y = -1./((x)**2 + a) + (x+b)**2 - (b+1)**2
        
        y += 39
        return y
    
    def zerosum_reward_main_USED(self, scores_t, outcomes_t):
        '''
        TODO CONNECT THIS TO AGENT SPECIFIC FUNCTIONS LIKE OBSERVATIN FUNCS
        
        NOTE THAT THE SCORES & OUTCOMES ARE ASSUMED TO BE FOR CURRENT TIMESTEP
        '''
        outcomes_t_noCopy = outcomes_t
        outcomes_t = outcomes_t.copy()
        scores_t = scores_t.copy()
        
        done_bool = any(outcomes_t.values())
        ##
        scores_game_t = scores_t['game'].copy()
        dis_t = scores_game_t['dis']
        gamma_t = scores_game_t['Gamma']
        gamma_uV_t = scores_game_t['Gamma_uV']
        # TODO MAKE THIS AGENT SPEICIF LIKE THE OBSERVATION FUNC
        
        timestep_factor = self.reward_MaxStepScaler # =1/(t_max/t_delta), to make reward timestep amount invariant
        # TODO IN ORDER TO MAKE REWARD SYSTEM TIMESTEP INVARIANT (Game of drones source)
        # TODO REWARD PER TIMESTEP WILL CHANGE ACROSS LEVELS!
        
        aidxP, aidxE = self.agent_specs['p0']['idx'], self.agent_specs['e0']['idx']
        # aaaaaaaaaaaa
        
        REWARD_P, REWARD_E = 0., 0.
        
        #r_time, r_INT, r_disDense, r_disT = 10., 10., 5.*timestep_factor, 0., # for minmax scaled Rdis
        
        #r_time, r_INT, r_disDense, r_disT = 5., 5., 10., 10. # for non-scaled Rdis, 10 is not required but might be nice for underflow
        #r_time, r_INT, r_disDense, r_disT = 5., 5., 0., 0. # for non-scaled Rdis, 10 is not required but might be nice for underflow

        ######
        
        rP_time, rP_INT, rP_dis, rP_disT = 0., 0., 0., 0.
        rE_time, rE_INT, rE_dis, rE_disT = 0., 0., 0., 0.
        
        ## time 
        #'''
        # DENSE SETUP
        r_time, r_INT, r_disDense, r_disT = 5., 5., 10., 10. # for non-scaled Rdis, 10 is not required but might be nice for underflow
        
        rP_time = -1*(self.t_idx/self.t_idx_limit)*done_bool # when episode is done, add time taken 
        # we do this to distinguish between episodes with gamma = 1
        #rE_time =  0#1*timestep_factor # uniform time reward
        '''
        # ZERO SUM SETUP
        r_time, r_INT, r_disDense, r_disT = 5., 5., 0., 0. # for non-scaled Rdis, 10 is not required but might be nice for underflow

        rP_time = -1*timestep_factor
        rE_time =  1*timestep_factor #
        
        #'''
        ## terminal
        rP_INT =  1*outcomes_t['I']
        rE_INT =  -1*outcomes_t['I'] 

        ## distance
        
        ## DISTANCE REWARDS

        func_d_scaler = self.func_d_scaler2 # USED BY BOTH r_disT  &  r_disDense
        
        if r_disT and done_bool:
            #pass
            #'''
            dis_all = self.state_space[:(self.t_idx+1),:,0,:3]
            dis_all = np.linalg.norm(dis_all[:,aidxE,:]-dis_all[:,aidxP,:], axis = -1)
            dis_all = np.maximum(dis_all-self.intercept_distance,0.) # relative to interception & > 0
            
            dis_all_scaled = func_d_scaler(dis_all)/100. # all values in range [0,~ 1]
             
            
            rP_disT = np.mean(-1.*dis_all_scaled) # [0,~ -1] bounded, sort of integral, notice -1 for P
            # notice no rE_dis reward
            #'''
        if r_disDense:
            ## NOTICE ALSO!! ACTIVE AT INTERCEPTION
            
            ### NON-ZERO SUM BUT AGENT SPECIFIC
            #'''
            ## compute

            tprev_idx = max(0,(self.t_idx-1))
            dis_t_clip = max(dis_t-self.intercept_distance,0.)
            dis_t_scaled =  func_d_scaler(dis_t_clip) 
            
            
            ## attribute
            #dis_rP_scaled = -1*(dis_t_scaled)*self.t_delta*r_disDense # <= 0, to promote early termination
            # times delta to represent the integra
            
            '''
            dis_rE_scaled =    (dis_t_scaled)*r_disDense # >= 0, to discourage early termination
            #'''
            
            dis_tE_delta =  np.linalg.norm(self.state_space[self.t_idx,aidxE,0,:3] \
                                        - self.state_space[tprev_idx,aidxE,0,:3]) # distance traversed by E since last step

            dis_tprevE = np.linalg.norm(self.state_space[self.t_idx,aidxP,0,:3] \
                                        - self.state_space[tprev_idx,aidxE,0,:3]) # current P, previous E position
            #dis_tE_clip = min(dis_tprevE-self.intercept_distance,1.0)
            #dis_tE_clip = func_d_scaler(dis_t_clip) 
            dis_tprevE_clip = max(dis_tprevE-self.intercept_distance,0.)#,1.0) # notice upper bound
            dis_tprevE_scaled =  func_d_scaler(dis_tprevE_clip + outcomes_t['I']*dis_tE_delta*1)
            # we add dis_tE_delta during intercept to penalize interception in a scaled manner
            # then it is as if the evader moved at least dis_tE_delta towards the evader
            # also note that this works in case both were already 0  (due to clip)
            
            dis_rE_scaled = ((dis_t_scaled-dis_tprevE_scaled)/(dis_tE_delta + 1e-5))*(not outcomes_t['I'])  #- outcomes_t['I']*10. + timestep_factor*5. # increase positive
            # the gradient is determined by the scaling function, but we stabilize to ensure vnorm -> 0 is not promoted
            # NOTE THAT USING THE DIFERENCE IN REWARD VIOLATES MDP; because this reward now depends on current and previous state i.e. R(s_t & s_t-1) rather than R(s_t)
            #rescale = 0.5*1e3 # empirically observed scale at 4-6*1e4
            rescale = 100#*1e4
            rE_dis = dis_rE_scaled/rescale # increase positive
            
            
            #rP_dis = -1.*(dis_t_scaled/100.)*timestep_factor*(not outcomes_t['I']) # all values in range [0,~ 1] combined forms sort of integral, notice -1 for P
            #rP_dis = -1.*(dis_t_scaled/100.)*(not outcomes_t['I']) # all values in range [0,~ 1] combined forms sort of integral, notice -1 for P

            
            # this non 'difference in state' reward aligns with the markov property where the current state reward holds all information
            # old -- notice no rP_dis reward--
            
        ##
        reward_parts = {}
        reward_parts['p0'] = {'Rtime':rP_time,
                             'Rint':rP_INT,
                             'Rdis':rP_dis,
                             'RdisT':rP_disT,
                        }.copy() # non scaled reward components, so always included even if weight = 0!
        REWARD_P += rP_time*r_time + rP_INT*r_INT + rP_dis*r_disDense + rP_disT*r_disT
        
        reward_parts['e0'] = {'Rtime':rE_time,
                             'Rint':rE_INT,
                             'Rdis':rE_dis,
                             'RdisT':rE_disT,
                        }.copy() # non scaled reward components, so always included even if weight = 0!
        REWARD_E += rE_time*r_time + rE_INT*r_INT + rE_dis*r_disDense + rE_disT*r_disT
        
        return REWARD_P, REWARD_E, reward_parts
    
    def zerosum_reward_main_OLDv3(self, scores_t, outcomes_t):
        '''
        TODO CONNECT THIS TO AGENT SPECIFIC FUNCTIONS LIKE OBSERVATIN FUNCS
        
        NOTE THAT THE SCORES & OUTCOMES ARE ASSUMED TO BE FOR CURRENT TIMESTEP
        '''
        outcomes_t_noCopy = outcomes_t
        outcomes_t = outcomes_t.copy()
        scores_t = scores_t.copy()
        
        ##
        scores_game_t = scores_t['game'].copy()
        dis_t = scores_game_t['dis']
        gamma_t = scores_game_t['Gamma']
        gamma_uV_t = scores_game_t['Gamma_uV']
        # TODO MAKE THIS AGENT SPEICIF LIKE THE OBSERVATION FUNC
        
        timestep_factor = self.reward_MaxStepScaler # =1/(t_max/t_delta), to make reward timestep amount invariant
        # TODO IN ORDER TO MAKE REWARD SYSTEM TIMESTEP INVARIANT (Game of drones source)
        # TODO REWARD PER TIMESTEP WILL CHANGE ACROSS LEVELS!
        
        aidxP, aidxE = self.agent_specs['p0']['idx'], self.agent_specs['e0']['idx']
        # aaaaaaaaaaaa
        
        REWARD_P, REWARD_E = 0., 0.
        
        #r_time, r_INT, r_disDense, r_disT = 10., 10., 5.*timestep_factor, 0., # for minmax scaled Rdis
        r_time, r_INT, r_disDense, r_disT = 0., 0., 1., 0., # for non-scaled Rdis, 10 is not required but might be nice for underflow
        ######
        
        rP_time, rP_INT, rP_dis = 0., 0., 0.
        rE_time, rE_INT, rE_dis = 0., 0., 0.
        
        ## time 
        rP_time = -1*timestep_factor # uniform time reward
        rE_time =  1*timestep_factor # uniform time reward

        ## terminal
        rP_INT =  1*outcomes_t['I']
        rE_INT = -1*outcomes_t['I'] 

        ## distance
        
        ## DISTANCE REWARDS
        #''' 
        # TERMINAL DISTANCE REWARD
        if r_disT:
            pass
            '''
            dis_all = self.state_space[:(self.t_idx+1),:,0,:3]
            dis_all = np.linalg.norm(dis_all[:,aidxE,:]-dis_all[:,aidxP,:], axis = -1)
            dis_all = np.maximum(dis_all-self.intercept_distance,0.) # relative to interception & > 0
            
            distance_trunc = np.sum(np.log((dis_all**disT_Flogd+1e-4)) + dis_all**disT_Fd)
            
            
            rP_dis = -1*outcomes_t['I']*distance_trunc*(self.t_idx/self.t_idx_limit)*r_disT
            rP_dis = -1*outcomes_t['T']*distance_trunc*timestep_factor*r_disT # TODO divide by time?
            #'''
        # DENSE DISTANCE REWARD
        if False: #r_disDense and not outcomes_t['I'] and not reset:
            ## NOTICE NOT ACTIVE AT INTERCEPTION
            raise
            '''
            ### ZERO SUM BUT NOT AGENT SPECIFIC
            tprev_idx = max(0,(self.t_idx-1))
            dis_tprev = np.linalg.norm(self.state_space[tprev_idx,1,0,:3] \
                                        - self.state_space[tprev_idx,0,0,:3])
                
                
            dis_t_LogLinear =  np.log((np.maximum(dis_t-self.intercept_distance,0.)**disT_Flogd+1e-4)) + dis_t**disT_Fd
            dis_tprev_LogLinear =  np.log((np.maximum(dis_tprev-self.intercept_distance,0.)**disT_Flogd+1e-4)) + dis_tprev**disT_Fd
            
            REWARD_P += -1*(dis_t_LogLinear-dis_tprev_LogLinear)*r_disDense
            
            #''' 
            ### NON-ZERO SUM BUT AGENT SPECIFIC
            #'''
            ## compute
            tprev_idx = max(0,(self.t_idx-1))
            # dis_t is already determined and trimpoint
            
            dis_tprevP = np.linalg.norm(self.state_space[self.t_idx,aidxE,0,:3] \
                                        - self.state_space[tprev_idx,aidxP,0,:3]) # current E, previous P position
            dis_tP_delta =  np.linalg.norm(self.state_space[self.t_idx,aidxP,0,:3] \
                                        - self.state_space[tprev_idx,aidxP,0,:3]) # distance traversed by P since last step
                
            dis_tprevE = np.linalg.norm(self.state_space[self.t_idx,aidxP,0,:3] \
                                        - self.state_space[tprev_idx,aidxE,0,:3]) # current P, previous E position
            dis_tE_delta =  np.linalg.norm(self.state_space[self.t_idx,aidxE,0,:3] \
                                        - self.state_space[tprev_idx,aidxE,0,:3]) # distance traversed by E since last step
            ## Min max determination & clip 
            dis_t_clip = max(dis_t-self.intercept_distance,0.)
            
            dis_tP_min = max(dis_tprevP - dis_tP_delta - self.intercept_distance, 0.) # note that this one can undershoot the range (i.e. intercept), but clip handels it
            dis_tP_max = max(dis_tprevP + dis_tP_delta - self.intercept_distance, 0.) # clipping not needed, but for consistency
               
            dis_tE_min = max(dis_tprevE - dis_tE_delta - self.intercept_distance, 0.) # note that this one can undershoot the range (i.e. intercept), but clip handels it
            dis_tE_max = max(dis_tprevE + dis_tE_delta - self.intercept_distance, 0.) # clipping not needed, but for consistency
            
            ##
            step_size = 0.1 # note stepsize /2 due to [-1,1] scale of minmax
            zP_MinMax = (dis_t_clip - dis_tP_min)/(dis_tP_max - dis_tP_min) # [0,1]
            zP_MinMax = max(0.,zP_MinMax-step_size) # [0, 1-step_size], for pseudo best distance of P
            dis_tP_min_zMinMax = zP_MinMax*(dis_tP_max - dis_tP_min) + dis_tP_min # reverse minmax scaled ton get pseudo-best
            
            zE_MinMax = (dis_t_clip - dis_tE_min)/(dis_tE_max - dis_tE_min) # [0,1]
            zE_MinMax = max(0.,zE_MinMax-step_size) # [0, -1+step_size, 1], for pseudo worst distance of E
            dis_tE_min_zMinMax = zE_MinMax*(dis_tE_max - dis_tE_min) + dis_tE_min # reverse minmax scaled to get psuedo-worst
            
            ## scale
            # NOTE this definition already relativizes wrt to velocity as its all based on ideal and actual distance change
            func_d_scaler = self.func_d_scaler1
            
            dis_t_scaled =  func_d_scaler(dis_t_clip)
            
            #dis_tP_min_scaled =  func_d_scaler(dis_tP_min)
            #dis_tP_max_scaled =  func_d_scaler(dis_tP_max)
            
            #dis_tE_min_scaled =  func_d_scaler(dis_tE_min) 
            #dis_tE_max_scaled =  func_d_scaler(dis_tE_max) 
            
            dis_tP_min_Zscaled =  func_d_scaler(dis_tP_min_zMinMax) 
            dis_tE_min_Zscaled =  func_d_scaler(dis_tE_min_zMinMax) 
            
            #'''
            # non-scaled version (difference), note that perfect score is discussed 
            # Pursuer
            # NOTE this does not have to be scaled due to e.g. velocity but wrt to other reward components***
            #dis_rP_scaled = (dis_tP_min_scaled - dis_t_scaled) # (best_P - true_P) <= 0, since Best_P <= True_P (i.e. getting closer)
            dis_rP_scaled = (dis_tP_min_Zscaled - dis_t_scaled) # minmax rescaled version
            # implementing (maximize) perfectly means sum(r) = 0, but this also promotes early truncation through interception (= desired for P)
            
            # Evader
            # dis_rE_scaled = (dis_tE_max_scaled - dis_t_scaled) # (best_E - true_E) >= 0, since Best_E >= True_E (i.e. getting further)
            # dis_rE_scaled *= -1 # to flip domain and ensure bound <= 0 BUT NOT USEFUL BECAUSE,
            # -> implementing perfectly means sum(r) = 0, but this also promotes early truncation through interception (= NOT desired for E)
            
            #dis_rE_scaled = -1*(dis_tE_min_scaled - dis_t_scaled) # -1*(worst_E - true_E) >= 0, since worst_E <= True_E (i.e. getting further)
            dis_rE_scaled = -1*(dis_tE_min_Zscaled - dis_t_scaled) # minmax rescaled version
            # -> implementing (maximize) perfectly means sum(r) > 0, which does not promote early truncation through interception (= desired for P)
            
            
            # NOTE THAT THIS WOULD REQUIRE HAND BASED SCALING!
            '''
            # minmax-scaled version [0,1], rescaled to  [-1,1], , note that perfect score is -1 & 1 for agents P & E (after [-1,1] rescaling)
            # NOTE minmax is easier to scale wrt to other reward functions  (also read ***)
            dis_rP_scaled = -1*((dis_t_scaled - dis_tP_min_scaled)/(dis_tP_max_scaled-dis_tP_min_scaled)-0.5)*2. # -1*(true - min)/(max-min)
            dis_rE_scaled =  1*((dis_t_scaled - dis_tE_min_scaled)/(dis_tE_max_scaled-dis_tE_min_scaled + self.offline*1e-4)-0.5)*2. #    (true - min)/(max-min)
            
            #'''
            ## attribute
            rP_dis = dis_rP_scaled # reduction positive
            rE_dis = dis_rE_scaled # increase positive
            #'''
            
        if False: #r_disDense and not reset:
            ## NOTICE ALSO!! ACTIVE AT INTERCEPTION
            
            ### NON-ZERO SUM BUT AGENT SPECIFIC
            ## compute
            tprev_idx = max(0,(self.t_idx-1))
            # dis_t is already determined and trimpoint
            
            dis_tprevP = np.linalg.norm(self.state_space[self.t_idx,aidxE,0,:3] \
                                        - self.state_space[tprev_idx,aidxP,0,:3]) # current E, previous P position
            #dis_tP_delta =  np.linalg.norm(self.state_space[self.t_idx,aidxP,0,:3] \
            #                            - self.state_space[tprev_idx,aidxP,0,:3]) # distance traversed by P since last step
                
            dis_tprevE = np.linalg.norm(self.state_space[self.t_idx,aidxP,0,:3] \
                                        - self.state_space[tprev_idx,aidxE,0,:3]) # current P, previous E position
            #dis_tE_delta =  np.linalg.norm(self.state_space[self.t_idx,aidxE,0,:3] \
            #                            - self.state_space[tprev_idx,aidxE,0,:3]) # distance traversed by E since last step
                
            dis_t_clip = max(dis_t-self.intercept_distance,0.)
            dis_tprevP_clip = max(dis_tprevP-self.intercept_distance,0.)
            dis_tprevE_clip = max(dis_tprevE-self.intercept_distance,0.)
            
            
            func_d_scaler = self.func_d_scaler2
            
            dis_t_scaled =  func_d_scaler(dis_t_clip) 
            dis_tprevP_scaled =  func_d_scaler(dis_tprevP_clip) 
            dis_tprevE_scaled =  func_d_scaler(dis_tprevE_clip) 
            
            
            ## attribute
            dis_rP_scaled = -1*(dis_t_scaled-dis_tprevP_scaled)*r_disDense # reduction positive
            dis_rE_scaled =    (dis_t_scaled-dis_tprevE_scaled)*r_disDense # increase positive
            ## attribute
            rP_dis = dis_rP_scaled # reduction positive
            rE_dis = dis_rE_scaled # increase positive

            
        if r_disDense and not reset:
            ## NOTICE ALSO!! ACTIVE AT INTERCEPTION
            
            ### NON-ZERO SUM BUT AGENT SPECIFIC
            #'''
            ## compute
            tprev_idx = max(0,(self.t_idx-1))

            dis_t_clip = max(dis_t-self.intercept_distance,0.)

            func_d_scaler = self.func_d_scaler2
            
            dis_t_scaled =  func_d_scaler(dis_t_clip) 
            
            
            ## attribute
            dis_rP_scaled = -1*(dis_t_scaled)*self.t_delta*r_disDense # <= 0, to promote early termination
            # times delta to represent the integra
            
            '''
            dis_rE_scaled =    (dis_t_scaled)*r_disDense # >= 0, to discourage early termination
            #'''
            dis_tprevE = np.linalg.norm(self.state_space[self.t_idx,aidxP,0,:3] \
                                        - self.state_space[tprev_idx,aidxE,0,:3]) # current P, previous E position
            dis_tprevE_clip = max(dis_tprevE-self.intercept_distance,0.)
            dis_tprevE_scaled =  func_d_scaler(dis_tprevE_clip)
            
            dis_rE_scaled =    (dis_t_scaled-dis_tprevE_scaled)*r_disDense  #- outcomes_t['I']*10. + timestep_factor*5. # increase positive
            #'''
            ## attribute
            rP_dis = dis_rP_scaled # reduction positive
            rE_dis = dis_rE_scaled # increase positive

            #'''
            
        ##
        reward_parts = {}
        reward_parts['p0'] = {'rTIME':rP_time,
                             'rINT':rP_INT,
                             'rDIS':rP_dis,
                        } # non scaled reward components, so always included even if weight = 0!
        REWARD_P += rP_time*r_time + rP_INT*r_INT + rP_dis*r_disDense
        
        reward_parts['e0'] = {'rTIME':rE_time,
                             'rINT':rE_INT,
                             'rDIS':rE_dis,
                        } # non scaled reward components, so always included even if weight = 0!
        REWARD_E += rE_time*r_time + rE_INT*r_INT + rE_dis*r_disDense
        
        return REWARD_P, REWARD_E, reward_parts
    
    
    def zerosum_reward_main_OLDv2(self, scores_t, outcomes_t):
        '''
        TODO CONNECT THIS TO AGENT SPECIFIC FUNCTIONS LIKE OBSERVATIN FUNCS
        
        NOTE THAT THE SCORES & OUTCOMES ARE ASSUMED TO BE FOR CURRENT TIMESTEP
        '''
        outcomes_t_noCopy = outcomes_t
        outcomes_t = outcomes_t.copy()
        scores_t = scores_t.copy()
        
        ##
        scores_game_t = scores_t['game'].copy()
        dis_t = scores_game_t['dis']
        gamma_t = scores_game_t['Gamma']
        gamma_uV_t = scores_game_t['Gamma_uV']
        # TODO MAKE THIS AGENT SPEICIF LIKE THE OBSERVATION FUNC
        
        timestep_factor = self.reward_MaxStepScaler # =1/(t_max/t_delta), to make reward timestep amount invariant
        # TODO IN ORDER TO MAKE REWARD SYSTEM TIMESTEP INVARIANT (Game of drones source)
        # TODO REWARD PER TIMESTEP WILL CHANGE ACROSS LEVELS!
        
        aidxP, aidxE = self.agent_specs['p0']['idx'], self.agent_specs['e0']['idx']
        # aaaaaaaaaaaa
        
        REWARD_P, REWARD_E = 0., 0.
        
        r_time, r_INT, r_disDense, r_disT = 10., 10., 1., 0., 
        
        # time 
        REWARD_P += -1*timestep_factor*r_time # uniform time reward
        REWARD_E +=  1*timestep_factor*r_time # uniform time reward

        # terminal
        REWARD_P +=    outcomes_t['I'] * r_INT 
        REWARD_E += -1*outcomes_t['I'] * r_INT 

        # distance
        disT_Flogd, disT_Fd = 2., 2. # exponent factor for log(dis**) and dis**
        
        # DISTANCE REWARDS
        #''' 
        # TERMINAL DISTANCE REWARD
        if r_disT:
            dis_all = self.state_space[:(self.t_idx+1),:,0,:3]
            dis_all = np.linalg.norm(dis_all[:,1,:]-dis_all[:,0,:], axis = -1)
            dis_all = np.maximum(dis_all-self.intercept_distance,0.) # relative to interception & > 0
            
            distance_trunc = np.sum(np.log((dis_all**disT_Flogd+1e-4)) + dis_all**disT_Fd)
            
            
            REWARD_P += -1*outcomes_t['I']*distance_trunc*(self.t_idx/self.t_idx_limit)*r_disT
            REWARD_P += -1*outcomes_t['T']*distance_trunc*timestep_factor*r_disT # TODO divide by time?

        # DENSE DISTANCE REWARD
        if r_disDense and not outcomes_t['I']:
            # NOTICE NOT ACTIVE AT INTERCEPTION
            '''
            ### ZERO SUM BUT NOT AGENT SPECIFIC
            tprev_idx = max(0,(self.t_idx-1))
            dis_tprev = np.linalg.norm(self.state_space[tprev_idx,1,0,:3] \
                                        - self.state_space[tprev_idx,0,0,:3])
                
                
            dis_t_LogLinear =  np.log((np.maximum(dis_t-self.intercept_distance,0.)**disT_Flogd+1e-4)) + dis_t**disT_Fd
            dis_tprev_LogLinear =  np.log((np.maximum(dis_tprev-self.intercept_distance,0.)**disT_Flogd+1e-4)) + dis_tprev**disT_Fd
            
            REWARD_P += -1*(dis_t_LogLinear-dis_tprev_LogLinear)*r_disDense
            
            #''' 
            ### NON-ZERO SUM BUT AGENT SPECIFIC
            #'''
            ## compute
            v_norm_P = np.linalg.norm(self.state_space[self.t_idx,0,1,:3])
            v_norm_E = np.linalg.norm(self.state_space[self.t_idx,1,1,:3])
            
            #dis_scalerP, dis_scalerE = 1/(9.*1.35), 1/(3.*1.35) #1/timestep_factor #100. 
            #dis_scalerP, dis_scalerE = 1/(abs(v_norm_P)**(1.5) + 1e-2), 1/(abs(v_norm_E)**(1.5) + 1e-2)*1/4
            
            dis_scalerP, dis_scalerE = 1/8., 1.
            
            tprev_idx = max(0,(self.t_idx-1))
            # dis_t is already determined
            
            dis_tprevP = np.linalg.norm(self.state_space[self.t_idx,1,0,:3] \
                                        - self.state_space[tprev_idx,0,0,:3]) # current E, previous P position
                
            dis_tprevE = np.linalg.norm(self.state_space[tprev_idx,1,0,:3] \
                                        - self.state_space[self.t_idx,0,0,:3]) # current P, previous E position
            
            ## clip
            dis_t_clip = max(dis_t-self.intercept_distance,0.)
            dis_tprevP_clip = max(dis_tprevP-self.intercept_distance,0.)
            dis_tprevE_clip = max(dis_tprevE-self.intercept_distance,0.)
            
            ## scale
            stable = 1e-4 
            dis_t_scaled =  np.log((dis_t_clip**disT_Flogd+stable)) + dis_t_clip**disT_Fd
            dis_tprevP_scaled =  np.log((dis_tprevP_clip**disT_Flogd+stable)) + dis_tprevP_clip**disT_Fd
            dis_tprevE_scaled =  np.log((dis_tprevE_clip**disT_Flogd+stable)) + dis_tprevE_clip**disT_Fd
            
            ## attribute
            REWARD_P += -1*(dis_t_scaled-dis_tprevP_scaled)*r_disDense*dis_scalerP # reduction positive
            REWARD_E +=    (dis_t_scaled-dis_tprevE_scaled)*r_disDense*dis_scalerE # increase positive
        return REWARD_P, REWARD_E 
    
            
            
    def zerosum_reward_main_OLDv1(self, scores_t, outcomes_t):
        '''
        TODO CONNECT THIS TO AGENT SPECIFIC FUNCTIONS LIKE OBSERVATIN FUNCS
        
        NOTE THAT THE SCORES & OUTCOMES ARE ASSUMED TO BE FOR CURRENT TIMESTEP
        '''
        outcomes_t_noCopy = outcomes_t
        outcomes_t = outcomes_t.copy()
        scores_t = scores_t.copy()
        
        ##
        scores_game_t = scores_t['game'].copy()
        dis_t = scores_game_t['dis']
        gamma_t = scores_game_t['Gamma']
        gamma_uV_t = scores_game_t['Gamma_uV']
        # TODO MAKE THIS AGENT SPEICIF LIKE THE OBSERVATION FUNC
        
        timestep_factor = self.reward_MaxStepScaler # =1/(t_max/t_delta), to make reward timestep amount invariant
        # TODO IN ORDER TO MAKE REWARD SYSTEM TIMESTEP INVARIANT (Game of drones source)
        # TODO REWARD PER TIMESTEP WILL CHANGE ACROSS LEVELS!
        
        
        REWARD_P, REWARD_E = 0., 0.
        ZEROSUM_BOOL = True
        ## shaping
        #REWARD_P += -1*(((dis2_t-dis2_prev)/dis2_prev)*100.)*timestep_factor 
        if self.CL_START_DIS_REWARD:
            if self.CL_taskLevel <= 2: # NOTICE 5!
                ZEROSUM_BOOL = False
                ''' 
                IN THIS CASE WE NEVER GO UP A CL LEVEL BECAUSE THIS REWARD DOES NOT SCALE LIKE EXPECTED
                '''
                ##
                ''' 
                t_prev = max(0,(self.t_idx-1))
                dis_t_prev = np.linalg.norm(self.state_space[t_prev,1,0,:3] \
                                            - self.state_space[t_prev,0,0,:3])
                # scale by previous distance
                #dis_scaler = dis_t_prev
                
                # scale by velocity norm
                vnormDiff_t_prev = np.linalg.norm(self.state_space[t_prev,1,1,:3] \
                                            - self.state_space[t_prev,0,1,:3])
                dis_scaler = vnormDiff_t_prev*self.t_delta

                
                REWARD_P += -1*((dis_t-dis_t_prev)/dis_scaler)*timestep_factor*5.

                #REWARD_P += -1*((dis2_t-dis2_prev)/dis2_prev) 
                #'''
                
                # AAAAAAA
                if self.SELFSPIEL:
                    r_z, r_crash, r_Vdownward, r_Zdownward, r_gammapp, r_thrustpp, r_gamma, r_Q = \
                        0., 0., 0., 0., 10., 0., 0., 0.#5. # no sign!
                    r_dis, r_time, r_T, r_Idis, r_I = 5., 0., 0., 5., 10.  # no sign!
                
                    
                else:
                    ''' 
                    r_z, r_crash, r_Vdownward, r_Zdownward, r_gammapp, r_thrustpp, r_gamma, r_Q  = \
                        0., 0., 5., 0., 5., 0., 0., 0. #5. # no sign!
                    r_dis, r_time, r_I = 5., 5., 10.  # no sign!
                    ''' 
                    r_z, r_crash, r_Vdownward, r_Zdownward, r_gammapp, r_thrustpp, r_gamma, r_Q = \
                        0., 0., 0., 0., 0., 0., 0., 0. #5. # no sign!
                    r_dis, r_time, r_Trunc, r_Idis, r_I = 1., 5., 10., 0., 0.  # no sign!
                    
                    '''
                    if outcomes_t['T']:
                        r_disSigmaT = 5.
                        
                        disSigma = self.state_space[:self.t_idx,:,0,:3]
                        disSigma = np.linalg.norm(disSigma[:,1,:]-disSigma[:,0,:], axis = -1)/self.r0_e0
                        disSigma = np.mean(np.power(disSigma,2))
                        
                        Rsigma = disSigma*r_disSigmaT
                        
                        REWARD_P += -1*Rsigma*outcomes_t['T']
                        REWARD_E += Rsigma*outcomes_t['T']
                    '''
                    #r_dis, r_time, r_Trunc, r_Idis, r_I = 10., 100., 10., 100., 100.  # no sign!
                    #''' 
                
                
                Q_P = R.from_euler('xyz', 
                                 self.state_space[self.t_idx,0,0,3:],
                                 degrees = False).as_quat(canonical = True)
                
                Q_E = R.from_euler('xyz', 
                                 self.state_space[self.t_idx,1,0,3:],
                                 degrees = False).as_quat(canonical = True)
                #
                '''
                # LEARN2FLY = False
    
                v_p_t = np.linalg.norm(self.state_space[self.t_idx,0,1,:3])
                L2F_reward = -1*(dis_t**2)*0.01 + -1*(v_p_t**2)*0.01 + -1*(1-(Q_P[-1]**2))
                REWARD_P += L2F_reward
                #''' 
                #

                z_min = -10.
                ## PURSUER
                # distance
  
                t_prev = max(0,(self.t_idx-1))
                #distance_trunc = float(dis_t/self.r0_e0)*float(self.min_distance/self.r0_e0) #self.min_distance/self.r0_e0
                #distance_trunc = float(dis_t/self.r0_e0)*float(self.min_distance/self.r0_e0)
                distance_trunc = float(dis_t/self.r0_e0)*float(self.min_distance/self.r0_e0)
                #'''
                dis_t_prev = np.linalg.norm(self.state_space[t_prev,1,0,:3] \
                                            - self.state_space[t_prev,0,0,:3])
                '''
                dis_t_prev = float(self.min_distance)
                #'''
                #vnormDiff_t_prev = np.linalg.norm(self.state_space[t_prev,1,1,:3] \
                #                            - self.state_space[t_prev,0,1,:3])
                dis_scaler = 1. #float(self.r0_e0)#dis_t_prev #vnormDiff_t_prev*self.t_delta + 1e-8

                distance_dense = (dis_t-dis_t_prev)/dis_scaler
                #distance_dense = float(dis_t/self.r0_e0)*timestep_factor
                #distance_dense = (dis_t-self.min_distance)/dis_scaler
                #distance_dense = float(dis_t/self.r0_e0)*float(self.min_distance/self.r0_e0)*timestep_factor
                
                #z_bias = self.r0_e_range[-1] * 1.5
                z_bias = min(10.,self.r0_e0 * 2.5) # keep positive
                
                #RoR_time = max(1 - 0.5*(1/(0.5*1/timestep_factor))*(self.t_idx/self.t_idx_limit)*(self.t_idx/self.t_idx_limit), 0.) # Total-first_triangle_area
                RoR_time = max((1.-self.t_idx/self.t_idx_limit),0.) # uniform
                RoR_dis = dis_t # always an extra factor to avoid improvement here
                '''
                COMMENTS ON DIS_SCALER = vnormDiff_t_prev
                if the velocity of the evader is constant and you do something
                completely different, then your positive reward is lost 
                
                PROBLEM; IT OFFERS A WAY TO REDUCE NEGATIVE REWARD, BY
                INVOKING VERY HIGH DELTA_V_NORM
                
                TODO; potentially scale this; i think that this might
                not lead to goal discovery (= interception); because it
                will likely speedup initially and get high reward for large
                gain, then speeddown
                
                DO SHAPING REWARD; IF YOU ALIGN WITH THE TARGET YOUR ORIENTATION!
                ''' 
                
                #REWARD_P += -1*((dis_t-dis_t_prev)/dis_scaler)*timestep_factor*r_dis
                REWARD_P += -1*distance_dense*r_dis
                

                #REWARD_P += -1*((dis_t-dis_t_prev))*timestep_factor*r_dis
                #REWARD_P += -1*((dis_t-dis_t_prev)/self.r0_e0)*timestep_factor*r_dis
                #REWARD_P += -1*((dis_t))*timestep_factor*r_dis
                
                #REWARD_P += -1*(np.mean(self.errors_last['p0'][-2:]**2))*timestep_factor*r_dis
                
                # stability related
                stable_Zdownward_P = scores_t['p0']['stable_Zdownward'] # pushes Z to -1
                stable_Zdownward_P = -1*(min(-1*(stable_Zdownward_P+0.65),0)**2)*1/2.7225 # [-1,0]
                #stable_Zdownward_P = -1*(min(-1*(stable_Zdownward_P+0.85),0)**2) # [-1,0] NOT SCALED!!!
                REWARD_P += stable_Zdownward_P*timestep_factor*r_Zdownward
                
                stable_Vdownward_P = scores_t['p0']['stable_Vdownward'] # pushed V_z away from 1 
                stable_Vdownward_P = -1*(min(2*(stable_Vdownward_P+0.5),0)**2) # [-1,0]
                #stable_Vdownward_P = -1*(min(2*(stable_Vdownward_P+0.),0)**2) # [-1,0] NOT SCALED!!!
                REWARD_P += stable_Vdownward_P*timestep_factor*r_Vdownward
                
                
                stable_Q_P = -1*(1-(Q_P[-1]**2))
                REWARD_P += stable_Q_P*timestep_factor*r_Q
                
                
                # crashing 
                crash_P = (np.min(self.state_space[:,0,0,2]) < z_min) # z < z_min
                REWARD_P += crash_P*-1*r_crash
                
                REWARD_P += -1*(min((self.state_space[self.t_idx,0,0,2]+z_bias)*10., 0.)**2)*r_z
                
                # objective related
                REWARD_P += scores_t['p0']['Gamma_pp']*timestep_factor*r_gammapp
                REWARD_P += -1.*gamma_t*timestep_factor*r_gamma
                REWARD_P += scores_t['p0']['thrust_pp']*timestep_factor*r_thrustpp
                REWARD_P += outcomes_t['I'] * r_I - outcomes_t['I']*dis_t*r_Idis
                REWARD_P += - outcomes_t['T']*distance_trunc*r_Trunc
                #REWARD_P += outcomes_t['E'] * -self.reward_CL_taskLevel  # NOTICE -1
                
                #REWARD_P += -1*(1/(0.5*1/timestep_factor))*(self.t_idx/self.t_idx_limit)*r_time 
                REWARD_P += -1*timestep_factor*r_time # uniform time reward
                
                ## EVADER
                # stability related
                stable_Zdownward_E = scores_t['e0']['stable_Zdownward'] # pushes Z to -1
                stable_Zdownward_E = -1*(min(-1*(stable_Zdownward_E+0.65),0)**2)*1/2.7225 # [-1,0]
                REWARD_E += stable_Zdownward_E*timestep_factor*r_Zdownward
                
                stable_Vdownward_E = scores_t['e0']['stable_Vdownward']  # pushed V_z away from 1 
                stable_Vdownward_E = -1*(min(2*(stable_Vdownward_E+0.5),0)**2) # [-1,0]
                REWARD_E += stable_Vdownward_E*timestep_factor*r_Vdownward
                
                stable_Q_E = -1*(1-(Q_E[-1]**2))
                REWARD_E += stable_Q_E*timestep_factor*r_Q
                
                # crashing 
                crash_E = (np.min(self.state_space[:,1,0,2]) < z_min) # z < z_min
                REWARD_E += crash_E*-1*r_crash
                
                REWARD_E += -1*(min((self.state_space[self.t_idx,1,0,2]+z_bias)*10., 0.)**2)*r_z
                
                # time related
                if self.SELFSPIEL:
                    # essentially E is a pursuer as well, thus 
                    #REWARD_E += -1*((dis_t-dis_t_prev)/dis_scaler)*timestep_factor*r_dis
                    REWARD_E += -1*distance_dense*r_dis
                    # objective related
                    REWARD_E += scores_t['e0']['Gamma_pp']*timestep_factor*r_gammapp # NOTICE 1
                    REWARD_E += -1.*gamma_t*timestep_factor*r_gamma
                    REWARD_E += scores_t['e0']['thrust_pp']*timestep_factor*r_thrustpp
                    REWARD_E += outcomes_t['I'] * r_I - outcomes_t['I']*dis_t*r_Idis # NOTICE 1
                    
                    #REWARD_E += -1*(1/(0.5*1/timestep_factor))*(self.t_idx/self.t_idx_limit)*r_time 
                    REWARD_E += -1*timestep_factor*r_time # uniform time reward
                    REWARD_E += - outcomes_t['T']*distance_trunc*r_Trunc
                else:
                    #REWARD_E += 1*((dis_t-dis_t_prev)/dis_scaler)*timestep_factor*r_dis
                    REWARD_E += 1*distance_dense*r_dis
                    # objective related
                    REWARD_E += -1*scores_t['e0']['Gamma_pp']*timestep_factor*r_gammapp # NOTICE -1
                    REWARD_E += gamma_t*timestep_factor*r_gamma
                    REWARD_E += -1*scores_t['e0']['thrust_pp']*timestep_factor*r_thrustpp
                    REWARD_E += outcomes_t['I'] * -1 * r_I + outcomes_t['I']*dis_t*r_Idis # NOTICE -1
                    
                    #REWARD_E += 1*(1/(0.5*1/timestep_factor))*(self.t_idx/self.t_idx_limit)*r_time 
                    REWARD_E += 1*timestep_factor*r_time # uniform time reward
                    REWARD_E += outcomes_t['T']*distance_trunc*r_Trunc
                ##
                #REWARD_P += -1.*gamma_t*timestep_factor*5. # 5 is coef here i.e. total reward
                
                #REWARD_P += -1.*gamma_uV_t*timestep_factor*1. #  low coef makes it fade over time
                #REWARD_P += -1.*(gamma_t**2)*10.*timestep_factor 
                #REWARD_P += -1*(((gamma_t-gamma_prev)/gamma_prev)*100.)*timestep_factor 

                # early truncation
                if self.TRUNC_EARLY:
                    ## check early truncation
                    P_Zlow = float(self.state_space[self.t_idx,0,0,2] - (-1*z_bias))
                    E_Zlow = float(self.state_space[self.t_idx,1,0,2] - (-1*z_bias))
                    
                    #trunc_early = ((P_Zlow < 0.) and (E_Zlow  < 0.))
                    trunc_early = (P_Zlow < 0.) 
                    outcomes_t_noCopy['T'] = (trunc_early or outcomes_t_noCopy['T']) # new truncation or old
                    
                    if trunc_early:
                        ## apply rest-of-reward
                        REWARD_P += -1.*(RoR_time*r_time) -1.*(RoR_dis*r_dis) + P_Zlow*(P_Zlow  < 0.)
                        
                        if self.SELFSPIEL:
                            REWARD_E += -1.*(RoR_time*r_time) -1.*(RoR_dis*r_dis) + E_Zlow*(E_Zlow  < 0.)
                        else:
                            REWARD_E += 0. + E_Zlow*(E_Zlow  < 0.) # you dont get (positive!) rewards (= net lower)
                
            else:
                REWARD_P += -1*timestep_factor*10. # uniform time reward
                #REWARD_P += -1*(1/(0.5*1/timestep_factor))*(self.t_idx/self.t_idx_limit)*5. # linear reward
        else:
            ## time 
            REWARD_P += -1*timestep_factor*10. # uniform time reward
            #REWARD_P += -1*(1/(0.5*1/timestep_factor))*(self.t_idx/self.t_idx_limit)*5. # linear time reward
        

        if ZEROSUM_BOOL:
            
            
            ## terminal
            REWARD_P += outcomes_t['I'] * 10. #self.reward_CL_taskLevel 
            REWARD_P += outcomes_t['E'] * 10. #-self.reward_CL_taskLevel # ESCAPED IS USED FOR TRUNCATION, NOT TERMINATION ATM
            
            '''
            distance_trunc = float(dis_t/self.r0_e0)*float(self.min_distance/self.r0_e0)
            #distance_trunc = float(self.min_distance/self.r0_e0)
            REWARD_P += -outcomes_t['T']*distance_trunc*5 # TODO divide by time?
            '''
            r_disDense, r_disT = 5., 0.

            disT_Flogd, disT_Fd = 2., 2. # factor
            
            ## DISTANCE REWARDS
            #''' 
            # TERMINAL DISTANCE REWARD
            if r_disT:
                dis_all = self.state_space[:(self.t_idx+1),:,0,:3]
                dis_all = np.linalg.norm(dis_all[:,1,:]-dis_all[:,0,:], axis = -1)
                dis_all = np.maximum(dis_all-self.intercept_distance,0.) # relative to interception & > 0
                
                distance_trunc = np.sum(np.log((dis_all**disT_Flogd+1e-4)) + dis_all**disT_Fd)
                
                
                REWARD_P += -1*outcomes_t['I']*distance_trunc*(self.t_idx/self.t_idx_limit)*r_disT
                REWARD_P += -1*outcomes_t['T']*distance_trunc*timestep_factor*r_disT # TODO divide by time?

            # DENSE DISTANCE REWARD
            if r_disDense:

                t_prev = max(0,(self.t_idx-1))
                dis_t_prev = np.linalg.norm(self.state_space[t_prev,1,0,:3] \
                                            - self.state_space[t_prev,0,0,:3])
                    
                    
                dis_t_LogLinear =  np.log((np.maximum(dis_t-self.intercept_distance,0.)**disT_Flogd+1e-4)) + dis_t**disT_Fd
                dis_tprev_LogLinear =  np.log((np.maximum(dis_t_prev-self.intercept_distance,0.)**disT_Flogd+1e-4)) + dis_t_prev**disT_Fd
                
                REWARD_P += -1*(dis_t_LogLinear-dis_tprev_LogLinear)*r_disDense
                

            
            # zerosum; so evader has opposite reward
            if self.SELFSPIEL:
                REWARD_E = REWARD_P 
            else:
                REWARD_E = -1.*REWARD_P 
        # note that gamma cannot be continuously improved, once youre on the pursuit manifold it should remain -1 
        
        #REWARD_P = -1*((dis2_t-dis2_prev)/dis2_prev) + intercepted * 10. + escaped * -10
        '''
        ^ rewards approach & intercept
        TODO make agent specific; & DEFINE ESCAPE SETTINGS
        
        IF YOU WANT TO KEEP ZERO-SUM GAME THEN THE REWARD_TOTAL = 0 ALWAYS!
        
        TODO MAKE ESCAPED AND INTERCEPTION DISTANCE SUBJECT TO RELATIVE VELO?
            -> and escaped also wrt to initial distance
            
        FINAL CHECKLIST REWARD DEFINITION;
        - check env_max timestep invariant for tractability across this config setting
        - check terminal reward also means termination (= single reward), otherwise it can be handed out multiple times
        - check terminal conditions and equation; e.g. isnt it always active? 
        is is ever active given config settings (e.g. min capture distance is impossible)? 
        
        - check scale of the dense reward; ensure its not greatly superior/inferior to terminal
        - check equation dense reward
        '''
        '''
        if self.SELFSPIEL:
            REWARD_E *= -1. # REWARD_P # same reward, since only a single agent (in pursuer's policy) 
        '''
        reward_parts = {}
        reward_parts['p0'] = {'Rtime':0,
                             'Rint':0,
                             'Rdis':0,
                             'RdisT':0,
                        }.copy() # non scaled reward components, so always included even if weight = 0!        
        reward_parts['e0'] = {'Rtime':0,
                             'Rint':0,
                             'Rdis':0,
                             'RdisT':0,
                          }.copy()    
        return REWARD_P, REWARD_E, reward_parts
        
    def reward_pursuer(self):
        '''
        NOTES
        - ALL REWARD SHOULD BE RELATIVE TO PREVIOUS STATE! i.e. no reward for
            doing nothing!
        - FIRST CALL TERMINATE FUNCS TO SEE IF WE NEED TERMINAL REWARD HERE
        
        IDEAS;
        - Ranos CCL divergence reward
        - CATD gamma reward
        - interception reward
        - minimal control input reward OR loss for invoking omega?
        - Time efficiency reward e.g. intercept+(all remaining time)
            - problem is that time to intercept depends on evader o.g. pos & heading
                - make it relative to initial conditions? must be possible
        
        TODO;
        - 
        
        FINAL CHECKLIST REWARD DEFINITION;
        - check env_max timestep invariant for tractability across this config setting
        - check terminal reward also means termination (= single reward), otherwise 
            it can be handed out multiple times
        - check terminal conditions and equation; e.g. isnt it always active? 
        is is ever active given config settings (e.g. min capture distance is impossible)? 
        
        - check scale of the dense reward; ensure its not greatly superior/inferior to terminal
        - check equation dense reward
        '''
        
        return 
    
    def reward_evader(self):
        '''
        NOTES
        - ALL REWARD SHOULD BE RELATIVE TO PREVIOUS STATE! i.e. no reward for
            doing nothing!
        - FIRST CALL TERMINATE FUNCS TO SEE IF WE NEED TERMINAL REWARD HERE
            
        IDEAS;
        - 'detection' related to CCL gamma or divergence from CATD gamma 
        - escape reward
            - based on distance between the two AND/OR how fast it escaped
        - minimal control input reward OR loss for invoking omega?
        - Time efficiency reward e.g. intercept+(all remaining time)
            - problem is that time to intercept depends on evader o.g. pos & heading
                - make it relative to initial conditions? must be possible
        TODO;
        - 
        
        
        FINAL CHECKLIST REWARD DEFINITION;
        - check env_max timestep invariant for tractability across this config setting
        - check terminal reward also means termination (= single reward), otherwise 
            it can be handed out multiple times
        - check terminal conditions and equation; e.g. isnt it always active? 
        is is ever active given config settings (e.g. min capture distance is impossible)? 
        
        - check scale of the dense reward; ensure its not greatly superior/inferior to terminal
        - check equation dense reward
        '''
        pass 
    
#%% TERMINATION FUNCTIONS

    def terminate_pursuer(self):
        
        if False:
            
            return True
        
        
        return False  
    
    def terminate_evader(self):
        pass
    
#%% UPDATE FUNCTION
    '''
    == Update state functions and dynamical system definitions ===
    '''
    #@latexify.function
    def TF_RotationRate(self, t, x_c, x, *, specs: dict):
        gain, tau, lim, ideal = specs['gain'], specs['tau'], specs['limit'], specs['ideal']
        
        ## transfer function & saturation
        #dx = ideal*x_c + (1.-ideal)*min(max((1/tau)*(gain*x_c-x), -lim), lim)
        t_delta = t - (self.t-self.t_delta) # t_current - t_start
        dx = ideal*x_c + \
            (1.-ideal)* min(max(gain*(1-np.exp(-1.*t_delta/tau))*x_c, -lim), lim)
        
        #print(f'XC = {x_c} -> DX = {dx} | t = {t_delta} | exp = {1-np.exp(-1.*t_delta/tau)}')
        # common step input time response, yet no consideration of 
        # initial condition
        '''
        NOTE THAT WE ASSUME HERE THAT INITIAL CONDITION = 0 (i.e. angle-dot = 0)
        which is not logical as previous timestep also applied this and the 
        rate logically has not returned to zero instanteously, consder
        the extreme case were we have e.g. dpsi_t-1 = -pi and now we want
        dpsi_t = pi; thats almost a twice difference.
        '''
            
        ''' 
        TODO FOR EFFICIENCY, CONSIDER DOING THIS IN MATRIX FORMAT?
        -> I DONT KNOW WHAT THE GAIN WOULD BE IF MAX STATES = 3
        ''' 
        return dx 

    #@latexify.function
    def EOM_2D1DOF_zoh(self,t,s, u):
        '''
        EOM for ideal 2-dimensional case and 1 DOF
        
        Dynamics:
            progression of system from angular perspective, where
            velocity vector always aligns with current heading (psi)
            
        Inputs:
            Zero-order-hold input of rotational rate (heading)
        
        Further details described in reset_state_space
        
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        x, y , psi = s
        psi_c, specs = u
        v = specs['v_rel']
        # zero-order hold implies this is input is constant over time interval
        
        dpsi = self.TF_RotationRate(t, psi_c, psi, specs = specs['psi'])
        ds = [v*np.cos(psi), v*np.sin(psi),dpsi] # dx,dy, dpsi, 
        return ds 
    
    def update_state_2D1DOF(self, u: dict):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:].copy() # (N,S), ground states
        state_prev = state_prev[:,[0,1,5]] # (N,3) select states afterwards, otherwise transpose!
        
        for aid in self.agent_id_stepped: # notice stepped!
            aid_idx = self.agent_specs[aid]['idx']
            v = self.agent_specs[aid]['v_rel'] # constant velocity
            #'''
            sol = sci.solve_ivp(self.EOM_2D1DOF_zoh, 
                                [float(self.t-self.t_delta), float(self.t)], 
                                state_prev[aid_idx,:],
                                args = ([*u[aid], self.agent_specs[aid]],), # must be tuple (inputs,)
                                method = "RK45")
            agent_state = sol.y[:,-1] # select final RK timestep
            '''
            sol = sci.odeint(self.EOM_2D1DOF_zoh, 
                                state_prev[aid_idx,:],
                                [float(self.t-self.t_delta), float(self.t)],  
                                args = ([*u[aid], self.agent_specs[aid]],), # must be tuple (inputs,)
                                tfirst = True)
            agent_state = sol[-1,:] # (S,)
            #''' 
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            ## handle & update states
            agent_state[2] = np.mod((agent_state[2] +np.pi),2.*np.pi) - np.pi
            # clip angles [-pi, pi]
            self.state_space[self.t_idx,aid_idx,0,[0,1,5]] = agent_state
            self.state_space[self.t_idx,aid_idx,1,[5]] = u[aid][0] # already update the inputted state (ideal)

            # align velocity
            self.state_space[self.t_idx,aid_idx,1,[0,1]] = \
                [np.cos(agent_state[2])*v,
                 np.sin(agent_state[2])*v] # body to inertial SEE self.ANGLES_SIGN
            
        
        ## Finalize & return 

        return 
        
    #@latexify.function
    def EOM_3D2DOF_zoh(self,t,s, u):
        ''' 
        
        EOM for ideal 3-dimensional case & 2 DOF
        
        Dynamics:
            progression of system from angular perspective with control 
            of pitch and yaw rate. Velocity vector alligns with body frame
            x-axis. Similar to missile.
            
            pitch and yaw are controlled as extension of 2d case, where
            only yaw was controlled. Therefore we dont opt for pitch
            and roll command which is more conventional.
            
        Inputs:
            Zero-order-hold input of rotational rates (euler).
        
        Rotation matrices
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ''' 
        ## unpack
        
        #'''
        x, y, z, phi, theta, psi =  s # states
        (theta_c, psi_c), specs = u # commanded inputs
        v = specs['v_rel']
        
        ## rotation matrix istance 
        RotMat = Rotations.body2inertial([phi, theta, psi])

        ## Dyanmics system definition
        # cartesian; from body to inertial
        dx, dy, dz = RotMat.apply([v,0.,0.]) # velocity aligns with body x-axis
        
        # angular (ideal atm)
        dphi =  0. # not controlled
        dtheta = self.TF_RotationRate(t, theta_c, theta, specs = specs['theta'])
        dpsi = self.TF_RotationRate(t, psi_c, psi, specs = specs['psi']) 

        
        ## Gather and return
        ds = [dx, dy, dz, dphi, dtheta, dpsi] 
        return ds 
    
    
    def update_state_3D2DOF(self, u):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:].copy() # (N,S), ground states

        for aid in self.agent_id_stepped: # notice stepped!
            aid_idx = self.agent_specs[aid]['idx']
            v = self.agent_specs[aid]['v_rel'] # constant velocity
            try:
                sol = sci.solve_ivp(self.EOM_3D2DOF_zoh, 
                                    [float(self.t-self.t_delta), float(self.t)], 
                                    state_prev[aid_idx,:],
                                    args = ([u[aid], self.agent_specs[aid]],), # must be tuple (inputs,)
                                    method = "RK45")
            except ValueError as e:
                # crashes when it pushes itself to inf
                msg = f'\nstateprev={state_prev[aid_idx,:]}\nu={u[aid]}'
                raise Exception(msg)
            agent_state = sol.y[:,-1] # (S,) select final RK timestep
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            ## handle & update states
            # clip pitch and modulus for phi & psi
            #agent_state[3] = ((agent_state[3] +0.5*np.pi) % (1.*np.pi)) - 0.5*np.pi
            #agent_state[3] = max(min(agent_state[3],0.5*np.pi), - 0.5*np.pi)
            agent_state[3] = ((agent_state[3] +np.pi) % (2.*np.pi)) - np.pi
            # OTHER DEFINITION?
            agent_state[4] = ((agent_state[4] +np.pi) % (2.*np.pi)) - np.pi
            #agent_state[4] = min(max(agent_state[4], -0.5*np.pi),0.5*np.pi)
            #agent_state[4] = ((agent_state[4] +0.5*np.pi) % (1.*np.pi)) - 0.5*np.pi
            agent_state[5] = ((agent_state[5] +np.pi) % (2.*np.pi)) - np.pi
            
            # clip angles [-pi, pi]
            self.state_space[self.t_idx,aid_idx,0,[0,1,2,3,4,5]] = agent_state
            self.state_space[self.t_idx,aid_idx,1,[4,5]] = u[aid] # already update the inputted state (ideal)

            # align velocity
            angles = self.state_space[self.t_idx,aid_idx, 0, 3:6]
            RotMat_agent = Rotations.body2inertial(angles) 
            self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = \
                RotMat_agent.apply([v,0.,0.])

        ## Finalize & return 

        return 
    
    
    #@latexify.function
    def EOM_3D3DOF_zoh(self,t,s, u):
        ''' 
        
        EOM for ideal 3-dimensional case & 3 DOF
        
        Dynamics:
            progression of system from angular perspective with control 
            of pitch, roll and yaw rate. Velocity vector alligns with body frame
            x-axis. Similar to missile.
            
            pitch and yaw are controlled as extension of 2d case, where
            only yaw was controlled. Therefore we dont opt for pitch
            and roll command which is more conventional.
            
        Inputs:
            Zero-order-hold input of rotational rates (euler).
        
        Rotation matrices
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ''' 
        ## unpack
        
        #'''
        x, y, z, phi, theta, psi =  s # states
        (phi_c, theta_c, psi_c), specs = u # commanded inputs
        v = specs['v_rel']
        # v = constant velocity

        
        ## rotation matrix istance 
        RotMat = Rotations.body2inertial([phi,theta, psi])
        
        ## Dyanmics system definition
        # cartesian; from body to inertial
        dx, dy, dz = RotMat.apply([v,0.,0.]) # velocity aligns with body x-axis
        
        # angular (ideal atm)
        '''
        dphi =  self.TF_RotationRate(t,phi_c, phi, specs = specs['phi'])  
        dtheta = self.TF_RotationRate(t,theta_c, theta, specs = specs['theta']) 
        dpsi = self.TF_RotationRate(t,psi_c, psi, specs = specs['psi'])  

        '''
        pqr = np.array([phi_c, theta_c, psi_c])[:,None] # (3,1)
        #pqr = np.array([p_c, q_c, r_c])[:,None] # (3,1)
        RotMat_pqr = Rotations.body2inertial_rates([phi, theta, psi]) #(3,3)
        dphi, dtheta, dpsi = (RotMat_pqr @ pqr)[:,0] # (3,)
        
        #dphi, dtheta, dpsi = p_c, q_c, r_c
        #'''
        ## Gather and return
        ds = [dx, dy, dz, dphi, dtheta, dpsi] 
        return ds 
    
    def update_state_3D3DOF(self, u):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:].copy() # (N,S), ground states

        for aid in self.agent_id_stepped: # notice stepped!
            aid_idx = self.agent_specs[aid]['idx']
            v = self.agent_specs[aid]['v_rel'] # constant velocity

            sol = sci.solve_ivp(self.EOM_3D3DOF_zoh, 
                                [float(self.t-self.t_delta), float(self.t)], 
                                state_prev[aid_idx,:],
                                args = ([u[aid], self.agent_specs[aid]],), # must be tuple (inputs,)
                                method = "RK45")
            agent_state = sol.y[:,-1] # (S,) select final RK timestep
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            ## handle & update states
            # clip pitch and modulus for phi & psi
            #agent_state[3] = ((agent_state[3] +0.5*np.pi) % (1.*np.pi)) - 0.5*np.pi
            
            agent_state[3] = ((agent_state[3] +np.pi) % (2.*np.pi)) - np.pi
            # OTHER DEFINITION?
            #agent_state[4] = max(min(agent_state[4],0.5*np.pi), - 0.5*np.pi)
            #agent_state[4] = max(min(agent_state[4],np.pi), 0.)
            agent_state[4] = ((agent_state[4] +np.pi) % (2.*np.pi)) - np.pi
            agent_state[5] = ((agent_state[5] +np.pi) % (2.*np.pi)) - np.pi
            
            # clip angles [-pi, pi]
            self.state_space[self.t_idx,aid_idx,0,[0,1,2,3,4,5]] = agent_state
            self.state_space[self.t_idx,aid_idx,1,[3,4,5]] = u[aid] # already update the inputted state (ideal)

            # align velocity
            angles = self.state_space[self.t_idx,aid_idx, 0, 3:6]
            RotMat_agent = Rotations.body2inertial(angles) 
            self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = \
                RotMat_agent.apply([v,0.,0.]) # body to inertial, wrong sign!

        
        ## Finalize & return 

        return 
    
    def clip_velocity(self, v, v_max = 1.):
        v_norm = np.linalg.norm(v)
        v_norm_clipped = min(v_norm, v_max) #np.minimum(v_norm, self.v_norm_max) # clip it
        v_clipped = (v/(v_norm+1e-8))*v_norm_clipped
        return v_clipped
    
    def clip_acceleration(self, a, a_max = 1.):
        a_norm = np.linalg.norm(a)
        a_norm_clipped = min(a_norm, a_max) #np.minimum(v_norm, self.v_norm_max) # clip it
        a_clipped = (a/(a_norm+1e-8))*a_norm_clipped
        return a_clipped
    
    
    def EOM_3D3DOF_Acontrol_zoh(self,t,s, u):
        ''' 
        
        EOM for ideal 3-dimensional case with acceleration control
        
        
        
        Dynamics:
            progression of system from cartesian perspective with control 
            of acceleration in the inertial frame. 
            
            ATTENTION: this function completely takes place in the inertial
            reference frame. Hence, one should ensure that both control inputs
            and states are already in the inertial frame!
            
        Inputs:
            Zero-order-hold acceleration commands inputs.
        
        Rotation matrices
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ''' 
        ## unpack
        
        #'''
        x, y, z, dx, dy, dz, ddx, ddy, ddz =  s # states
        (ddx_c, ddy_c, ddz_c), specs = u # commanded inputs    
        a_g, a_T_range, drag_C, tau = specs['a_g'], specs['a_T_range'], \
                                    specs['drag_C'], specs['tau']
        
        ## Dynamics system definition
        
        # clipping 
        #ddx_c, ddy_c, ddz_c = self.clip_acceleration(np.array([ddx_c, ddy_c, ddz_c])) * a_T_range[1] 
        ddx_c, ddy_c, ddz_c = self.clip_acceleration(np.array([ddx_c, ddy_c, ddz_c]), a_max = a_T_range[1])
        # this means we can never command more than (|a_c|=1)*T
        
        # acceleration state (thrust - gravity - drag)
        dd = np.array([ddx, ddy, ddz]) # clip to |T| norm
        ddx, ddy, ddz = self.clip_acceleration(dd,a_max= a_T_range[1]) # clip norm such that it never exceeds limit
        
        # acceleration control state 
        dddx = (ddx_c-ddx)/tau
        dddy = (ddy_c-ddy)/tau
        dddz = (ddz_c-ddz)/tau
        
        # dv
        dv_x = ddx - drag_C[0]*dx - drag_C[3]*(dx**2)*np.sign(dx)
        dv_y = ddy - drag_C[1]*dy - drag_C[4]*(dy**2)*np.sign(dy)
        dv_z = ddz - drag_C[2]*dz - drag_C[5]*(dz**2)*np.sign(dz) - a_g
        # ^ dv = acc - c_d1*v - c_d2*(v**2)*sign(v) - a_g
        
        ## Gather and return
        ds = [dx, dy, dz, dv_x, dv_y, dv_z, dddx, dddy, dddz] 
        return ds 
    
    def update_state_3D3DOF_Acontrol(self, u):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:].copy() # (N,S), ground states (N,6)
        state_prev = np.hstack((state_prev, self.state_space[(self.t_idx-1),:,1,:3].copy())) #(N,6) + (N,3) -> (N,9)
        state_prev = np.hstack((state_prev, self.state_space[(self.t_idx-1),:,2,:3].copy())) #(N,9) + (N,3) -> (N,12)
        
        for aid in self.agent_id_stepped: # notice stepped!
            aid_idx = self.agent_specs[aid]['idx']
            
            '''
            rotmat_BI =  Rotations.body2inertial(state_prev[aid_idx, [3,4,5]])
            a_c_I = rotmat_BI.apply(u[aid]) # u[aid] = a_c_B, body frame acceleration command
            # a_c_I; inertial frame acceleration command
            '''
            a_c_I = u[aid]
            # A_CONTROL_3D3DOF BODY FRAME
            #'''
            sol = sci.solve_ivp(self.EOM_3D3DOF_Acontrol_zoh, 
                                [float(self.t-self.t_delta), float(self.t)], 
                                state_prev[aid_idx,[0,1,2,6,7,8, 9,10,11]],
                                args = ([a_c_I, self.agent_specs[aid]],), # must be tuple (inputs,)
                                method = "RK45")
            agent_state = sol.y[:,-1] # (S,) select final RK timestep
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            ## handle & update states
            '''
            a_I = a_c_I # acceleration state, TODO REPLACE WITH ACCELERATION STATE! 
            angles_aid = Rotations.body2inertial_align(self.thrust_orientation.copy(),
                                                       a_I, 
                                                       normalize = 2) # normalize a_I
            '''
            agent_state[6:] = self.clip_acceleration(agent_state[6:],
                                   a_max= self.agent_specs[aid]['a_T_range'][1])
            angles_aid = 0.
            # A_CONTROL_3D3DOF BODY FRAME
            #'''
            self.state_space[self.t_idx,aid_idx,0,[3,4,5]] = angles_aid
            self.state_space[self.t_idx,aid_idx,0,[0,1,2]] = agent_state[:3]
            self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = agent_state[3:6] 
            self.state_space[self.t_idx,aid_idx,2,[0,1,2]] = agent_state[6:] 
            # TODO SAVE ACCELERATION STATE! 
            '''
            ## PSEUDO CODE 3D3DOF A_CONTROL
            
            while True:
            (1)    x_I = get_observation_set()
            (2)    x_B = R_IB(attitude)@x_I 
            
            (3)    A_c_B = model(x_b)
            (4)    A_c_I = R_BI(attitude)@A_c_B 
            
            (5)    state_I = simulate_system_I(state_prev_I, A_c_I)
                   A_I, ... = state_I
            (6)    attitude = KabashAlgorithm(Thrust_orientation_B, A_I) 
                
            #'''
        ## Finalize & return 
        return 
    
    def EOM_3D4DOF_zoh_TF(self,t,s, u):
        ''' 
        
        EOM for ideal 3-dimensional case & 4 DOF
        
        Dynamics:
            progression of system from angular perspective with control 
            of pitch, roll and yaw angles as well as thrust.
            According to ROS book on drone dynamics for MPC models.
            This implies velocity vector does not align with any axes 
            necessarily.
            
        Inputs:
            Zero-order-hold input of rotational rates (euler) and thrust.
            Note that thrust is normalized by mass and relative to hover thrust.
        
        Rotation matrices
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ''' 
        ## unpack
        x, y, z, phi, theta, psi, dx, dy, dz, p, q, r, a_T =  s # states
        phi = ((phi +np.pi) % (2.*np.pi)) - np.pi
        theta = max(min(theta,0.5*np.pi), - 0.5*np.pi) # theta clip
        psi = ((psi +np.pi) % (2.*np.pi)) - np.pi

        #(T_cG, phi_c, theta_c, psi_c), specs = u # commanded inputs, T_cG is thrust relative to hower 
        (T_c, p_c, q_c, r_c), specs = u
        a_g, a_T_range = specs['a_g'], specs['a_T_range'] # gravity & thrust magnitude
        tau, drag_C = specs['tau'], specs['drag_C']
        a_pos = specs['a_pos']
        
        a_limT = a_pos*a_T_range[0] + a_T_range[1] 
        a_limB = -1*(1-a_pos)*a_T_range[0] 
        ## Compute acceleration command
        a_T_c = (a_pos+min(T_c,0.))*a_T_range[0] + max(T_c,0.)*a_T_range[1] 
        # [m/s^2], acceleration relative hover (in body frame!)
        # first is [-1,0], where -1 gives a_c = 0 as bias added, other is [0,1] 

        ## Clip states
        p = max(min(p, self.rate_limit), -self.rate_limit) # p
        q = max(min(q, self.rate_limit), -self.rate_limit) # q
        r = max(min(r, self.rate_limit), -self.rate_limit) # r
        a_T = max(min(a_T,a_limT),a_limB) # a_T; [0,a_max]
        a_T_c = max(min(a_T_c,a_limT),a_limB) # a_c; [0,a_max]
        
        ## rotation matrix instance 
        RotMat_BI = Rotations.body2inertial([phi, theta, psi])
        RotMat_IB = Rotations.inertial2body([phi, theta, psi])
        
        ## Control Dynamics
        dp = (p_c - p)/tau
        dq = (q_c - q)/tau 
        dr = (r_c - r)/tau
        da_T = (a_T_c - a_T)/tau
        
        ## Dynamics system definition
        # cartesian states
        # notice dx, dy, dz pass through, implicitly connected by shift of outputs
        # SOURCED; End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight
        
        dx_B, dy_B, dz_B = RotMat_IB.apply([dx, dy, dz])
        F_B = [-drag_C[0]*dx_B - drag_C[3]*(dx_B**2)*np.sign(dx_B),         # Fx_B
               -drag_C[1]*dy_B - drag_C[4]*(dy_B**2)*np.sign(dy_B),         # Fy_B
               -drag_C[2]*dz_B - drag_C[5]*(dz_B**2)*np.sign(dz_B) + a_T,   # Fz_B
               ]

        ddx, ddy, ddz = RotMat_BI.apply(F_B) \
            -  [0.,0.,a_g] # notice [0,0,g] in inertial frame already!

        # angular states
        pqr = np.array([p, q, r])[:,None] # (3,1)
        RotMat_pqr = Rotations.body2inertial_rates([phi, theta, psi]) #(3,3)
        dphi, dtheta, dpsi = (RotMat_pqr @ pqr)[:,0] # (3,)
        
        ## 
        '''
        pi_scaler = np.pi/0.1
        dphi = ((dphi +pi_scaler) % (2.*pi_scaler)) - pi_scaler # dphi
        dtheta = max(min(dtheta, pi_scaler), - pi_scaler) # dtheta
        dpsi = ((dpsi +pi_scaler) % (2.*pi_scaler)) - pi_scaler # dpsi
        #'''
        
        ## Gather and return
        ds = [dx, dy, dz, dphi, dtheta, dpsi, ddx, ddy, ddz, dp, dq, dr, da_T] 
        return ds 
    
    def update_state_3D4DOF_TF(self, u):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:].copy() # (N,S), ground states (N,6)
        state_prev = np.hstack((state_prev, self.state_space[(self.t_idx-1),:,1,:].copy())) #(N,6) + (N,6) -> (N,12), velocity & p,q,r
        state_prev = np.hstack((state_prev, self.state_space[(self.t_idx-1),:,2,3].copy()[:,None])) #(N,12) + (N,1) -> (N,13), T
        # ^ note that dim is added due to special indexing in numpy
        
        for aid in self.agent_id_stepped: # notice stepped!
            aid_idx = self.agent_specs[aid]['idx']
        
            #'''
            # rk45
            sol = sci.solve_ivp(self.EOM_3D4DOF_zoh_TF, 
                                [float(self.t-self.t_delta), float(self.t)], 
                                state_prev[aid_idx,:],
                                args = ([u[aid], self.agent_specs[aid]],), # must be tuple (inputs,)
                                method = "RK45")
            assert sol.status == 0, f'solver failed to converge, status: {sol.status}'
            agent_state = sol.y[:,-1] # (S,) select final RK timestep
            
            '''
            # forward euler
            agent_state = state_prev[aid_idx,:].copy()
            agent_dstate = np.array(self.EOM_3D4DOF_zoh_TF(self.t_delta,
                            agent_state.copy(),[u[aid], self.agent_specs[aid]]))
            agent_state += self.t_delta*agent_dstate # s + dt*ds
            
            #''' 
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            ## handle states
            # clip angles [-pi, pi]
            agent_state[3] = ((agent_state[3] +np.pi) % (2.*np.pi)) - np.pi
            agent_state[4] = max(min(agent_state[4],0.5*np.pi), - 0.5*np.pi)
            agent_state[5] = ((agent_state[5] +np.pi) % (2.*np.pi)) - np.pi
            
            # clip control states
            agent_state[9] = max(min(agent_state[9], self.rate_limit), -self.rate_limit) # p
            agent_state[10] = max(min(agent_state[10], self.rate_limit), -self.rate_limit) # q
            agent_state[11] = max(min(agent_state[11], self.rate_limit), -self.rate_limit) # r
            agent_state[12] = max(min(agent_state[12],sum(self.agent_specs[aid]['a_T_range'])),0.) # T; [0,a_max]
            
            ## update states
            self.state_space[self.t_idx,aid_idx,0,[0,1,2,3,4,5]] = agent_state[[0,1,2,3,4,5]] # x,y,z, phi, theta, psi
            self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = agent_state[[6,7,8]] # dx, dy, dz
            self.state_space[self.t_idx,aid_idx,1,[3,4,5]] = agent_state[[9,10,11]] # p,q,r
            self.state_space[self.t_idx,aid_idx,2,3] = agent_state[12] # T
            
            ## derivative states
            #'''
            RotMat_BI = Rotations.body2inertial(agent_state[[3,4,5]])
            RotMat_IB = Rotations.inertial2body(agent_state[[3,4,5]])
            
            dx_B, dy_B, dz_B = RotMat_IB.apply(agent_state[[6,7,8]])
            specs = self.agent_specs[aid].copy()
            a_g, drag_C = specs['a_g'], specs['drag_C']
            a_T = agent_state[12]
            
            F_B = [-drag_C[0]*dx_B - drag_C[3]*(dx_B**2)*np.sign(dx_B),         # Fx_B
                   -drag_C[1]*dy_B - drag_C[4]*(dy_B**2)*np.sign(dy_B),         # Fy_B
                   -drag_C[2]*dz_B - drag_C[5]*(dz_B**2)*np.sign(dz_B) + a_T,   # Fz_B
                   ]
            self.state_space[self.t_idx,aid_idx,2,[0,1,2]] = RotMat_BI.apply(F_B) - [0.,0.,a_g] # acc
            
            
            '''
            #self.state_space[self.t_idx,aid_idx,2,[0,1,2]] = agent_dstate[[6,7,8]]
            #'''
        ## Finalize & return 
        return 
    
    def EOM_3D4DOF_zoh_QUAT(self,t,s, u):
        ''' 
        
        EOM for ideal 3-dimensional case & 4 DOF
        
        Dynamics:
            progression of system from angular perspective with control 
            of pitch, roll and yaw angles as well as thrust.
            According to ROS book on drone dynamics for MPC models.
            This implies velocity vector does not align with any axes 
            necessarily.
            
        Inputs:
            Zero-order-hold input of rotational rates (euler) and thrust.
            Note that thrust is normalized by mass and relative to hover thrust.
        
        Rotation matrices
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ''' 
        ## unpack
        x, y, z, dx, dy, dz, p, q, r, a_T, qt1, qt2, qt3, qt4 =  s # states


        #(T_cG, phi_c, theta_c, psi_c), specs = u # commanded inputs, T_cG is thrust relative to hower 
        (T_c, p_c, q_c, r_c), specs = u
        a_g, a_T_range = specs['a_g'], specs['a_T_range'] # gravity & thrust magnitude
        tau, drag_C = specs['tau'], specs['drag_C']
        a_pos, rate_limit = specs['a_pos'], specs['rate_limit']
        
        a_limT = a_pos*a_T_range[0] + a_T_range[1] 
        a_limB = -1*(1-a_pos)*a_T_range[0] 
        ## Compute acceleration command
        a_T_c = (a_pos+min(T_c,0.))*a_T_range[0] + max(T_c,0.)*a_T_range[1] 
        # [m/s^2], acceleration relative hover (in body frame!)
        # first is [-1,0], where -1 gives a_c = 0 as bias added, other is [0,1] 

        ## Clip states
        p = max(min(p, rate_limit), -rate_limit) # p
        q = max(min(q, rate_limit), -rate_limit) # q
        r = max(min(r, rate_limit), -rate_limit) # r
        a_T = max(min(a_T,a_limT),a_limB) # a_T; [a_min, a_max]
        a_T_c = max(min(a_T_c,a_limT),a_limB) # a_c; [a_min, a_max]
        
        ## rotation matrix instance 
        quats = np.array([qt1, qt2, qt3, qt4]) # x,y,z,w format
        #'''
        RotMat_IB = R.from_quat(quats) # expects x,y,z,w format
        RotMat_BI = RotMat_IB.inv()
        ''' 
        # incorrect
        RotMat_BI = R.from_quat(quats) # expects x,y,z,w format
        RotMat_IB = RotMat_BI.inv()
        #'''
        ## Control Dynamics
        dp = (p_c - p)/tau
        dq = (q_c - q)/tau 
        dr = (r_c - r)/tau
        da_T = (a_T_c - a_T)/tau
        
        ## Dynamics system definition
        # cartesian states
        # notice dx, dy, dz pass through, implicitly connected by shift of outputs
        # SOURCED; End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight
            
        dx_B, dy_B, dz_B = RotMat_IB.apply([dx, dy, dz])
        F_B = [-drag_C[0]*dx_B - drag_C[3]*abs(dx_B)*dx_B       - drag_C[6]*(dx_B)/(self.t_delta*self.null_delay),  # Fx_B  ## ALEX
               -drag_C[1]*dy_B - drag_C[4]*abs(dy_B)*dy_B       - drag_C[7]*(dy_B)/(self.t_delta*self.null_delay),  # Fy_B
               -drag_C[2]*dz_B - drag_C[5]*abs(dz_B)*dz_B + a_T - drag_C[8]*(dz_B)*(1/(1.+abs(a_T)**(0.25)))/(self.t_delta*self.null_delay),   # Fz_B
               ]

        ddx, ddy, ddz = RotMat_BI.apply(F_B) \
            -  [0.,0.,a_g] # notice [0,0,g] in inertial frame already!

        # angular states
        RotMat_pqr = Rotations.quats_body2quats_rates([p, q, r]) #(4,3)
        dqt1, dqt2, dqt3, dqt4 = (RotMat_pqr @ (quats[:,None]))[:,0] # (4,4)@(4,1) = (4,)
        
        
        ## Gather and return
        ds = [dx, dy, dz, ddx, ddy, ddz, dp, dq, dr, da_T, dqt1, dqt2, dqt3, dqt4] 
        return ds 
    
    def update_state_3D4DOF_QUAT(self, u):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        
        zzzzzzzzzzzz
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:3].copy() # (N,S), ground states (N,3)
        state_prev = np.hstack((state_prev, self.state_space[(self.t_idx-1),:,1,:].copy())) #(N,3) + (N,6) -> (N,9), velocity & p,q,r
        state_prev = np.hstack((state_prev, self.state_space[(self.t_idx-1),:,2,3].copy()[:,None])) #(N,9) + (N,1) -> (N,10), T
        # ^ note that dim is added due to special indexing in numpy
        
        angles_prev = self.state_space[(self.t_idx-1),:,0,3:].copy()
        
        for aid in self.agent_id_stepped: # notice stepped!
            aid_specs = self.agent_specs[aid].copy()
            aid_idx = aid_specs['idx']
            aid_rate_limit = aid_specs['rate_limit']
        
            quats = Rotations.inertial2body(angles_prev[aid_idx,:])\
                .as_quat(canonical = True) # x,y,z,w
            state_prev_aid = np.hstack((state_prev[aid_idx,:],quats))  
        
            '''
            ## QUATCONTROL
            Rotmat_quats = Rotations.quats_quats2body_rates(quats)
            pqr = list((Rotmat_quats @ u[aid][1:,None])[:,0])
            #'''
            
            ''' 
            # rk45
            sol = sci.solve_ivp(self.EOM_3D4DOF_zoh_QUAT, 
                                [float(self.t-self.t_delta), float(self.t)], 
                                state_prev_aid,
                                args = ([u[aid], self.agent_specs[aid]],), # must be tuple (inputs,)
                                #args = ([(*pqr, u[aid][0]), self.agent_specs[aid]],), # must be tuple (inputs,)
                                method = "RK45") #RK45
            assert sol.status == 0, f'solver failed to converge, status: {sol.status}'
            agent_state = sol.y[:,-1] # (S,) select final RK timestep
            '''
            # forward euler
            H = 5#5
            t_h = self.t_delta/H
            agent_state = state_prev_aid.copy()
            for h in range(H):
                agent_dstate = np.array(self.EOM_3D4DOF_zoh_QUAT(self.t_delta,
                                agent_state.copy(),[u[aid], self.agent_specs[aid]]))
                agent_state += t_h*agent_dstate # s + dt*ds
            
            #''' 
            
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            
            RotMat_IB = R.from_quat(agent_state[[10,11,12,13]])
            RotMat_BI = RotMat_IB.inv()
            phi, theta, psi = RotMat_IB.as_euler('xyz')
            
            ## handle states
            # clip angles [-pi, pi]
            #'''
            phi = ((phi + np.pi) % (2.*np.pi)) - np.pi
            theta = max(min(theta,0.5*np.pi), - 0.5*np.pi)
            psi = ((psi + np.pi) % (2.*np.pi)) - np.pi
            #'''
            
            # clip control states
            agent_state[6] = max(min(agent_state[6], aid_rate_limit), -aid_rate_limit) # p
            agent_state[7] = max(min(agent_state[7], aid_rate_limit), -aid_rate_limit) # q
            agent_state[8] = max(min(agent_state[8], aid_rate_limit), -aid_rate_limit) # r
            
            a_T_range, a_pos = aid_specs['a_T_range'], aid_specs['a_pos'] 
            a_limT = a_pos*a_T_range[0] + a_T_range[1] 
            a_limB = -1*(1-a_pos)*a_T_range[0] 
            agent_state[9] = max(min(agent_state[9],a_limT),a_limB) # T; [0,a_max]
            
            ## update states
            self.state_space[self.t_idx,aid_idx,0,[0,1,2]] = agent_state[[0,1,2]] # x,y,z, phi, theta, psi
            self.state_space[self.t_idx,aid_idx,0,[3,4,5]] = [phi, theta, psi]
            self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = agent_state[[3,4,5]] # dx, dy, dz
            self.state_space[self.t_idx,aid_idx,1,[3,4,5]] = agent_state[[6,7,8]] # p,q,r
            self.state_space[self.t_idx,aid_idx,2,3] = agent_state[9] # T
            
            ## derivative states
            #'''
            
            dx_B, dy_B, dz_B = RotMat_IB.apply(agent_state[[3,4,5]])
            a_g, drag_C = aid_specs['a_g'], aid_specs['drag_C']
            a_T = agent_state[9]
            
            F_B = [-drag_C[0]*dx_B - drag_C[3]*abs(dx_B)*dx_B - drag_C[6]*(dx_B)/(self.t_delta*self.null_delay),         # Fx_B
                   -drag_C[1]*dy_B - drag_C[4]*abs(dy_B)*dy_B - drag_C[7]*(dy_B)/(self.t_delta*self.null_delay),         # Fy_B
                   -drag_C[2]*dz_B - drag_C[5]*abs(dz_B)*dz_B + a_T - drag_C[8]*(dz_B)*(1/(1.+abs(a_T)**(0.25)))/(self.t_delta*self.null_delay),   # Fz_B
                   ]
            self.state_space[self.t_idx,aid_idx,2,[0,1,2]] = RotMat_BI.apply(F_B) - [0.,0.,a_g] # acc
            
            
            '''
            #self.state_space[self.t_idx,aid_idx,2,[0,1,2]] = agent_dstate[[6,7,8]]
            #'''
        ## Finalize & return 
        return 
    
    
    
    
#%% IDEAL OR UNUSED ONES

    def EOM_3D4DOF_zoh(self,t,s, u):
        ''' 
        
        EOM for ideal 3-dimensional case & 4 DOF
        
        Dynamics:
            progression of system from angular perspective with control 
            of pitch, roll and yaw angles as well as thrust.
            According to ROS book on drone dynamics for MPC models.
            This implies velocity vector does not align with any axes 
            necessarily.
            
        Inputs:
            Zero-order-hold input of rotational rates (euler) and thrust.
            Note that thrust is normalized by mass and relative to hover thrust.
        
        Rotation matrices
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ''' 
        ## unpack
        x, y, z, phi, theta, psi, dx, dy, dz =  s # states
        theta = max(min(theta,0.5*np.pi), - 0.5*np.pi) # theta clip

        #(T_cG, phi_c, theta_c, psi_c), specs = u # commanded inputs, T_cG is thrust relative to hower 
        (T_c, p_c, q_c, r_c), specs = u
        a_g, a_T_range = specs['a_g'], specs['a_T_range'] # gravity & thrust magnitude
        
        a_c = (1+min(T_c,0.))*a_T_range[0] + max(T_c,0.)*a_T_range[1] 
        # [m/s^2], acceleration relative hover (in body frame!)
        # first is [-1,0], where -1 gives a_c = 0 as bias added, other is [0,1] 

        a_c = max(min(a_c, sum(a_T_range)),0) # enforce bounds
        ## LIMIT VELOCITY 
        # recognize that velocity is unaffected by acc this timestep (thus shoudl stay within bounds)
        #dx, dy, dz = self.clip_velocity(np.array([dx, dy, dz]))
        
        ## rotation matrix istance 
        RotMat_BI = Rotations.body2inertial([phi, theta, psi])
        RotMat_IB = Rotations.inertial2body([phi, theta, psi])
        ## Dyanmics system definition
        # cartesian 
        #notice dx, dy, dz pass through, implicitly connected by shift of outputs
        SIGN = 1.
        '''
        # only thrust in frame
        F_B = [0., 0., SIGN*a_cG] # force in body frame
        '''
        # SOURCED; End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight
        drag_C = specs['drag_C'] # [0.34, 0.43, 0.]
        
        dx_B, dy_B, dz_B = RotMat_IB.apply([dx, dy, dz])
        F_B = [-drag_C[0]*dx_B, 
               -drag_C[1]*dy_B, 
               SIGN*a_c - drag_C[2]*dz_B]
        #''' 
        ddx, ddy, ddz = RotMat_BI.apply(F_B) \
            -  [0.,0.,SIGN*a_g] # notice a_cG & [0,0,g] in inertial frame already!

        # angular (ideal atm)

        #COMMANDS IN INTERTIAL FRAME I.E. phi_c, theta_c, psi_c
        #dphi =  self.TF_RotationRate(t,phi_c, phi, specs = specs['phi'])  
        #dtheta = self.TF_RotationRate(t,theta_c, theta, specs = specs['theta']) 
        #dpsi = self.TF_RotationRate(t,psi_c, psi, specs = specs['psi'])

        # COMMANDS IN BODY FRAME (REQUIRES CONVERSION)
        #'''
        #pqr = np.array([dphi, dtheta, dpsi])[:,None] # (3,1)
        pqr = np.array([p_c, q_c, r_c])[:,None] # (3,1)
        RotMat_pqr = Rotations.body2inertial_rates([phi, theta, psi]) #(3,3)
        dphi, dtheta, dpsi = (RotMat_pqr @ pqr)[:,0] # (3,)
        
        '''
        dphi, dtheta, dpsi = p_c, q_c, r_c
        #'''
        ## Gather and return
        ds = [dx, dy, dz, dphi, dtheta, dpsi, ddx, ddy, ddz] 
        return ds 
    
    def update_state_3D4DOF(self, u):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:].copy() # (N,S), ground states (N,6)
        state_prev = np.hstack((state_prev, self.state_space[(self.t_idx-1),:,1,:3].copy())) #(N,6) + (N,3) -> (N,9)
        for aid in self.agent_id_stepped: # notice stepped!
            aid_idx = self.agent_specs[aid]['idx']
        
            sol = sci.solve_ivp(self.EOM_3D4DOF_zoh, 
                                [float(self.t-self.t_delta), float(self.t)], 
                                state_prev[aid_idx,:],
                                args = ([u[aid], self.agent_specs[aid]],), # must be tuple (inputs,)
                                method = "RK45")
            agent_state = sol.y[:,-1] # (S,) select final RK timestep
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            ## handle & update states
            # clip pitch and modulus for phi & psi
            #agent_state[6] = ((agent_state[6] +0.5*np.pi) % (1.*np.pi)) - 0.5*np.pi
            agent_state[3] = ((agent_state[3] +np.pi) % (2.*np.pi)) - np.pi
            # OTHER DEFINITION?
            agent_state[4] = max(min(agent_state[4],0.5*np.pi), - 0.5*np.pi)
            #agent_state[4] = max(min(agent_state[4],np.pi), 0.)
            #agent_state[4] = ((agent_state[4] +np.pi) % (2.*np.pi)) - np.pi
            agent_state[5] = ((agent_state[5] +np.pi) % (2.*np.pi)) - np.pi
            
            # clip angles [-pi, pi]
            self.state_space[self.t_idx,aid_idx,0,[0,1,2,3,4,5]] = agent_state[[0,1,2,3,4,5]]
            self.state_space[self.t_idx,aid_idx,1,[3,4,5]] = u[aid][1:] # already update the inputted state (ideal)

            # velocity forms part of integrated states, so no alignment
            #self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = self.clip_velocity(agent_state[[6,7,8]])
            self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = agent_state[[6,7,8]]
            #raise Exception('TODO accelerations!')
            
        ## Finalize & return 
 
        return 
    
    #########################################################################
    #@latexify.function
    def EOM_3D4DOF_Vcontrol_zoh(self,t,s, u):
        ''' 
        
        EOM for ideal 3-dimensional case & 3 DOF
        
        Dynamics:
            progression of system from angular perspective with control 
            of pitch, roll and yaw rate. Velocity vector alligns with body frame
            x-axis. Similar to missile.
            
            pitch and yaw are controlled as extension of 2d case, where
            only yaw was controlled. Therefore we dont opt for pitch
            and roll command which is more conventional.
            
        Inputs:
            Zero-order-hold input of rotational rates (euler).
        
        Rotation matrices
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ''' 
        ## unpack
        
        #'''
        x, y, z, phi, theta, psi =  s # states
        theta = max(min(theta,0.5*np.pi), - 0.5*np.pi) # theta clip

        #(v_c, phi_c, theta_c, psi_c), specs = u # commanded inputs
        (v_c, p_c, q_c, r_c), specs = u # commanded inputs

        ## rotation matrix istance 
        RotMat = Rotations.body2inertial([phi,theta, psi])
        
        ## Dyanmics system definition
        # cartesian; from body to inertial
        dx, dy, dz = RotMat.apply([v_c,0.,0.]) # velocity aligns with body x-axis
        
        # angular (ideal atm)
        '''
        dphi =  self.TF_RotationRate(t,phi_c, phi, specs = specs['phi'])  
        dtheta = self.TF_RotationRate(t,theta_c, theta, specs = specs['theta']) 
        dpsi = self.TF_RotationRate(t,psi_c, psi, specs = specs['psi'])  
        '''
        
        #'''
        pqr = np.array([p_c, q_c, r_c])[:,None] # (3,1)
        RotMat_pqr = Rotations.body2inertial_rates([phi, theta, psi]) #(3,3)
        dphi, dtheta, dpsi = (RotMat_pqr @ pqr)[:,0] # (3,)
        '''
        dphi, dtheta, dpsi = p_c, q_c, r_c # TODO THETA
        #'''
        
        ## Gather and return
        ds = [dx, dy, dz, dphi, dtheta, dpsi] 
        return ds 
    
    def update_state_3D4DOF_Vcontrol(self, u):
        '''
        TODO 
        - IMPROVE EFFICIENCY; a lot of packing and unpacking
        '''
        ## update step and time
        self.t += self.t_delta # update time
        self.t_idx += 1 # update time index
        
        ## System integration 
        state_prev = self.state_space[(self.t_idx-1),:,0,:].copy() # (N,S), ground states (N,6)
        for aid in self.agent_id_stepped: # notice stepped!
            aid_idx = self.agent_specs[aid]['idx']
            
            v_c = u[aid][0]
            v_c = max(1. + v_c, 1e-6)*(self.agent_specs[aid]['v_rel']/2.)
            #u[aid][0] = v_c
            
            sol = sci.solve_ivp(self.EOM_3D4DOF_Vcontrol_zoh, 
                                [float(self.t-self.t_delta), float(self.t)], 
                                state_prev[aid_idx,:],
                                args = ([(v_c, *u[aid][1:]), self.agent_specs[aid]],), # must be tuple (inputs,)
                                method = "RK45")
            agent_state = sol.y[:,-1] # (S,) select final RK timestep
            agent_state = np.nan_to_num(agent_state, 
                                        copy = False, nan = 0., posinf=0., neginf=0.)
            ## handle & update states
            #agent_state[3] = ((agent_state[3] +0.5*np.pi) % (1.*np.pi)) - 0.5*np.pi
            
            agent_state[3] = ((agent_state[3] +np.pi) % (2.*np.pi)) - np.pi
            # OTHER DEFINITION?
            #agent_state[4] = ((agent_state[4] +np.pi) % (2.*np.pi)) - np.pi # TODO THETA
            agent_state[4] = max(min(agent_state[4],0.5*np.pi), - 0.5*np.pi)
            agent_state[5] = ((agent_state[5] +np.pi) % (2.*np.pi)) - np.pi
            
            # clip angles [-pi, pi]
            self.state_space[self.t_idx,aid_idx,0,[0,1,2,3,4,5]] = agent_state
            self.state_space[self.t_idx,aid_idx,1,[3,4,5]] = u[aid][1:] # already update the inputted state (ideal)

            # align velocity
            angles = self.state_space[self.t_idx,aid_idx, 0, 3:6]
            RotMat_agent = Rotations.body2inertial(angles) 
            self.state_space[self.t_idx,aid_idx,1,[0,1,2]] = \
                RotMat_agent.apply([v_c,0.,0.]) # body to inertial, wrong sign!
            
        ## Finalize & return 
 
        return 
    
'''
# RENDER POLICY IDEA
import matplotlib.pyplot as plt

def render_policy(env, policy):
    fig, ax = plt.subplots()
    for s in range(env.nS):
        y, x = np.unravel_index(s, env.shape)
        action = policy[s]
        if action == 0:  # left
            dx, dy = -0.5, 0
        elif action == 1:  # down
            dx, dy = 0, -0.5
        elif action == 2:  # right
            dx, dy = 0.5, 0
        elif action == 3:  # up
            dx, dy = 0, 0.5
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='k', ec='k')
    plt.show()

# Assume `env` is your environment and `policy` is your policy
render_policy(env, policy)


'''