import inspect
from typing import Callable, Any, Iterable, Union, List, Tuple, Set, Dict
#https://stackoverflow.com/questions/2489669/how-do-python-functions-handle-the-types-of-parameters-that-you-pass-in

import numpy as np
import torch
import warnings
import types
import glob
import copy
import os

import ray
from ray.tune.registry import register_env

from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info

from ray.rllib.algorithms.ppo import PPOConfig


from ray.rllib.models import ModelCatalog
from ray.rllib.utils.spaces import space_utils

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
    
from ray.rllib.policy.sample_batch import SampleBatch

from ray.tune.registry import register_env

import pytorch_lightning as pl

#%%

SELECT_ENV = "PEG1_env"
        
import PEG1_env as PEGenv
import PEG1_models as PEGmodels

#%% MISC

'''
import time

tic = time.time()
toc = time.time()-tic
round(toc,3)
    
'''

'''
## TRACKING HOW DICT HAS CHANGED
# do not reuse previous models config e.g. for change in # timesteps 

assert isinstance(ModelReplace_custom_model_config, dict)
# track changes
custom_model_config_old = aid_model_config['custom_model_config'].copy()
config_key_changes = set(custom_model_config_old) & set(ModelReplace_custom_model_config) # common keys
config_key_changes.update(set(ModelReplace_custom_model_config)-set(custom_model_config_old)) # also add new keys
config_changes ={k:(custom_model_config_old.get(k,'-'), ModelReplace_custom_model_config[k]) \
                 for k in config_key_changes \
                 if (ModelReplace_custom_model_config[k] != custom_model_config_old.get(k,0.))} # find associated values
print(f'Replaced model with updated config (old-> new) & (new):\n{config_changes}')
# update with new keys, keep rest the same
aid_model_config['custom_model_config'] = {**aid_model_config['custom_model_config'], 
                                         **ModelReplace_custom_model_config} 

'''

#%% POLICIES 

class PolicyUtils:
    '''
    Class holding all policy(specs) related functions
    
    MISC note 1; 
    what if instead of replacing a model, we want to define a fixed policy
    SOURCE/EXAMPLE; https://discuss.ray.io/t/how-to-provide-tune-with-fixed-policy/3273/2
    INTERESTINGLY; WE can define trajectoryview reqs here
    -> i dont think we need this because our model can also define this and it seems a better location
    
    ''' 
    
    def extra_action_out(
        self, input_dict, state_batches, model, action_dist, 
    ) -> dict:
        """Returns dict of extra info to include in experience batch.

        Args:
            input_dict: Dict of model input tensors.
            state_batches: List of state tensors.
            model: Reference to the model object.
            action_dist: Torch action dist object
                to get log-probs (e.g. for already sampled actions).

        Returns:
            Extra outputs to return in a `compute_actions_from_input_dict()`
            call (3rd return value).
            
        USE THIS BY DOING THE FOLLOWING:
            
        for aid in agents:
            algo.get_policy(aid).extra_action_out = ...utils.Policy.extra_action_out
            
        """
        ## always included (original) (see ray/rllib/policy/torch_mixin)
        out_dct = {SampleBatch.VF_PREDS: model.value_function()}
        
        ## own inclusions
        dLoS_pred, dLoS_target = model._dLoS_pred, model._dLoS_target
        
        gain_mu, gain_logstd = model._gain_mu, model._gain_logstd
        
        ##
        out_dct = {**out_dct, 
                   "dLoS_pred":dLoS_pred,
                   "dLoS_target":dLoS_target,
                   "gain_mu":gain_mu,
                   "gain_logstd":gain_logstd,
                   }
        return out_dct
    
    @staticmethod
    def override_policies_extra_action_out(algo, agent_ids: list):
        
        for aid in agent_ids:
            #algo.get_policy(aid).extra_action_out = PEGutils.Policy.extra_action_out
            '''
            funcType = type(algo.get_policy(aid).extra_action_out)
            algo.get_policy(aid).extra_action_out = funcType(PEGutils.Policy.extra_action_out, 
                                                             algo.get_policy(aid).extra_action_out, 
                                                             ray.rllib.algorithms.ppo.ppo_torch_policy.PPOTorchPolicy)
            '''
            algo.get_policy(aid).extra_action_out = \
                types.MethodType(TorchPolicyV2.extra_action_out, 
                                 algo.get_policy(aid))
        print(f'Overriden extra_action_out for {agent_ids}')
        return 
        
        
    
    def register_models_from_PolicySpecs(PEGmodelsModule, AlgoConfig):
        '''
        Registers models by inferring the model type
        from the provided model identifier whcih is part of the policyspec.
    
        Note that this function always registers models on under this 
        model identifier. The reason for this is because the policy spec
        will refer to a model identifier (a string) thus we should not
        change the model identifier after defining the model spec.
        '''
        ## get all model types
        modelTypes = inspect.getmembers(PEGmodelsModule, inspect.isclass)
        modelTypesNames = [x[0] for x in modelTypes]
    
        ## register models
        policies_models = dict.fromkeys(list(AlgoConfig.policies.keys()), None)
        for aid in policies_models.keys():
            ## unpack
            PolicySpec_aid = AlgoConfig.policies[aid]
            
            # get model identifier
            if isinstance(PolicySpec_aid, tuple):
                PolicySpec_aid_config = PolicySpec_aid[-1]
            else:
                # ray rllib policyspec object
                PolicySpec_aid_config = PolicySpec_aid.config
                
            model_identifier_aid = PolicySpec_aid_config['model']['custom_model']
            modelTypeName_aid = model_identifier_aid.split('-')[1] # Pytorchmodeltype string
            # identifier = PEGmodels_version + - + Pytorchmodeltype + - + aid + extra_label
            
            # get model object from PEGmodels
            Model = modelTypes[modelTypesNames.index(modelTypeName_aid)][1] # cls object
            # if this fails then the model is not in PEGmodels
        
            ModelCatalog.register_custom_model(model_identifier_aid,Model)
            policies_models[aid] = model_identifier_aid
        print(f'Registered models for agents: {policies_models}')
        return 
        
    
    def set_PolicySpecs(PolicySpecs: dict, AlgoConfig = None, extra_label = ''):
        '''
        Universal function to setup and override policySpec objects.
        For overriding a AlgoConfig has to be provided which has the 
        attribute .policies containing the policyspecs for the algorithm
        set previously.
        
        PolicySpecs is a multidimensional dict with the following forM:
        {... (for every agent) ...
            aid:{'model':PEGmodels.PytorchModelObject, # model object to be registered
                     'model_config':{},  # model settings & potentially 'identifier' key to force model registration
                     'explore':False, # exploration boolean  note that eval has forced exploitation (= no explore)
                     'extra':{}, # extra settings e.g. gamma
                      ## spaces definition: for initial setup (i.e. override = False) we require spaces
                     'observation_space': env.observation_space[aid],
                     'action_space': env.action_space[aid],
                    },
        }
        
        What is a PolicySpec object (in the context of MARL)?
        essentially we provide an agent-specific policy config which overrides the main config which is established below
        - PPOconfig.override is a nice method to provide args and check if these args exist; 
        - Note that environment observation and action spaces mean nothing, unless we provide them to policy_spec,
            i.e. ray rllib does not infer the spaces from the environment!
            - Source A: https://discuss.ray.io/t/multiagents-type-actions-observation-space-defined-in-environement/5120/8
            - Source B: https://discuss.ray.io/t/obervation-space-and-action-space-in-multi-agent-env/3235/2

        '''    
            
        ##
        override = not isinstance(AlgoConfig, type(None)) # algoconfig exists and will be overriden
        policies_old_keys = set()
        if override:
            policies_old_keys = set(AlgoConfig.policies.keys())
            # check whether AlgoConfig.policies is a dict of PolicySpecs or tuples 
            psTupleType = isinstance(AlgoConfig.policies[next(iter(policies_old_keys))], tuple)
            
        policies_new_keys = set(PolicySpecs.keys())
        PolicySpecs_new = {}
        for aid in PolicySpecs.keys():
            PolicySpec_aid = PolicySpecs[aid]
    
            ## model identifier and config setup
            model_identifier = PolicySpec_aid['model_config'].get('identifier',None) 
            if isinstance(model_identifier, type(None)):
                # model identifier not provided, so setup
                model_identifier = str(PolicySpec_aid['model']).split("'")[1].replace('.','-') \
                    + f'-{aid}' + bool(extra_label)*f'-{extra_label}' 
                # identifier = PEGmodels_version + - + Pytorchmodeltype + - + aid + runLabel
            
            model_config = {'custom_model':model_identifier, # model identifier (!= name), used for registration
                            'custom_model_config':PolicySpec_aid['model_config'], # config passed to model 
                           }
            
            if PolicySpec_aid.get('model_action_dist', False): # check if has_key & if is not empty
                model_config = {**model_config,
                                'custom_action_dist':PolicySpec_aid['model_action_dist'],
                                }
            # TODO 1; CHECK OVERRIDING WITH THE FUNCTION AND INFORM THE USER, SEE FUNCTION BELOW
            # TODO 2; add entire other settings to this one? the env settings for tractability
            # TODO 3; reuse previous custom_model_config if not defined? similar to how we do in eval
            # ensure that environment specs does not change 
    
            ## setup PolicySpec config
            include_always = {## hardcode inclusion for tractability
                'normalize_actions':True,   # default true
                'clip_actions':False,  # default false
                # TODO NORMALIZE OBS?
            }
            
            PolicySpec_aid_config = {
                            # agent model
                            'model':model_config, # dict with custom_model & custom_model_config keys
                            ## agent specs
                            'explore':PolicySpec_aid['explore'],
                            # obs_ & action_space not includes as we only want to override agent, not main
                            **include_always,
                            **PolicySpec_aid.get('extra',{}), # additional settings, notice order; will override always
            }
            PolicySpec_aid_config = PPOConfig.overrides(**PolicySpec_aid_config) # check passed arguments
            if not PolicySpec_aid_config['normalize_actions']: warnings.warn('NOTE: normalize_actions CHANGED FROM DEFAULT (true)!')
            if PolicySpec_aid_config['clip_actions']: warnings.warn('NOTE: clip_actions CHANGED FROM DEFAULT (false)!')
            
            ## extract & check spaces
            if override:
                observation_space_aid =  PolicySpec_aid.get('observation_space',
                    AlgoConfig.policies[aid].observation_space if not psTupleType else AlgoConfig.policies[aid][1]) 
                # reuse previous unless provided 
                action_space_aid =  PolicySpec_aid.get('action_space',
                   AlgoConfig.policies[aid].action_space if not psTupleType else AlgoConfig.policies[aid][2]) 
                # reuse previous unless provided 
                
                # check if spaces have defined been defined previously
                assert not isinstance(observation_space_aid, type(None)), 'observation space has to have been defined, to be overriden'
                assert not isinstance(action_space_aid, type(None)), 'action space has to have been defined, to be overriden'
            else:
                # first time setup, (spaces have to be provided, otherwise keyerror)
                observation_space_aid = PolicySpec_aid['observation_space']
                action_space_aid = PolicySpec_aid['action_space']
                
            ## setup or override algoConfig's policies PolicySpec object
            PolicySpecs_new[aid] = PolicySpec(policy_class = None, # keep None; default inferred from algo
                                   observation_space = observation_space_aid, # reuse previous unless provided 
                                   action_space  = action_space_aid,  # reuse previous unless provided 
                                   config = PolicySpec_aid_config,  # new specs in dict (not AlgoConfig object)
                                   # overrides algo's settings for agent specifically (rest is kept the same, so refer to main also for details)
                                  )
            if override:
                AlgoConfig.policies[aid] = PolicySpecs_new[aid] if not psTupleType else tuple(PolicySpecs_new[aid].__dict__.values()) 
                # update policy (with policyspec or tuple object type)
    
        ## finalize and return
        print(f'Initialized policies\nsetup: {policies_new_keys-policies_old_keys} \
        | overriden: {policies_new_keys & policies_old_keys} \
        | untouched: {policies_old_keys-policies_new_keys}\
        \nBe sure to register the models!')
        return PolicySpecs_new
    
    
    def clear_policy_states(PEGmodelsModule, policy_states, PolicySpecs_tbcleared):
        '''
        for replaced policies in a restored algorithm
        it is required to clear their optimizers and 
        saved weights.
    
        policy_states can be accessed from states['worker']['policy_states']
        '''
        modelTypes = inspect.getmembers(PEGmodelsModule, inspect.isclass)
        modelTypesNames = [x[0] for x in modelTypes]
    
        cleared_agents = []
        for aid in PolicySpecs_tbcleared.keys():
            ## get new model object (overrides previous)
            PolicySpec_aid = PolicySpecs_tbcleared[aid]
            
            # get model identifier
            if isinstance(PolicySpec_aid, tuple):
                observation_space_aid = PolicySpec_aid[1]
                action_space_aid = PolicySpec_aid[2]
                PolicySpec_aid_config = PolicySpec_aid[-1]
            else:
                # ray rllib policyspec object
                observation_space_aid = PolicySpec_aid.observation_space
                action_space_aid = PolicySpec_aid.action_space
                PolicySpec_aid_config = PolicySpec_aid.config
                
            custom_model_config = PolicySpec_aid_config['model']
                
            model_identifier_aid = PolicySpec_aid_config['model']['custom_model']
            modelTypeName_aid = model_identifier_aid.split('-')[1]
            # identifier = PEGmodels_version + - + Pytorchmodeltype + - + aid
            
            # get model object from PEGmodels
            Model = modelTypes[modelTypesNames.index(modelTypeName_aid)][1] # cls object
            # if this fails then the model is not in PEGmodels
        
            
            ## adjust model weights/state
            # initialize model with observation, action & num_outputs
            '''
            Note that the spaces (obs & action) here originate 
            from the restored algo, as we intent to maintain consistency.
            *(this can form a chain on multiple subsequent calls)
            '''
            Model_instance = Model(observation_space_aid, # obs_space
                                 action_space_aid, # action_space
                                 int(2*action_space_aid._shape[0]), # num_ouputs
                                 custom_model_config, # entire model config
                                 '', # name
                                )
            # setup model & optimizers weight/state dict
            optim =  torch.optim.Adam(Model_instance.parameters())
            optim_state = optim.state_dict()
            
            ModelInstance_weights = {}
            for n, p in Model_instance.named_parameters():
                ModelInstance_weights[n] = p.detach().numpy()
            
            # replace model & optimizer state
            policy_states[aid]['weights'] = ModelInstance_weights
            policy_states[aid]['_optimizer_variables'] = [optim_state] # takes list
            # finalize
            cleared_agents.append(aid)
        print(f'Cleared agents: {cleared_agents}')
        return 
    
    
    def setup_with_strippedresources(checkpoint_path: str):
        
        ## get checkpoint
        checkpoint_info = get_checkpoint_info(checkpoint_path)
        
        ## policy mapping
        def PolMap(aid, *args, **kwargs):
            return aid
        policy_mapping_fn_override = PolMap


        algo_state = Algorithm._checkpoint_info_to_algorithm_state(
            checkpoint_info=checkpoint_info,
            policy_ids=None, #['p0','e0'],
            policy_mapping_fn=policy_mapping_fn_override, # overrides worker, but not eval policy_map
            policies_to_train=None, #['p0','e0'],
        ) # these none's are fine, ray rllib infers them from elsewhere

        algo_state['eval_policy_mapping_fn'] = policy_mapping_fn_override # override eval config as well!
    
        ## Callbacks
        
        #algo_state['config'].callbacks_class = CustomCallback # for old versions we might have to override the callback class type
        #CustomCallback.setup_metrics(custom_metrics_dict) # add metrics list
        #'''
        algo_state['config'].callbacks_class.setup_functionality(restore = False, HoF = False, curriculum= False,
                                                                 episode = True, evaluate = True) # switch off policy/model restoration callback
                
        ## resources 
        algo_state['config'].num_gpus = 0 # to speed up inference use a gpu
        algo_state['config'].num_rollout_workers = 0 # for training, so keep to zero
        algo_state['config'].evaluation_num_workers = 0 # for training, so keep to zero
        algo_state['config'].num_envs_per_worker = 10 # 10 speedup inference by (num_envs_per_worker,T,F) batches
        
        ##
        algo_state['config'].explore = False
        
        ##
        
        register_env(SELECT_ENV, lambda config: PEGenv.PursuitEvasionEnv(config)) # apparently registering env is required for ray to load in policy

        ## REGISTER MODELS
        PolicyUtils.register_models_from_PolicySpecs(PEGmodels, algo_state['config']) # register models
        
        algo_instance = Algorithm.from_state(algo_state) # This `algo` has been stripped of resources
        
        return algo_instance 
    
#%% SAMPLEBATCH

class SampleBatchUtils:
    
    @staticmethod
    def SB_readNformat_json_files(fileDir: str) -> tuple:
        
        filepaths = glob.glob(f"{fileDir}\*.json") # get all files in directory

        print(f'Reading files (N={len(filepaths)}):\n')
        for f in filepaths: print(f)
        
        JsonReader = ray.rllib.offline.json_reader.JsonReader(filepaths)
        
        samplebatchGen = JsonReader.read_all_files() # out; sample batch generator, will return episodes randomly (not in order) 
        # NOTE that actions are already normalized (this is documentation default for saving in algoConfig.offline_data())
        samplebatch_lst, samplebatch_all, samplebatch_p0term, samplebatch_e0term = [], None, None, None
        
        runs_outcome, runNames = {out:[] for out in ['I','E','M','T']}, []
        runs_outcome['M'] = ['NOT SUPPORTED (cannot be inferred)']
        for i, sb in enumerate(samplebatchGen): 
            runName = f'R{i}*' # * indicates that this is the order of the Json reader (= random)
            
            samplebatch_lst.append(sb)
            samplebatch_all = SampleBatch.concat(samplebatch_all, sb) if i != 0 else sb
            trunc = any(sb['p0'].get('truncateds'))
            trunc = any(sb['p0'].get('truncateds'))
            if not trunc:
                p0_term = any(sb['p0'].get('terminateds'))
                e0_term = any(sb['e0'].get('terminateds'))
                if p0_term:
                    runs_outcome['I'].append(runName)
                    runName += '-I'
                    samplebatch_p0term = SampleBatch.concat(samplebatch_p0term, sb) if not isinstance(samplebatch_p0term, type(None)) else sb
                elif e0_term: 
                    runs_outcome['E'].append(runName)
                    runName += '-E'
                    samplebatch_e0term = SampleBatch.concat(samplebatch_e0term, sb) if not isinstance(samplebatch_e0term, type(None)) else sb
            else:
                runs_outcome['T'].append(runName)
                runName += '-T'
            runNames.append(runName)
        ##
        runs_N = i+1
        outcome_summary = '\n= OUTCOME SUMMARY =\n'
        for k,v in runs_outcome.items():
            outcome_summary += f'{k} runs = {len(v)}/{runs_N}:\n   {runs_outcome[k]}' + '\n'
        outcome_summary += '\n(note that * indicates that this is the Jsonreaders order, w/o guarantee for matching order outside this nb.)\n'
        print(outcome_summary)
        
        #runs_N, samplebatch_all
        
        return (samplebatch_lst, samplebatch_all, samplebatch_p0term, samplebatch_e0term), \
            (runNames, runs_outcome, outcome_summary)
            
    @staticmethod
    def SB_extract_seeds_SampleBatches(sb_lst: list) -> list:
        sb_lst_seeds, errors = [], []
        aids = list(sb_lst[0].policy_batches.keys())
        for i, sb in enumerate(sb_lst):
            try:
                seed = sb[aids[0]].get('infos')[1]['game']['seed_unique']
                sb_lst_seeds.append(seed)
            except KeyError:
                errors.append(i)
        
        all_unique = (len(set(sb_lst_seeds)) == len(sb_lst_seeds))
        if not all_unique: warnings.warn('duplicate seeds in sb-list encountered')
        if errors: print(f'Errors (N = {len(errors)}) at idx: {errors}')
        print('NOTE: Seeds returned in order of samplebatch list inputted')
        return sb_lst_seeds
    
    @staticmethod
    def SB_extract_seeds_histories(histories: list) -> list:
        his_seeds, errors = [], []
        for i, his in enumerate(histories):
            try:
                seed = his[-1]['env']['seed_unique'] #his.statics.env.
                his_seeds.append(seed)
            except KeyError:
                errors.append(i)
        
        all_unique = (len(set(his_seeds)) == len(his_seeds))
        if not all_unique: warnings.warn('duplicate seeds in histories encountered')
        if errors: print(f'Errors (N = {len(errors)}) at idx: {errors}')
        print('NOTE: Seeds returned in order of histories list inputted')
        return his_seeds
        
    
    @staticmethod
    def SB_setup_his2sb_idx_mapping(sb_seeds: list = None, his_seeds: list = None):
        sb2h_idx_mapping = {}
        
        assert (len(set(sb_seeds)) == len(sb_seeds)), \
            'duplicates in samplebatch found, remove before calling function'
        assert (len(set(his_seeds)) == len(his_seeds)), \
            'duplicates in history found, remove before calling function'
            
        h_miss = list(set(sb_seeds)-set(his_seeds))
        sb_miss = list(set(his_seeds)-set(sb_seeds))
        if len(sb_seeds) != len(his_seeds):
            # note this does not occur in case duplicates, that has been checked before
            warnings.warn('Notice missing!')
            print(f'His missing: {h_miss} | sb missing: {sb_miss}')
            
        for s, s_seed in enumerate(sb_seeds):
            match = [h for h, h_seed in enumerate(his_seeds) \
                       if s_seed == h_seed]

            sb2h_idx_mapping[s] = match[0] # return the first one
    
        h2sb_idx_mapping = {v:k for k,v in sb2h_idx_mapping.items()} # inverse
        h2sb_idx_mapping = dict(sorted(h2sb_idx_mapping.items()))
        h2sb_idx_mapping = {**h2sb_idx_mapping, **{his_seeds.index(s):None for s in sb_miss}} 
        # sb miss because these are the cases of h, where no sb exists
        return h2sb_idx_mapping 
    
    @staticmethod
    def SB_append_sb2hisTuples(histories: list = None,
                               sb_lst: list = None, 
                               h2sb_idx_mapping: dict = None):
        
        his_seeds = SampleBatchUtils.SB_extract_seeds_histories(histories)
        sb_seeds = SampleBatchUtils.SB_extract_seeds_SampleBatches(sb_lst)
        h2sb_idx_mapping = SampleBatchUtils.\
            SB_setup_his2sb_idx_mapping(sb_seeds = sb_seeds, his_seeds = his_seeds)
        
        histories_appended = []
        count = 0
        for h, his in enumerate(histories):
            sb_match =  sb_lst[h2sb_idx_mapping[h]] \
                if not isinstance(h2sb_idx_mapping[h], type(None)) else False
            his_appended = his + (sb_match,) # append tuple to sorai tuple
        
            histories_appended.append(his_appended)
            if bool(sb_match): count += 1
        print(f'Succesfully appended {count}/{len(histories)} histories')
        return histories_appended
        
#%% POLICY MAPPING & Hall OF Fame


class PolicyMapper:

    policies_to_train = ['p0', 'e0']
    functionality = {
            'selfspiel':False,
            'HoF':False,
            'HoF_TRAIN':False,
            'force':False,
                    }
    
    SELFSPIEL_STATS = ['p0',1.]
    
    HoF_STATS = ['e0',0.75, 5] # base_prob, amount
    
    FORCE_DICT = {'p0':'p0',
                  'e0':'e0'}
    
    @classmethod
    def setup_functionality(cls, force =None, selfspiel = None, HoF = None, HoF_TRAIN = None):
        
        if not isinstance(force, type(None)):
            cls.functionality['force'] = bool(force) # agent id or False
            if isinstance(force, dict): 
                cls.FORCE_DICT = force
            print(f'FORCE DICT: {cls.FORCE_DICT}') 
            
        if not isinstance(selfspiel, type(None)):
            cls.functionality['selfspiel'] = bool(selfspiel) # agent id or False
            if isinstance(selfspiel, list):
                cls.SELFSPIEL_STATS = selfspiel
                print('Selfspiel stats list provided, overriding default!')
            if cls.functionality['selfspiel']:
                cls.SELFSPIEL_NAMES = [cls.SELFSPIEL_STATS[0]] + list(set(['p0','e0'])-set([cls.SELFSPIEL_STATS[0]]))
                cls.SELFSPIEL_PROBS = [cls.SELFSPIEL_STATS[1], (1.-cls.SELFSPIEL_STATS[1])]
                
                print(f'Selfspiel stats: {cls.SELFSPIEL_STATS} (= agent, prob)\n{cls.SELFSPIEL_NAMES}\n{cls.SELFSPIEL_PROBS}')
                
        if not isinstance(HoF, type(None)):
            cls.functionality['HoF'] = bool(HoF) # agent id or False
            if isinstance(HoF, list):
                cls.HoF_STATS = HoF
                print('Hall of Fame stats list provided, overriding default!')
            if cls.functionality['HoF']:
                if not isinstance(HoF_TRAIN, type(None)):
                    cls.functionality['HoF_TRAIN'] = bool(HoF_TRAIN)
                cls.HoF_NAMES = [cls.HoF_STATS[0]]+[f'{cls.HoF_STATS[0]}HoF{x}' for x in range(cls.HoF_STATS[2])]
                cls.HoF_PROBS = [cls.HoF_STATS[1]] + [(1.-cls.HoF_STATS[1])/cls.HoF_STATS[2]]*cls.HoF_STATS[2]
                # ^ = [p_base] + [(1-p_base)/N_HoF]*N_HoF 
                print(f'Hall of Fame (TRAIN = {cls.functionality["HoF_TRAIN"]}) stats: {cls.HoF_STATS} (= agent, base_prob, amount)\n{cls.HoF_NAMES}\n{cls.HoF_PROBS}')
            
            
        print(f'PolicySelector CLASS functionality setting: {cls.functionality}')
        return cls 
    
    '''
    @classmethod
    def prepare_policy_mapping(cls, policySpecs_dict, override_dict = {}):
        
        ##
        for pol_id in policies_to_train:
            if ppo_policySpecs[pol_id]['model'] == PEGmodels.CustomTorch_DummyNullModel:
                policies_to_train.remove(pol_id)
                
        if SELFSPIEL:
            policies_to_train = [x for x in policies_to_train if x == SELFSPIEL]

        ##
        for pol_id in policies_to_train:
            if ppo_policySpecs[pol_id]['model'] == PEGmodels.CustomTorch_DummyNullModel:
                policies_to_train.remove(pol_id)
        
        if cls.
            aid_copy = cls.HoF_STATS[0]
            for i in range(cls.HoF_STATS[2])
            
            
        
        return cls.policies_to_train.copy(), policySpecs_expanded
    '''
    
    @classmethod
    def policy_mapping_fn(cls, agent_id, *args, **kwargs):
        '''
        Comments from Ray RLLib's AI:
        Q; Is it true that the policy_mapping_fn is called for every step in the episode?
        A; The policy_mapping_fn is not called for every step in the episode. 
            It is called only once per episode, when an agent first appears in the environment. 
            The selected policy through this function is kept constant throughout that episode. 
            If the policy_mapping_fn is updated during an ongoing episode, 
            the new mapping will be used from the next episode onwards.
    
        IDEA; change the policy (hyperparamters) during training, example at; 
        -> SOURCE: https://github.com/ray-project/ray/issues/7023
        '''
        ## implement logic here
        policy_id = agent_id
        
        training_bool = (not args[0].__dict__['worker'].config.in_evaluation)

        ## FORCE:
        if cls.functionality['force']:
            return cls.FORCE_DICT[agent_id] # notice return!
        
        ## SELF-SPIEL  
        if cls.functionality['selfspiel']:
            if agent_id != cls.SELFSPIEL_STATS[0]: # if not the selfspiel agent
                policy_id = np.random.choice(cls.SELFSPIEL_NAMES, p = cls.SELFSPIEL_PROBS) # choose a policy, based on probabilities

        ## Hall of Fame (HoF) implementation;
        if cls.functionality['HoF']:
            if agent_id == cls.HoF_STATS[0]:
                # if HoF agent
                if (not cls.functionality['HoF_TRAIN']) and training_bool:
                    # if (do NOT apply hof in training) and (in training mode)
                    # then return the main policy (no HoF)
                    return policy_id

                # only if all other clauses have not been met do we implement HoF
                policy_id = np.random.choice(cls.HoF_NAMES, p = cls.HoF_PROBS) # choose a policy, based on probabilities
                # ensure that none of these are trained & that they are added to 'policies' object 
        
        return policy_id
    
    @classmethod
    def prepare_HoF(cls, policySpecs_dict, override_dict = {}):
        
        aid_copy = cls.HoF_STATS[0]
        for i in range(cls.HoF_STATS[2]):
            HoFid = f'{aid_copy}HoF{i}' 
            policySpecs_dict[HoFid] = copy.deepcopy(policySpecs_dict[aid_copy])
            policySpecs_dict[HoFid].update(override_dict) 
            
        return policySpecs_dict
    
#%% CALLBACKS

class CustomCallback(DefaultCallbacks):
        '''
        
        NOTES;
        - Initial checking seems to indicate that this class is saved/pcikeled
            in its entirety when saving the algorithm. Later adjustments and algo
            restoration do not add up
        - f
        TODOS;
        - 
        '''
        DefaultFuncs = DefaultCallbacks
        ##
        custom_metrics_dict = None
        ##
        policy_restore_dict = {}
        
        ##
        
        setting_eval = False
        explore_eval = False
        CLlevel_eval = 3# 5 #False
          
        ##
        functionality = {'restore': False, 'episode':False, 
                         'curriculum':False, 'HoF':False, 'evaluate':False}
        
        '''
        print('\nHARD FORCING RESTORE!!!\n')
        self.functionality['restore'] = True
        self.policy_restore_dict['p0'] =
        #'''
        
        CURRICULUM_STATS = [6,4,2,10] # base, bias, factor, patience
        CL_patience_count = 0
        
        HoF_STATS = ['e0',5] # agent_restore, N_HoF
        HoF_Nrestores = 0

        '''
        def __call__(self, *args, **kwargs):
            # ATTENTION DO NOT USE 
            # 'hacking' function to pass ray's callable tests
            # we dont want this as the system intends to create a callback on every worker
            return self
        '''
        def aggfunc_getter(self, funcName):
            if funcName =='Mean':
                aggfunc = np.mean
            elif funcName =='Max':
                aggfunc = np.max
            elif funcName =='Min':
                aggfunc = np.min
            elif funcName =='Sum':
                warnings.warn('This does not seem to work nicely across non-equal time length episodes')
                aggfunc = np.sum
            else:
                raise Exception(f'Unknown funcname encounterd {funcName}')
            return aggfunc
        
        ## TODO MERGE THIS INTO ONE FUNC WHICH SETS ALL SETTINGS

        # todo; update entire env_config;  https://discuss.ray.io/t/update-env-config-in-workers/1831/4
        def set_setting_eval(self, setting_eval: bool):
            self.setting_eval = setting_eval
            return 

        def set_explore_eval(self, explore_eval: Union[bool, dict]):
            self.explore_eval = explore_eval
            return 

        def set_CLlevel_eval(self, CLlevel_eval: int):
            # note that if you do not set this, eval_config will be followed
            self.CLlevel_eval = CLlevel_eval
            return

        @classmethod
        def set_CLlevel_eval(cls, CLlevel_eval: int):
            # note that if you do not set this, eval_config will be followed
            cls.CLlevel_eval = CLlevel_eval
            return
        
        ## 
        @classmethod
        def setup_metrics(cls, custom_metrics_dict):
            cls.custom_metrics_dict = custom_metrics_dict
            return cls
        
        @classmethod
        def setup_restore_config(cls, policy_restore_dict = {}):
            cls.policy_restore_dict = policy_restore_dict
            print(f'Setup CLASS restore dict with {cls.policy_restore_dict}')
            return cls
        
        @classmethod
        def setup_functionality(cls, restore = None, episode = None, 
                                curriculum = None, HoF = None, evaluate = None):
            warnings.warn('\nNOTE THAT CALLING THIS FUNCTION ADJUST ALL CONFIGS (not built algos)!\n')
            if not isinstance(restore, type(None)):
                if restore:
                    cls.functionality['restore'] = True
                    print(f'\n restore dict {cls.policy_restore_dict}')
                else:
                    cls.functionality['restore'] = False
            ## part B: episode logging & train results
            if not isinstance(episode, type(None)):
                if episode:
                    cls.functionality['episode'] = True
                else:
                    cls.functionality['episode'] = False
            
            # curriculum learning is part of Part B (if its not on, CL will not be done anyway)
            if cls.functionality['episode']:
                if not isinstance(curriculum, type(None)):
                    cls.functionality['curriculum'] = bool(curriculum)
                    if isinstance(curriculum, list):
                        cls.CURRICULUM_STATS = curriculum
                        print('Curriculum stats list provided, overriding default!')
                    if cls.functionality['curriculum']:
                        print(f'Curriculum stats: {cls.CURRICULUM_STATS} (= base, bias, factor, patience)')
            else:
                if curriculum:
                    print('Callback: Curriculum argument ignored as episode is off.')
                    
            # Hall of Fame is part of Part B (if its not on, HoF will not be done anyway)
            if cls.functionality['episode']:
                if not isinstance(HoF, type(None)):
                    cls.functionality['HoF'] = bool(HoF)
                    if isinstance(HoF, list):
                        cls.HoF_STATS = HoF
                        print('Hall of Fame list provided, overriding default!')
                    if cls.functionality['HoF']:
                        print(f'Hall of Fame stats: {cls.HoF_STATS} (= agent_restore, N_HoF)')
            else:
                if HoF:
                    print('Callback: Hall of Fame argument ignored as episode is off.')
                
            ## part C: evaluation
            if not isinstance(evaluate, type(None)):
                if evaluate:
                    cls.functionality['evaluate'] = True
                else:
                    cls.functionality['evaluate'] = False
            ##
            print(f'Callback CLASS functionality setting: {cls.functionality} (CL = {cls.CLlevel_eval})')
            return cls
        
        def switch_functionality(self, restore = None, episode = None, 
                                curriculum = None, HoF = None, 
                                evaluate = None, verbose = True):
            
            '''
            Function to switch on/off certain function types. Switching off
            means we siwtch to DefaultCallbacks default functions.
            
            None means current setting is not adjusted.
            '''
            ## part A: initialize
            if not isinstance(restore, type(None)):
                if restore:
                    self.on_algorithm_init = self.PartA_on_algorithm_init
                    self.functionality['restore'] = True
                    print(f'\n restore dict {self.policy_restore_dict}')
                else:
                    self.on_algorithm_init = self.DefaultFuncs.on_algorithm_init
                    self.functionality['restore'] = False
            ## part B: episode logging & train results
            if not isinstance(episode, type(None)):
                if episode:
                    self.on_episode_created = self.PartB_on_episode_created
                    self.on_episode_start = self.PartB_on_episode_start
                    self.on_episode_step = self.PartB_on_episode_step
                    self.on_episode_end = self.PartB_on_episode_end
                    self.on_train_result = self.PartB_on_train_result
                    self.functionality['episode'] = True
                else:
                    self.on_episode_created = self.DefaultFuncs.on_episode_created
                    self.on_episode_start = self.DefaultFuncs.on_episode_start
                    self.on_episode_step = self.DefaultFuncs.on_episode_step
                    self.on_episode_end = self.DefaultFuncs.on_episode_end
                    self.on_train_result = self.DefaultFuncs.on_train_result
                    self.functionality['episode'] = False
                    
            # curriculum learnin is part of Part B (if its not on, CL has will not be done anyway) 
            if self.functionality['episode']:
                if not isinstance(curriculum, type(None)):
                    self.functionality['curriculum'] = bool(curriculum)
                    if isinstance(curriculum, list):
                        self.CURRICULUM_STATS = curriculum
                        print('Curriculum stats list provided, overriding default!')
                    if self.functionality['curriculum']:
                        print(f'Curriculum stats: {self.CURRICULUM_STATS} (= base, bias, factor, patience)')
            else:
                if verbose:
                    print('Callback: Curriculum argument ignored as episode is off.')
                    
            # Hall of Fame is part of Part B (if its not on, HoF will not be done anyway)
            if self.functionality['episode']:
                if not isinstance(HoF, type(None)):
                    self.functionality['HoF'] = bool(HoF)
                    if isinstance(HoF, list):
                        self.HoF_STATS = HoF
                        print('Hall of Fame list provided, overriding default!')
                    if self.functionality['HoF']:
                        print(f'Hall of Fame stats: {self.HoF_STATS} (= agent_restore, N_HoF)')
            else:
                if verbose:
                    print('Callback: Hall of Fame argument ignored as episode is off.')
                    
                    
            ## part C: evaluation
            if not isinstance(evaluate, type(None)):
                if evaluate:
                    self.on_evaluate_start = self.PartC_on_evaluate_start
                    self.on_evaluate_end = self.PartC_on_evaluate_end
                    self.functionality['evaluate'] = True
                else:
                    self.on_evaluate_start = self.DefaultFuncs.on_evaluate_start
                    self.on_evaluate_end = self.DefaultFuncs.on_evaluate_end
                    self.functionality['evaluate'] = False
            ## 
            if verbose:
                print(verbose*f'Callback INSTANCE functionality setting: {self.functionality} (CL = {self.CLlevel_eval})')
            return

        ## Part 0: initialization 
        def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
            if legacy_callbacks_dict:
                pass
            
            '''
            print('\nHARD FORCING RESTORE!!!\n')
            self.functionality['restore'] = True
            self.policy_restore_dict['p0'] =
            #'''
            
            self.DefaultFuncs = self.DefaultFuncs() 
            # initialize dfeault funcs class, so that they have access to self
            self.switch_functionality(**self.functionality, verbose=True)
            
            np.random.seed(8147)
            return 
                

        ## PART A: algo initialization, BC
        def PartA_on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
    
            '''
            algo_state = ....
            strip_resources(algo_state)
            Algorithm.from_state(algo_state)
            '''
            algo_restore_weights = {}
            for aid in self.policy_restore_dict:
                '''
                algo_state = ....
                strip_resources(algo_state)
                Algorithm.from_state(algo_state)
    
                TODO MAYBE RESTORING A POLICY IS EASIER, less overhead since it does not load in workers
                and we only need weights anyway
                '''
                policy_restore_aid = Policy.from_checkpoint(self.policy_restore_dict[aid], policy_ids = [aid])[aid]
                # Policy.from_checkpoint returns a {aid:policy}
                
                #algo_restore_aid = Algorithm.from_checkpoint(...) 
                # ensure this checkpoint has been stripped of resources, 
                # otherwise we need to do some logic here (overriding state, see above)
                
                algo_restore_weights[aid] = policy_restore_aid.get_weights()
                
            ## get and set/override weights ON LOCAL (OPTIMIZER) WORKER   
            algorithm.set_weights(algo_restore_weights)
            
            #'''
            # OVERRIDE WEIGHTS ON ALL OTHER WORKERS
            def pol_setW_func(pol, pol_id):
                w = algo_restore_weights.get(pol_id,{})
                if w:
                    pol.set_weights(w)
                return 
            algorithm.workers.foreach_worker(
                        lambda ev: ev.foreach_policy(pol_setW_func))
            #'''
            return 
        
        
        ## PART B: episode & training related
        def PartB_on_episode_created(
            self,
            *,
            episode, worker = None, env_runner = None,
            base_env = None, env= None,
            policies = None, rl_module = None, env_index: int,
            **kwargs,
        ) -> None:
            """Callback run when a new episode is created (but has not started yet!).
    
            This method gets called after a new Episode(V2) (old stack) or
            SingleAgentEpisode/MultiAgentEpisode instance has been created.
            This happens before the respective sub-environment's (usually a gym.Env)
            `reset()` is called by RLlib.
    
            1) Episode(V2)/Single-/MultiAgentEpisode created: This callback is called.
            2) Respective sub-environment (gym.Env) is `reset()`.
            3) Callback `on_episode_start` is called.
            4) Stepping through sub-environment/episode commences.
    
            Args:
                episode: The newly created episode. On the new API stack, this will be a
                    SingleAgentEpisode or MultiAgentEpisode object. On the old API stack,
                    this will be a Episode or EpisodeV2 object.
                    This is the episode that is about to be started with an upcoming
                    `env.reset()`. Only after this reset call, the `on_episode_start`
                    callback will be called.
                env_runner: Replaces `worker` arg. Reference to the current EnvRunner.
                env: Replaces `base_env` arg.  The gym.Env (new API stack) or RLlib
                    BaseEnv (old API stack) running the episode. On the old stack, the
                    underlying sub environment objects can be retrieved by calling
                    `base_env.get_sub_environments()`.
                rl_module: Replaces `policies` arg. Either the RLModule (new API stack) or a
                    dict mapping policy IDs to policy objects (old stack). In single agent
                    mode there will only be a single policy/RLModule under the
                    `rl_module["default_policy"]` key.
                env_index: The index of the sub-environment that is about to be reset
                    (within the vector of sub-environments of the BaseEnv).
                kwargs: Forward compatibility placeholder.
            """
            
            '''
            
            TODO RESET ENV SEED HERE, this is before .reset()! but why 
            wouldnt CL change not work (i.e. we do we need the padding)?
            
            PREVIOUS ATTEMPT
            
            def do_func(env):

                ###
                NOTE THIS IS NOT A PROPER RESET OF SEED COUNTER
                this is because .RESET() has apparently already been called
                before this cell. This gives example output for a single worker
                    [ prev_seed_unique_counter_of_env,   0,   1,   2, ...]
                    
                TODO; improve;
                - doing this through callback will not work (i think) because
                    this cell already does it before .evaluate() and the callback
                    acts within .evaluate(); so it has to happen either at
                    or even before (= impossible) on_evalaute_end()
                - FOLLOW UP; best spot to reset seed is at getNwipe_histories_all, but
                     then you can not call .evaluate() multiple times (since
                     it will be reset)
                    - potentially this would be fine if we have remote workers
                    and can use duration func (so we dont need to call multiple times)
                ###
                env.seed_unique = int(env.seed_unique0)
                
                return env.seed_unique

            a = algo.evaluation_workers.foreach_worker( # NOTICE WORKER SET TYPE
                                        lambda ev: ev.foreach_env(
                                            lambda env: do_func(env)))
            a
            '''
            return
        
        def PartB_on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
        ):
            # Make sure this episode has just been started (only initial obs
            # logged so far).
            '''
            # THIS FAILS BUT IS THEIR EXAMPLE? indeed the counter seems to start at -1
            assert episode.length == 0, (
                "ERROR: `on_episode_start()` callback should be called right "
                "after env reset!"
            )
            '''
            #func1 = lambda pol, id: (pol, id)
            #OUT = worker.foreach_policy(func1)
            #assert False, f'pol= \n {policies}\n OUT = {OUT}'
            '''
            out = algorithm.

            '''
            
            #assert False, f'START: {episode.length}'
            # game state 
            for key in self.custom_metrics_dict['game']:
                episode.user_data[key] = []
                #episode.hist_data[key] = []

            ## agent state specifics
            for aid in self.custom_metrics_dict['agent_id']: 
                episode.user_data[aid] = {}
                for key in self.custom_metrics_dict['agent']:
                    episode.user_data[aid][key] = []
            return

        def PartB_on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
        ):
            '''
            # THIS FAILS BUT IS THEIR EXAMPLE?
            # Make sure this episode is ongoing.
            assert episode.length > 0, (
                "ERROR: `on_episode_step()` callback should not be called right "
                "after env reset!"
            )
            #'''
            #assert False, f'STEP: {episode.length} KEYS: {list(episode.user_data.keys())}'
            #if episode.length > 1: # work around, means initial state is lost

            ## game (accessible for any agent)
            game_info = episode.last_info_for(self.custom_metrics_dict['agent_id'][0])['game']
            for key in self.custom_metrics_dict['game']:
                episode.user_data[key].append(game_info[key])
            
            ## agent specific
            for aid in self.custom_metrics_dict['agent_id']: 
                aid_info = episode.last_info_for(aid)['agent']
                for key in self.custom_metrics_dict['agent']:
                    episode.user_data[aid][key].append(aid_info[key])
            # TODO AGENT SPECIFIC
            return
            
        def PartB_on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv, #`base_env.get_sub_environments()`.
            #env : gym.Env = None,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
        ):
            # NOTE THIS IS FOR EVERY EPISODE, ITS NOT A SINGLE RESULT!
            
            '''
            # THIS FAILS BUT IS THEIR EXAMPLE?
            # Check if there are multiple episodes in a batch, i.e.
            # "batch_mode": "truncate_episodes".
            if worker.config.batch_mode == "truncate_episodes":
                # Make sure this episode is really done.
                assert episode.batch_builder.policy_collectors["default_policy"].batches[
                    -1
                ]["dones"][-1], (
                    "ERROR: `on_episode_end()` should only be called "
                    "after episode is done!"
                )
            '''
            # aggregates over all timesteps within an episode (i.e. not over multiple episodes)
            
            pol_Ap0, pol_Ae0 = episode.policy_for("p0"), episode.policy_for("e0")[:5] # :5 cuts off number of HoF
            
            polMap = f'{pol_Ap0}V{pol_Ae0}'
            
            ## game 
            #  REDOLOG og episode.custom_metrics[f'game_{key}{value}']
            for key,value in self.custom_metrics_dict['game'].items():
                episode.custom_metrics[f'{polMap}_game_{key}{value}'] = \
                    self.aggfunc_getter(value)(episode.user_data[key])
                #episode.hist_data[key] = episode.user_data[key]
                
            ## agent specific
            #  REDOLOG og episode.custom_metrics[f'{aid}_{key}{value}']
            for aid in self.custom_metrics_dict['agent_id']: 
                for key, value in self.custom_metrics_dict['agent'].items():
                    episode.custom_metrics[f'{polMap}_{aid}_{key}{value}'] = \
                        self.aggfunc_getter(value)(episode.user_data[aid][key])

            ## 
            '''
            NOTE:
            custom_metrics does not have agent_id's nested dist (e.g. dict[e0][metric]) setup 
            and also I do not know if subsequent agg efforts will work
            '''
            return


        def PartB_on_train_result(self, algorithm, result, **kwargs):
            ''' 
            WOULD BE NICE TO ONLY DO THIS BASED ON EVALUATION
    
            then use on_evaluation_result! -> probably NOT possible
            since the evaluation workers are spawned indepedent to training
            workers so if we change task level it will only be for eval workers 
                -> I should try this anyway
    
            PROBLEMS;
            - you might switch back levels in case result drops between tasks 
                this could imply a loop between reaching level & dropping
            -
            -
            ''' 
            ##############
            ## Custom metrics
            # aggregates over all episodes, deleting episode specific info in the process
            # TODO WOULD BE NICE TO HAVE HIST DATA OVER THESE EPISODES
            '''
            custom metrics defined in on_episode_end have already mean, max and min

            TODO; this is because they are considered standard metrics, there should
            be a way to switch this off, see the custom eval?
            '''
            #assert False, f'custom metrics {list(result["custom_metrics"].keys())}'
            #result["custom_metrics"]["Gamma_mean"] = np.mean(result["custom_metrics"]["Gamma_mean"])
            #result["custom_metrics"]["BS_bool_mean_e"] = np.mean(result["custom_metrics"]["BS_bool_mean_e"])
            #result["custom_metrics"]["BS_lag_max_e"] = np.mean(result["custom_metrics"]["BS_lag_max_e"])
            
            ##   
            if self.functionality['curriculum']:
                self.PartB_curriclum_learning(algorithm, result)
            ##
            if self.functionality['HoF']:
                self.PartB_HallofFame(algorithm, result)
            
            return
        
        def PartB_curriclum_learning(self, algorithm, result):
            '''
            
            TODO MAKE A CURRICULUM_CONFIG WITH THE BIAS, BASE ETC. 
            even if default so that we can get it back
            -> NOTE THAT THIS WILL NOT BE SAVED THOUGH!
            '''
            ## get task levels
            # NOTE THAT THE FOLLOWING DOES NOT GET THE CL_LEVEL FOR EVERY ENV ENCOUNTERED
            # BUT ASSESSES EVERY WORKER I.E. WE DONT GET n_episodes BACK, BUT num_env_per_w*num_worker
            task_levels = algorithm.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.get_task())) # [[empty GPU],*([[task_level]*num_env_per_w]*(num_worker))]
            task_levels = np.unique(task_levels[1:]) # cutoff empty gpu
            assert len(task_levels) <= 1, f'Multiple task levels encountered: {task_levels}' # <= instead == 0 due to worker amount 
            CL_TaskLevel = task_levels[0] # check passed, unlist
            result['custom_metrics']['CL_level'] = CL_TaskLevel # save task level of sampled iteration
            
            ## assess reward & define task level
            '''
            CL_bias = 4 # 6 DEFAULT, 
            # note that if zerosum escape vs intercept then getting reward ~10 is not likely
            CL_base = 6#8
            CL_factor = 2
            #'''

            CL_base, CL_bias, CL_factor, CL_patience = self.CURRICULUM_STATS 
            bias = max(CL_bias - CL_factor*CL_TaskLevel,0) # >0
            ''' 
            
            policy_reward = result['policy_reward_mean']['p0'] + bias # add bias, to avoid perfect score requirement
            CL_TaskLevel_new = min(policy_reward // CL_base + CL_TaskLevel, CL_TaskLevel + 1) # recursive structure, clipped at +1 per step
            ''' 
            
            rate_INT = result['custom_metrics']['game_INToutcomeMax_mean'] + bias/10.
            rate_thres = (CL_base)/10.
            CL_TaskLevel_new = CL_TaskLevel + int((rate_INT > rate_thres))
            #'''
            ## set task level
            #CL_TaskLevel_new = max(CL_TaskLevel_new,CL_TaskLevel) # no drop in level possible
            if CL_TaskLevel_new > CL_TaskLevel: # no drop in level possible
                self.CL_patience_count += 1 # increment patience counter
                if self.CL_patience_count >= CL_patience:
                    self.CL_patience_count = 0 # reset counter
                    
                    # upgrade CL level
                    algorithm.workers.foreach_worker(
                        lambda ev: ev.foreach_env(
                            lambda env: env.set_task(CL_TaskLevel_new)))
            else:
                # CL threshold not met (this loses any progress made!)
                self.CL_patience_count = 0  # reset counter
                
            result['custom_metrics']['CL_patience'] = self.CL_patience_count
            return 
            
        
    
        def PartB_HallofFame(self, algorithm, result):

            aid_restore, N_HoF = self.HoF_STATS.copy()

            
            checkpoint_dir = algorithm.logdir
            ## get all current checkpoint names
            checkpoint_names = np.array([f.path for f in os.scandir(checkpoint_dir) if f.is_dir()])
            checkpoint_N = len(checkpoint_names)
            
            if (checkpoint_N > self.HoF_Nrestores): # if more than last time
                self.HoF_Nrestores = min(checkpoint_N,N_HoF)
                checkpoint_nums = [int(x[-6:]) for x in checkpoint_names] # get checkpoint numbers
                checkpoint_nums = np.unique(np.linspace(1,len(checkpoint_nums)+1,
                                                        num = N_HoF, endpoint = False, dtype = np.int64))*-1
                # select 5 evenly spaced checkpoints (given that there are enough), select from the most recent
                #'''
                # randomization of restoration
                if checkpoint_N >= 15:
                    s = max(1, ((checkpoint_N-15) // 10)) # (25,1), (35, 2), ...
                    #max(1, (25-15) // 10)
                    randomizer = np.array([0, *np.random.randint(low=-1*s,
                                                                high = 1*s, 
                                                                size = 4)])
                    # randomizer to ensure some shifting in agent dynamics
                    # note that shifting already occurs due to lengthing of checkpoint_nums
                    checkpoint_nums += randomizer 
                #'''
                
                checkpoint_names = checkpoint_names[checkpoint_nums] # get checkpoints names from selection
                
                algo_restore_weights = {}
                algo_restore_map = {}
                for i, checkpoint_aidHoF in enumerate(checkpoint_names):
                    # note that not all HoF policies might be filled if N_names < N_pols
                    aidHoF = f'{aid_restore}HoF{i}' # start with filling at HoF = 0
                
                    # restore policy from checkpoint
                    policy_restore_aidHoF = Policy.from_checkpoint(checkpoint_aidHoF, 
                                                policy_ids = [aid_restore])[aid_restore]
                    # Policy.from_checkpoint returns a {aid:policy}
    
                    algo_restore_weights[aidHoF] = policy_restore_aidHoF.get_weights()
                    algo_restore_map[aidHoF] = checkpoint_aidHoF[-17:]
                
                ##
                result['custom_metrics']['HoF_Nrestores'] = len(algo_restore_weights) #self.HoF_Nrestores
                '''
                algorithm.set_weights(algo_restore_weights) # note that this is the algo, not on a worker!
                # I assume that the algorithm in the next run will update all workers with weights
                # ^this is might be true but I dont know for sure therefore set all policies manually (see next clause)!
                '''
                ## set/override weights ON LOCAL (OPTIMIZER) WORKER   
                algorithm.set_weights(algo_restore_weights)

                # OVERRIDE WEIGHTS ON ALL OTHER WORKERS
                def pol_setW_func(pol, pol_id):
                    w = algo_restore_weights.get(pol_id,{})
                    if w:
                        pol.set_weights(w)
                    return 
                algorithm.workers.foreach_worker(
                            lambda ev: ev.foreach_policy(pol_setW_func))
                #'''
                
                '''
                TODO TRACK METRICS
                '''
                print(f'Hall of Fame| current policies:\n{algo_restore_map}')
            return 
        
        '''
        def update_HoF_policies(self, pol, pol_id):
            ## get and set/override weights    
            algorithm.set_weights(self.algo_restore_weights)
            return 
        '''
        
        
        ## PART C: EVALUATION RELATED
        def change_policy_explore_setting(self, pol, pol_id):
            '''
            Note that I deem it very likely that every worker 
            has a 'current' copy of the policy since ray rllib
            offers a lot of asyncronous algorithms. These will never
            be connected to the true one. Hence it is required/doable
            to adjust the policies within every worker!

            NOTE THAT EVEN IF YOU HAVE NO REMOTE WORKERS 
            (I.E. LOCAL ONLY, e.g. on gpu+cpu main bundle) then still these use a copy
            of the policy objects and not the main instance. Hence you can adjust the 
            config settings (e.g. explore) without adjusting the main algo.
            To test use:
                # main, local policy
                policy = algo.get_policy('p0')
                pol_state = policy.get_state()
                pol_main_explores = pol_state['policy_spec']['config']['explore']

                # eval workers policy
                pol_ew_explores = algo.\
                            evaluation_workers.foreach_worker(
                                lambda ev: ev.foreach_policy(
                                    lambda env, a: \
                                    env.get_state()['policy_spec']['config']['explore']))
            -> NOTE THIS IS THE REASON I DO NOT BELIEVE WE EVER HAVE TO RESTORE IT
             (if in future this is so, you can create self.restore_dict and a restore func 
              and call it during on_evaluate_end)
        
            IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
            policy, even if this is a stochastic one. Setting "explore=False" here
            will result in the evaluation workers not using this optimal policy! 
            - WHY? because the optimal policy might be stochastic and not 
                deterministic i.e. sometimes is best to do random!
            '''
            pol_state = pol.get_state()
            if isinstance(self.explore_eval, dict):
                # agent specific eval setting
                pol_state['policy_spec']['config']['explore'] = self.explore_eval.get(pol_id,False)
            elif isinstance(self.explore_eval, bool):
                # general agent eval setting
                pol_state['policy_spec']['config']['explore'] = self.explore_eval

            pol.set_state(pol_state)
            return 
        
        def PartC_on_evaluate_start(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
            ) -> None:
            """Callback before evaluation starts.
            
            This method gets called at the beginning of Algorithm.evaluate().
            and (IMPORTANT) seemingly before env.reset is first called!

            IMPORTANT apperently using callbacks is the only way to
            properly change/set attributes of the environments (as in lines below)
            if we do the same outside this callback then either;
            (1) reset has already been called for the first trial in each workers 
                so the attribute change comes to late (i.e. after instead of before 
                the reset call)
            
            (2) the attribute is completely ignored e.g. for use of the following outside
            the callback (i.e. just any random cell) we dont see change in CL level;
                algorithm.evaluation_workers.foreach_worker(
                        lambda ev: ev.foreach_env(
                            lambda env: env.set_task(self.CLlevel_eval)))
            
            Args:
                algorithm: Reference to the algorithm instance.
                kwargs: Forward compatibility placeholder.
            """
            ## switch off exploration setting ALWAYS
            # (see comment on accessing policies above)
            
            algorithm.evaluation_workers.foreach_worker(
                        lambda ev: ev.foreach_policy(
                            self.change_policy_explore_setting))
            ##
            # todo; update entire env_config;  https://discuss.ray.io/t/update-env-config-in-workers/1831/4
            if self.setting_eval:
                algorithm.evaluation_workers.foreach_worker(
                        lambda ew: ew.foreach_env(
                            lambda env: env.set_setting_eval(True)))
            if isinstance(self.CLlevel_eval, int): 
                # if not False (bool), 0 is allowed
                algorithm.evaluation_workers.foreach_worker(
                        lambda ew: ew.foreach_env(
                            lambda env: env.set_task(self.CLlevel_eval)))
            return 
            
        def PartC_on_evaluate_end(
                self,
                *,
                algorithm: "Algorithm",
                evaluation_metrics: dict,
                **kwargs,
            ) -> None: 
            '''

            task_levels = algorithm.evaluation_workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.get_task())) # [[empty GPU],*([[task_level]*num_env_per_w]*(num_worker))]
            task_levels = np.unique(task_levels[1:]) # cutoff empty gpu
            assert len(task_levels) <= 1, f'Multiple task levels encountered: {task_levels}' # <= instead == 0 due to worker amount 
            CL_TaskLevel = task_levels[0] # check passed, unlist
            evaluation_metrics['custom_metrics']['CL_level'] = CL_TaskLevel # save task level of sampled iteration
            #'''            
            # SWITCH BACK EVAL? no because we only adjusted the eval workers 
            # not the other workers.
            ##
            if self.setting_eval:
                '''
                env_settings = algorithm.evaluation_workers.foreach_worker(
                    lambda ev: ev.foreach_env(
                        lambda env: env.setting_eval)) # sorai
                '''
                
                env_histories = algorithm.evaluation_workers.foreach_worker(
                    lambda ew: ew.foreach_env(
                        lambda env: env.getNwipe_histories_all()))[0] # sorai
                # [0] gives nested list with len = [workers*[envs_per_worker*history]] 
                env_histories = [h for w in env_histories for h in w] # unpack 
                
                evaluation_metrics["env_histories"] = env_histories
                # TODO ADD POLICY INFORMATION?

                ## revert eval setting in case of future use
                # this seems unnecessary, but is nice as it returns the object to original state 
                # also note that this does not remove information added during eval setting e.g. histories, seeds etc.
                algorithm.evaluation_workers.foreach_worker(
                            lambda ew: ew.foreach_env(
                                lambda env: env.set_setting_eval(False)))

            return 


''' 
TODO REMOVE

def on_algorithm_init_PURSUER(self, *, algorithm: "Algorithm", **kwargs) -> None:

    BC_AIDS = ['p0']
    #CHECKPOINT_DIR = 'BC/runs/TEST_3D2DOF_1'
    
        
    
    if self.FIRST_INIT:
        self.FIRST_INIT = False
        policy = Policy.from_checkpoint(self.CHECKPOINT_DIR, policy_ids = BC_AIDS) 
        self.CHECKPOINT_DIR = None
        # FIRST_INIT AND CHECKPOINT_DIR ARE TO ENSURE WE DONT LOAD BC DURING RESTORATION
        
        # restoring policy is quickest, does not have to initiate entire algorithm ecosystem
        #policy['p0'].get_weights() # dict with keys and values from model.named_parameters(), note that matrices are in numpy float32
        for aid in BC_AIDS:
            algorithm.set_weights({aid: policy[aid].get_weights()})

        if False: #'e0' in self.policies_to_train:
            CHECKPOINT_DIR_e0 = FFF
            policy_e0 = Policy.from_checkpoint(CHECKPOINT_DIR_e0, policy_ids = 'e0') 
            algorithm.set_weights({aid: policy_e0['e0'].get_weights()})
    return

def on_algorithm_init_ALL(self, *, algorithm: "Algorithm", **kwargs) -> None:

    #BC_AIDS = ['p0','e0']
    #CHECKPOINT_DIR = 'BC/runs/TEST_3D2DOF_1'
    BC_AIDS = self.policies_to_train
    
    if self.FIRST_INIT:
        self.FIRST_INIT = False
        policy = Policy.from_checkpoint(self.CHECKPOINT_DIR, policy_ids = BC_AIDS) 
        self.CHECKPOINT_DIR = None
        # FIRST_INIT AND CHECKPOINT_DIR ARE TO ENSURE WE DONT LOAD BC DURING RESTORATION
        

        for aid in BC_AIDS:
            algorithm.set_weights({aid: policy[aid].get_weights()})

    return
'''
    


#%% BEHAVIOURAL CLONING





#%% RESTORATION




#%% CRITIC ANALYSIS & 

class LightningLearner(pl.LightningModule):
    def __init__(self, model, loss_func, lr = 0.01, RL = False):
        super().__init__()
        '''
        README; we assume that all arguments have already been 
        configured/initialized.
        
        TOOD;
        - see custom trainer; incorporate metrics computation and
        setup & saving of models status report, global config etc
        
        '''
        self.model = model
        self.lr = lr
        self.loss_func = loss_func
        return 

    def training_step(self, batch, batch_idx): 
        #x, w, y = batch # todo, incorporate weights into loss
        x, y = batch # todo, incorporate weights into loss
        #y = y[:,-6:-3] # mse
        ##
        assert self.model.training
        # no call to model.train() to speed up training
        
        y_hat = self.model(x)
        #y_hat = y_hat.view_as(y)
        loss = self.loss_func(y_hat, y)
        
        ##
        self.log("Loss_train", loss, 
                 on_step=False, on_epoch=True, 
                 prog_bar=True,
                 )
        # PLOT LOSSES IN SAME FIGURE https://github.com/Lightning-AI/pytorch-lightning/issues/665
        return loss #{"loss": loss}


    def validation_step(self, batch, batch_idx):
        #x, w, y = batch # todo, incorporate weights into loss
        x, y = batch # todo, incorporate weights into loss
        #y = y[:,-6:-3] # mse
        ##
        self.model.eval()
        with torch.no_grad():

            y_hat = self.model(x)
            #y_hat = y_hat.view_as(y)
            loss = self.loss_func(y_hat, y)

        self.model.train()
        ##
        self.log("Loss_validation", loss,
                 on_step=False, on_epoch=True, 
                 prog_bar=False,
                 )
        # PLOT LOSSES IN SAME FIGURE https://github.com/Lightning-AI/pytorch-lightning/issues/665
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        ## already configured
        return torch.optim.Adam(self.model.parameters(), lr=self.lr) # lr = 0.005

    
    def predict(self, x, numpy_out = False):
        
        
        if hasattr(self.model, 'predict'):
            y_hat = self.model.predict(x)
            
        else:
            self.model.eval()
            with torch.no_grad():
                
                y = self.model(x)
                if isinstance(y, tuple):
                    # rnn type, 
                    y, _ = y
                    y = y[:,-1,:] # select last timestep from recurrent network
            if numpy_out:
                y = y.numpy()
        return y
