from typing import Callable, Any, Iterable, Union, List, Tuple, Set, Dict


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots

import warnings

from tqdm import tqdm
from datetime import datetime
import scipy as sc

import copy

from scipy.spatial.transform import Rotation as R

#%% ANALYSIS CONDUCTORS (carries out an entire analysis)

class Analysis:
    
    OUTCOMES_LABELS = ['I','E','M','T']
    
    @staticmethod
    def run_safe_unpack(run):
        ''' 
        Function 'post-pends' tuple with false's to
        get desired length
        ''' 
        # run awlays has s.o.r.a.i. & statics = 6
        # sb makes 7
        desired_len = 7
        run += -1*min(len(run)-desired_len,0)*(False,)
        
        return run
        
    def thin_timeseries(s,o,r,a,i, thin = 1):
        
        s = s.copy()[::thin,...]
        dict_containers = [o.copy(),r.copy(),a.copy(),i.copy()]
        for idx, dct in enumerate(dict_containers):
            for key in dct.keys():
                dct[key] = dct[key][::thin,...]
            dict_containers[idx] = dct
        o,r,a,i = dict_containers
        return s,o,r,a,i
    
    
    def TEMPLATE_runs(cls, histories: list, condition: Union[list, str] = '', 
                                      runNames: list = None):

        runs_outcome = {out:[] for out in Analysis.OUTCOMES_LABELS} 
        runs_N = len(histories)
        count_check = 0
        for run, run_history in tqdm(enumerate(histories.copy())):
            ## unpack & determine outcome
            s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history)
            
            check, outcome_label = Analysis.condition_outcome(statics['end'],
                                                              condition)
            if runNames:
                # predefined run name
                runName = runNames[run]
            else:
                # generic run name
                runName = f'R{run}'
            
            runs_outcome[outcome_label].append(runName) # track outcome
        
            ## visualize if condition is met
            if check:
                count_check += 1
                
                ''' 
                NEW CODE HERE
                '''
                out = []
                
        ## print outcome summary
        print(f'Filtered on condition: "{condition}", left = {count_check}/{runs_N}.\nPotential outcomes: {Analysis.OUTCOMES_LABELS}')
        for k,v in runs_outcome.items():
            print(f'{k} runs = {len(v)}/{runs_N} = {len(v)/runs_N:.2}:\n{runs_outcome[k]}')
        
        return out, runs_outcome
    
    
    @staticmethod
    def condition_outcome(outcome: dict, condition: Union[list, str]) -> tuple:
        outcome_label = [k for k, v in outcome.items() if v][0] 
        
        if isinstance(condition,list):
            raise Exception('multiple conditions not supported yet')
        
        if condition not in Analysis.OUTCOMES_LABELS:
            return True, outcome_label
        else:
            check = (outcome_label == condition)
            return check, outcome_label
    
    @staticmethod
    def filter_histories(histories: list, condition: Union[list, str]) -> list:
        
        selected_histories = []
        for run, run_history in enumerate(histories.copy()):
            s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history)
            if not Analysis.condition_outcome(statics['end'], condition)[0]:
                continue
            else:
                selected_histories.append(run_history)
            
        print(f'runs left for cond = {condition}: {len(selected_histories)}/{len(histories)}')
        return selected_histories 
    
    #%% Analysis.VISUALIZATION
    
    ## class variables
    NB_SETTINGS ={
    'color':{'p0':'blue',
             'p0BS':'green',
             'e0':'red',
             'e0BS':'orange',
             'dis':'lightgreen', #limegreen
            },
             
    }
    
    COLORS = ['blue','green','red','cyan','magenta' ,'yellow',
        'black',#'white',
        'gray' ,'purple','orange','pink','brown','gold',
        'lime','navy',]
    COLORS_CODE = ['#0000FF','#008000','#FF0000', '#00FFFF',
        '#FF00FF','#FFFF00','#000000',#'#FFFFFF',
        '#808080','#800080,''#FFA500','#FFC0CB',
         '#A52A2A''#FFD700','#00FF00','#000080','#FF6347', '#D2B48C', '#D8BFD8', 
         '#ADFF2F', '#FA8072', '#F0E68C', '#E6E6FA', '#FFFACD']

    ##
    
    @classmethod
    def visualize_trajectories_runs(cls, histories: list, condition: Union[list, str] = '', 
                                      show_los: int = 0,
                                      title: str = 'Trajectories',
                                      velocity_cones = False,
                                      attitude_cones = False,
                                      acceleration_cones = False,
                                      min_distance = True,
                                      thin = 0,
                                      runNames: list = None):
        ''' 
        Visualized and summarizes outcomes across runs 
        
        show_los is used in range(0,len(s), SHOW_LOS) and determines
        whether Line-of-sight is shown (SHOW_LOS>0) and if so with what frequency
        ''' 
        assert sum([velocity_cones, attitude_cones, acceleration_cones]) <= 1, 'choose one type of cones'
        warnings.warn('browserfig instance returned!')
        runs_outcome = {out:[] for out in Analysis.OUTCOMES_LABELS} 
        runs_N = len(histories)

        title += ' - ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")  
        
        if thin:
            warnings.warn(f'ATTENTION: thinning time with {thin}')

        fig = pgo.Figure()
        do_bs = histories[0][5]['env']['DO_BS']
        count_check = 0
        for run, run_history in tqdm(enumerate(histories.copy())):
            ## unpack & determine outcome
            s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history)
            
            if not (statics['env']['DO_BS'] == do_bs):
                raise Exception('Inconsistent BS setting across episodes not supported ')    
            do_bs = statics['env']['DO_BS']
            
            
            check, outcome_label = Analysis.condition_outcome(statics['end'],
                                                              condition)
            if runNames:
                # predefined run name
                runName = runNames[run]
            else:
                # generic run name
                runName = f'R{run}'
            
            runs_outcome[outcome_label].append(runName) # track outcome

            if thin:

                assert sb is False, 'not incorporated with sb'
                s,o,r,a,i = cls.thin_timeseries(s,o,r,a,i, thin = thin)
                
            ## visualize if condition is met
            if check:
                count_check += 1
                runName_Fig = runName + '-' + outcome_label
        
                        
                for aid_idx, aid in enumerate(o.keys()):
                    fig = Visualization.setup_3dplotly(fig, s[:,aid_idx,0,:3], aid,cls.NB_SETTINGS['color'][aid], title, 
                                                              legendgroup = runName_Fig, BrowserFig = True, 
                                                              )
                
                    
                    if velocity_cones:
                        fig = Visualization.setup_3dplotly(fig, s[:,aid_idx,0,:3], aid,cls.NB_SETTINGS['color'][aid], title, 
                                                                  legendgroup = runName_Fig, BrowserFig = True, showlegend = False,
                                                                  dxyz = s[:,aid_idx,1,:3], # <- for cone info (= velocity)
                                                                  )
                    elif attitude_cones:
                        
                        angles = s[:,aid_idx,0,3:]
                        RotMat_BI = R.from_euler('zyx', 
                                              angles[:,::-1]*-1,
                                              degrees = False)
                        
                        thrust_orientation = [0.,0.,1.]
                        cones = RotMat_BI.apply(np.repeat([thrust_orientation],len(angles), axis = 0))
                        
                        fig = Visualization.setup_3dplotly(fig, s[:,aid_idx,0,:3], aid,cls.NB_SETTINGS['color'][aid], title, 
                                                                  legendgroup = runName_Fig, BrowserFig = True, showlegend = False,
                                                                  dxyz = cones, # <- for cone info (= attitude unit vector)
                                                                  )
                    elif acceleration_cones:
                        
                        cones = s[:,aid_idx,2,3:]
                        
                        fig = Visualization.setup_3dplotly(fig, s[:,aid_idx,0,:3], aid,cls.NB_SETTINGS['color'][aid], title, 
                                                                  legendgroup = runName_Fig, BrowserFig = True, showlegend = False,
                                                                  dxyz = cones, # <- for cone info (= acceleration)
                                                                  )
                    
                        
                    fig = Visualization.setup_3dplotly(fig, s[[0],aid_idx,0,:3], aid+'ini',cls.NB_SETTINGS['color'][aid], title, 
                                                              marker_size = 5, showlegend = False,mode = 'markers', legendgroup = runName_Fig, BrowserFig = True)
                    if do_bs:
                        # ADD BLINDSIGHTING LABELS
                        i_bs_aid = i[aid][:,[-(2+4+1)]]
                        i_bs_aid[i_bs_aid == 0] = np.nan
                        s_bs_aid = s[:,aid_idx,0,:3]*i_bs_aid
                        fig = Visualization.setup_3dplotly(fig, s_bs_aid, aid+'BS',cls.NB_SETTINGS['color'][aid+'BS'], title, marker_size = 5, 
                                                                  showlegend = False,mode = 'markers', legendgroup = runName_Fig, BrowserFig = True)
                ## 
                if bool(show_los):
                    # ADD LINES OF SIGHT TRACES (= DISTANCE)
                    for idx in range(0,len(s), show_los):
        
                        LOS = s[idx,:,0,:3]
                        fig = Visualization.setup_3dplotly(fig, LOS, 'LOS','black', title, #marker_size = 5, 
                                                                  showlegend = False,mode = 'lines', legendgroup = runName_Fig, BrowserFig = True, width = 0.5)

                if min_distance:
                    # MINIMUM DISTANCE TRACE
                    dis = np.linalg.norm(s[:,1,0,:3]-s[:,0,0,:3], axis = -1) # (T,1)
                    dis_min_idx, dis_min = np.argmin(dis), np.min(dis)
                    s_disMin = s[dis_min_idx,:,0,:3] # out: (2,3)
                    
                    fig = Visualization.setup_3dplotly(fig, s_disMin, 'dis',cls.NB_SETTINGS['color']['dis'], title, 
                                                              legendgroup = runName_Fig, BrowserFig = True, showlegend = False,
                                                              #dxyz = s[:,aid_idx,1,:3], # <- for distance info (= velocity)
                                                              dis = dis_min,
                                                              width = 6.,
                                                              )
        ##
        fig.update_layout(#height=800, width=2400, 
                          title_text=title + f' - I = {len(runs_outcome["I"])}/{runs_N}',
                         )
        
        ## print outcome summary
        print(f'Filtered on condition: "{condition}", left = {count_check}/{runs_N}.\nPotential outcomes: {Analysis.OUTCOMES_LABELS}')
        for k,v in runs_outcome.items():
            print(f'{k} runs = {len(v)}/{runs_N} = {len(v)/runs_N:.2}:\n{runs_outcome[k]}')
        
        return fig, runs_outcome
    
    @classmethod
    def visualize_trajectoryWslider_single_run(cls, run_history: tuple,
                                      show_los: int = 0,
                                      title: str = 'Trajectories',
                                      thin = 0,
                                      runName: str = ''):
        warnings.warn('browserfig instance returned!')
        if show_los> 0: warnings.warn('show_los >0 not incorporated yet; ignored')
        
        title += ' - ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        
        s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history)
        
        check, outcome_label = Analysis.condition_outcome(statics['end'],
                                                          'all')
        if runName:
            runName += ' - ' + outcome_label
            
        do_bs = statics['env']['DO_BS']
        
        if thin:
            warnings.warn(f'ATTENTION: thinning time with {thin}')
            assert sb is False, 'not incorporated with sb'
            s,o,r,a,i = cls.thin_timeseries(s,o,r,a,i, thin = thin)
            
        fig = pgo.Figure()
        for t in range(1,len(s)):
            for aid_idx, aid in enumerate(o.keys()):
                fig = Visualization.setup_3dplotly(fig, s[:t,aid_idx, 0,:3], aid,cls.NB_SETTINGS['color'][aid], 
                                                          title, legendgroup = aid, BrowserFig = True, visible = False)
                if do_bs:
                    i_bs_aid = i[aid][:t,[-(2+4+1)]]
                    i_bs_aid[i_bs_aid == 0] = np.nan
                    s_bs_aid = s[:t,aid_idx, 0,:3]*i_bs_aid
                    name = aid+'BS'
                    fig = Visualization.setup_3dplotly(fig, s_bs_aid, name,cls.NB_SETTINGS['color'][name], title, marker_size = 5, showlegend = True,
                                                              mode = 'markers', legendgroup = aid, BrowserFig = True, visible = False)
        
            '''
            if bool(SHOW_LOS):
                LOS = s[(t-1),:,0,:3]
                fig = PEGanalysis.Visualization.setup_3dplotly(fig, LOS, 'LOS','black', 'Trajectories', #marker_size = 5, 
                                                          showlegend = False,mode = 'lines', legendgroup = runName_Fig, BrowserFig = True, width = 0.5)
            #'''
        # Make the first traces visible
        for idx in ([0,1] + [2,3]*do_bs + []*bool(show_los)):
            fig.data[idx].visible = True
        
        
        steps = []
        for idx in range(0, len(fig.data), 2 + 2*do_bs): # stepsize is number of traces per timestep
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],
                label=str(round((idx // (2 + 2*do_bs))/(len(fig.data)/(2 + 2*do_bs)),2)) # % of run
                #label = str(idx // (2 + 2*env.DO_BS)) # step idx, not timestep!
            )
            for idx_x in ([0,1] + [2,3]*do_bs + []*bool(show_los)):
                step["args"][0]["visible"][idx+idx_x] = True  # Toggle idx'th trace to "visible"
            steps.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "%: "},
            pad={"t": 50},
            steps=steps
        )]
        
        ''' 
        # add initial positions
        for aid_idx, aid in enumerate(setup_env.agent_id):
            fig = PEGanalysis.Visualization.setup_3dplotly(fig, s[[0],aid_idx, 0,:3], aid+'ini',NB_SETTINGS['color'][aid], 'Trajectories',
                                                      marker_size = 5, showlegend = False,mode = 'markers',
                                                      legendgroup = aid, BrowserFig = True, visible = True)
        ''' 
        
        fig.update_layout(sliders=sliders, 
                          title = f'{runName} - {title} - {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}')

        return fig
    
    @classmethod
    def visualize_progression_runs(cls, histories: list, condition: Union[list, str] = '', 
                                      show_los: int = 0,
                                      min_distance = True,
                                      title: str = 'Progression (scores)',
                                      runNames: list = None,
                                      thin = 0,
                                      agg = False):
        warnings.warn('browserfig instance returned!')
        if thin:
            warnings.warn(f'ATTENTION: thinning time with {thin}')
        runs_outcome = {out:[] for out in Analysis.OUTCOMES_LABELS} 
        
        title += ' - ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        
        ###
        ''' 
        FORCE_EXTRA_TRACE = True
        if FORCE_EXTRA_TRACE:
            ## TODO REMOVE
            t = x.copy()
            ## random value traces
            p = np.cos(8*t) + np.sin(15*t)
            p[230:] = p[230] + np.linspace(0,0.2, 260-230) 
            p /= np.max(p)
            e = np.sin(10*t) + np.cos(16*t)
            e -= np.linspace(0,0.3, len(e))
            e /= np.max(e)
            e = e[::-1]*-1
            g = e-p
            g /= np.max(g)
        
            force_y = {'p0':p, 
                       'e0':e, 
                       'game':g}
        '''
        ###
        runs_N = len(histories) 
        if agg:
            t_max = max([len(x[0]) for x in histories])
            x_extended  = np.linspace(0,1, t_max)
            game_progression = np.full((runs_N,t_max,1,3), 
                                        np.nan,
                                         np.float64)
            #'''
        dur_max = 0
        do_bs = histories[0][5]['env']['DO_BS']
        
        alpha = 0.6
        
        nb_rows = 2 # the number of rows in your plot
        nb_cols = 3 # the number of columns in your plot
        fig = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles = ['p0', 'e0','game']*2)
        xaxis_type = 0
        xaxis_types = ['%','dur'] # dur or time
        print(f'x_axis type chosen: {xaxis_types[xaxis_type]} from :{xaxis_types}')
        count_check = 0
        axes_ranges = [[-1,1],
                       #[-1e3,1e3],
                       [-10,10],
                       ]
        for run, run_history in enumerate(histories.copy()): 
            
            s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history)
            check, outcome_label = Analysis.condition_outcome(statics['end'],
                                                              condition)
            
            if not (statics['env']['DO_BS'] == do_bs):
                raise Exception('Inconsistent BS setting across episodes not supported ')    
            do_bs = statics['env']['DO_BS']
            
            if runNames:
                # predefined run name
                runName = runNames[run]
            else:
                # generic run name
                runName = f'R{run}'
            
            runs_outcome[outcome_label].append(runName) # track outcome
            
            if thin:
                assert sb is False, 'not incorporated with sb'
                s,o,r,a,i = cls.thin_timeseries(s,o,r,a,i, thin = thin)
                
            #runName = f'{runnames_override[run]}-{outcome_label}'
            if min_distance:
                # MINIMUM DISTANCE TRACE
                dis = np.linalg.norm(s[:,1,0,:3]-s[:,0,0,:3], axis = -1) # (T,1)
                dis_min_idx, dis_min = np.argmin(dis), np.min(dis)
                s_disMin = s[dis_min_idx,:,0,:3] # out: (2,3)

            ##
            if check:
                runName_Fig = runName + '-' + outcome_label
                
                count_check += 1
                
                value_preds, value_targets = [], []
                for c, aid in enumerate(['p0', 'e0','game']): # i.keys(), but hard code has expected order
                    showlegend = (c== 0)        
                    ## INFO
                    if xaxis_type == 0:
                        # %; rescaled to [0,1] always TODO ADJUST MEAN (THROUGH INTERPOLATION?)
                        x, y, y2 = i['game'][:,2]/np.max(i['game'][:,2]), i[aid][:,0], i[aid][:,1] # gamma, gamma_uV
                    elif xaxis_type == 1:
                        # duration wrt to end time
                        x, y, y2 = i['game'][:,2], i[aid][:,0], i[aid][:,1] # gamma, gamma_uV
                        dur_max = max(np.max(i['game'][:,0]), dur_max) # used later
                    
                    ## PLOTTING
                    # agent or game gamma(_pp)
                    fig.add_trace(pgo.Scatter(x=x, y=y, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='black', width=alpha),showlegend=showlegend), row=1, col=c+1) # Gamma
                    if aid == 'game':
                        # gamma_uv
                        fig.add_trace(pgo.Scatter(x=x, y=y2, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='grey', width=alpha),showlegend=False), row=1, col=c+1) # Gamma_Uv
                    if i[aid].shape[1] > 2 and aid != 'game': 
                        # BS traces
                        if do_bs:
                            yBS = i[aid][:,-(2+4+1)] # bs_bool = 2, lag = 1
                            fig.add_trace(pgo.Scatter(x=x, y=yBS, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='orange', width=alpha),showlegend=False), row=1, col=c+1)
                    
                                
                    ## Aggregates
                    if agg:
                        game_progression[run,:,0,c] = sc.interpolate.interp1d(x, y)(x_extended) # gamma
                            
                    ## reward
                    #r_cs = r[aid][:-1,0]
                    r_cs = np.cumsum(r[aid][:-1,0])
                    if xaxis_type == 0:
                        x, y = i['game'][:-1,2]/np.max(i['game'][:-1,2]), r_cs
                    elif xaxis_type == 1:
                        x, y = i['game'][:-1,2], r_cs
                    fig.add_trace(pgo.Scatter(x=x, y=y, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='grey', width=alpha),showlegend=False), row=2, col=c+1)
            
                    if sb:
                        # if sample batch is available, then provide value_pred & value_target
                        if aid != 'game':
                            # NOTE THAT GAME IS THE LAST AID!
                            value_pred, value_target = sb[aid].get('vf_preds'), sb[aid].get('value_targets')
                            value_preds.append(value_pred)
                            value_targets.append(value_target)
                        else:
                            # because p0 & e0 has been saved, we can define game as p0-e0
                            value_preds = np.array(value_preds)
                            value_targets = np.array(value_targets)
                            
                            value_pred = value_preds[1,:] + value_preds[0,:] 
                            # zero sum, but this is not a hard constraint, and the discrepany gives info
                            value_target = value_targets[1,:] + value_targets[0,:] #zero sum always
                        
                        fig.add_trace(pgo.Scatter(x=x, y=value_pred, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='red', width=alpha, dash = 'dash'),showlegend=False), row=2, col=c+1)
                        fig.add_trace(pgo.Scatter(x=x, y=value_target, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='red', width=alpha),showlegend=False), row=2, col=c+1)
                    ''' 
                    if FORCE_EXTRA_TRACE:
                        fig.add_trace(pgo.Scatter(x=x, y=force_y[aid], mode='lines', name=runName, legendgroup = runName, line=dict(color='red', width=alpha),showlegend=False), row=2, col=c+1)
                    ''' 
                    # min distance indicator
                    if min_distance and ((outcome_label != 'I') and (dis_min_idx != len(x))):
                        # if intercepted then minimum distance is at end
                        # or if not intercepted but final distance is min distance
                        
                        fig.add_trace(pgo.Scatter(x=[x[dis_min_idx]]*2, y=axes_ranges[0], mode='lines', 
                                                  name=runName_Fig, legendgroup = runName_Fig, 
                                                  line=dict(color=cls.NB_SETTINGS['color']['dis'], 
                                                            #width=alpha, 
                                                            dash = 'dash',
                                                            ),
                                                  showlegend=False), row=1, col=c+1)
                        fig.add_trace(pgo.Scatter(x=[x[dis_min_idx]]*2, y=axes_ranges[1], mode='lines', 
                                                  name=runName_Fig, legendgroup = runName_Fig, 
                                                  line=dict(color=cls.NB_SETTINGS['color']['dis'], 
                                                            #width=alpha, 
                                                            dash = 'dash',
                                                            ),
                                                  showlegend=False), row=2, col=c+1)

        ## AGGREGATE REULSTS
        if agg:
            quantiles = [0.25, 0.5, 0.75]
            game_agg = np.nanquantile(game_progression, quantiles, axis = 0) 
            
        for c, aid in enumerate(['p0', 'e0','game']):
            if agg:
                for r in range(1): 
                    showlegend = (c== 0) & (r == 0)
                    for q_i, q in enumerate(quantiles):

                        #if aid == 'game':
                        if q_i == 1:
                            line_dict = dict(color='blue')
                        else:
                            line_dict = dict(color='blue', width=alpha, dash = 'dash')
                            
                        fig.add_trace(pgo.Scatter(x=x_extended, y=game_agg[q_i,:,r,c], 
                                                 mode='lines', name=f'q{int(q*100)}', legendgroup = 'agg', line=line_dict,showlegend=showlegend), 
                                      row=r+1, col=c+1)
        
        
        ## LAYOUT
        fig.update_xaxes(matches='x')
        '''
        #y matching
        fig.update_yaxes(matches='y1', range = [-1,1], row = 1)
        fig.update_yaxes(matches='y4', row = 2)
        '''
        # TODO SEMINAR
        # TODO THIS DOES NOT WORK NICELY, I DONT WANT TO SET THE RANGE FOR THE SECOND ROW
        for row in range(nb_rows):
            match_y = "y" if row == 0 else f"y{row * nb_cols + 1}"
            for col in range(nb_cols):
                if row == 0:
                    fig.update_yaxes(matches=match_y, range = axes_ranges[0], row=row + 1, col=col + 1)
                else:
                    fig.update_yaxes(matches=match_y, range = axes_ranges[1], row=row + 1, col=col + 1)
        #''' 
        fig.update_layout(#height=800, width=2400, 
                          title_text=title + f' - I = {len(runs_outcome["I"])}/{runs_N}',
                         )
        
        fig.update_layout(xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.01)
                
        ## print outcome summary
        print(f'Filtered on condition: "{condition}", left = {count_check}/{runs_N}.\nPotential outcomes: {Analysis.OUTCOMES_LABELS}')
        for k,v in runs_outcome.items():
            print(f'{k} runs = {len(v)}/{runs_N} = {len(v)/runs_N:.2}:\n{runs_outcome[k]}')
            
        return fig, runs_outcome
    
    
    @classmethod
    def visualize_controls_runs(cls, histories: list, condition: Union[list, str] = '', 
                                      show_los: int = 0,
                                      min_distance= True,
                                      title: str = 'Control inputs',
                                      runNames: list = None,
                                      agg = False,
                                      thin = 0, 
                                      ):
                    
        if thin:
            warnings.warn(f'ATTENTION thinngin time series with {thin}')
            
        warnings.warn('browserfig instance returned!')
        runs_outcome = {out:[] for out in Analysis.OUTCOMES_LABELS} 
        
        title += ' - ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        
        ###
        ''' 
        FORCE_EXTRA_TRACE = True
        if FORCE_EXTRA_TRACE:
            ## TODO REMOVE
            t = x.copy()
            ## random value traces
            p = np.cos(8*t) + np.sin(15*t)
            p[230:] = p[230] + np.linspace(0,0.2, 260-230) 
            p /= np.max(p)
            e = np.sin(10*t) + np.cos(16*t)
            e -= np.linspace(0,0.3, len(e))
            e /= np.max(e)
            e = e[::-1]*-1
            g = e-p
            g /= np.max(g)
        
            force_y = {'p0':p, 
                       'e0':e, 
                       'game':g}
        '''

        ##
        dur_max = 0
        do_bs = histories[0][5]['env']['DO_BS']
        
        alpha = 0.6
        SHOW_OBS = True
        
        ##
        nb_rows = 2 # the number of rows in your plot
        nb_cols = 4 # the number of columns in your plot
        
        subplot_titles_OG = ['T','p','q','r']
        #subplot_titles_OG = ['A_x','A_y','A_z']
        subplot_titles = [f'{x} - {y}' for x in ['p0','e0'] for y in subplot_titles_OG]
        fig = make_subplots(rows=nb_rows, cols=nb_cols, 
                            subplot_titles = subplot_titles,
                            #column_titles = ['phi','theta','psi'],
                            #row_titles  = ['p0','e0'],
                            )
        
        ###
        runs_N = len(histories)
        if agg:
            t_max = max([len(x[0]) for x in histories])
            x_extended  = np.linspace(0,1, t_max)
            game_progression = np.full((runs_N,t_max,nb_rows,nb_cols), 
                                        np.nan,
                                         np.float64)
            
        xaxis_type = 0
        xaxis_types = ['%','dur'] # dur or time
        print(f'x_axis type chosen: {xaxis_types[xaxis_type]} from :{xaxis_types}')
        count_check = 0
        for run, run_history in enumerate(histories.copy()): 
            
            s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history)
            check, outcome_label = Analysis.condition_outcome(statics['end'],
                                                              condition)
            if thin:
                assert sb is False, 'not incorporated with sb'
                s,o,r,a,i = cls.thin_timeseries(s,o,r,a,i, thin = thin)
            
            if not (statics['env']['DO_BS'] == do_bs):
                raise Exception('Inconsistent BS setting across episodes not supported ')    
            do_bs = statics['env']['DO_BS']
            
            if runNames:
                # predefined run name
                runName = runNames[run]
            else:
                # generic run name
                runName = f'R{run}'
            
            runs_outcome[outcome_label].append(runName) # track outcome
            
            #runName = f'{runnames_override[run]}-{outcome_label}'
            ##
            if min_distance:
                # MINIMUM DISTANCE TRACE
                dis = np.linalg.norm(s[:,1,0,:3]-s[:,0,0,:3], axis = -1) # (T,1)
                dis_min_idx, dis_min = np.argmin(dis), np.min(dis)
                s_disMin = s[dis_min_idx,:,0,:3] # out: (2,3)
                
            ##
            a_gain, a_gain_agents = False, []
            n_cols = len(subplot_titles_OG)-1
            
            if check:
                runName_Fig = runName + '-' + outcome_label
                
                count_check += 1
                for r, aid in enumerate(o.keys()):
                    if xaxis_type == 0:
                        x = i['game'][:,2]/np.max(i['game'][:,2])
                    elif xaxis_type == 1:
                        x = i['game'][:,2]
                        dur_max = max(np.max(i['game'][:,0]), dur_max) # used later
                    
                    #for c, colIdx in zip(range(n_cols,(2-a[aid].shape[-1]),-1), range(-1,-(1+a[aid].shape[-1]),-1)): # acceleration control
                    for c, colIdx in zip(range(n_cols,(3-a[aid].shape[-1]),-1), range(-1,-(1+a[aid].shape[-1]),-1)): 
                        #print(c, colIdx, r)
                        '''
                        # phi, theta, psi control
                        y_control = a[aid][:,colIdx]
                        y_state = s[:,r,0,colIdx] if colIdx == -4 else np.zeros((len(x),)) # acc has no state
                        '''
                        '''
                        # a_x,y,z control 
                        y_control = a[aid][:,colIdx]
                        y_state = s[:,r,2,(colIdx-3)] #if colIdx != 3 else np.zeros((len(x),)) # acc has no state
                        '''
                        
                        #'''
                        # p,q,r, T control
                        y_control = a[aid][:,colIdx]
                        if colIdx >= -3: 
                            y_state = s[:,r,1,(colIdx)] # p,q,r states
                        elif colIdx == -4:
                            y_state = s[:,r,2,3] # T
                            a_T_range = [9.81*1.]*2
                            y_control = (1+np.minimum(y_control,0.))*a_T_range[0] + np.maximum(y_control,0.)*a_T_range[1] 
                        else:
                            print(colIdx)
                            raise
                        #'''
                        
                        
                        showlegend = (c== n_cols) & (r == 0)
                    
                        # agent or game gamma(_pp)
                        fig.add_trace(pgo.Scatter(x=x, y=y_control, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='black', width=alpha),showlegend=showlegend), row=r+1, col=c+1) # 
                        fig.add_trace(pgo.Scatter(x=x, y=y_state, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='grey', width=alpha),showlegend=False), row=r+1, col=c+1) # Gamma_Uv
                        
                        if i[aid].shape[1] > 2 and aid != 'game': 
                            # BS traces
                            if do_bs:
                                yBS = i[aid][:,-(2+4+1)] # bs_bool = 2, lag = 1
                                fig.add_trace(pgo.Scatter(x=x, y=yBS, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='orange', width=alpha),showlegend=False), row=r+1, col=c+1)
                        
                        if aid+'_gain' in a:
                            # SHOW GAIN INPUTS
                            a_gain = True
                            if showlegend:
                               a_gain_agents.append(aid)
                               
                            y_gainDLos = a[aid+'_gain'][:,1]
                            y_gainLos = a[aid+'_gain'][:,0]
                            
                            fig.add_trace(pgo.Scatter(x=x, y=y_gainLos, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='darkgreen', width=alpha),showlegend=False), row=r+1, col=c+1) 
                            fig.add_trace(pgo.Scatter(x=x, y=y_gainDLos, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='darkblue', width=alpha),showlegend=False), row=r+1, col=c+1) 
                 
                        if SHOW_OBS & (aid+'_gain' in a):
                            # SHOW OBSERVATIONS (OF RATES)
                            #gain = 1. if 'p0' in aid else -1.
                            y_dlos = o[aid][:,(c*2-1)]#*y_gainDLos
                            y_los = o[aid][:,(c*2-2)]#*y_gainLos
                            
                            fig.add_trace(pgo.Scatter(x=x, y=y_los, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='green', width=alpha, dash = 'dash'),showlegend=False), row=r+1, col=c+1) # dLos
                            fig.add_trace(pgo.Scatter(x=x, y=y_dlos, mode='lines', name=runName_Fig, legendgroup = runName_Fig, line=dict(color='blue', width=alpha, dash = 'dash'),showlegend=False), row=r+1, col=c+1) # los
                            
                        ## Aggregates
                        if agg:
                            game_progression[run,:,r,c] = sc.interpolate.interp1d(x, y_control)(x_extended) # gamma
                            
                        # min distance indicator
                        if min_distance and ((outcome_label != 'I') and (dis_min_idx != len(x))):
                            # if intercepted then minimum distance is at end
                            # or if not intercepted but final distance is min distance
                            #raise
                            
                            fig.add_trace(pgo.Scatter(x=[x[dis_min_idx]]*2, y =[-1,1],#y=[-np.inf,np.inf], 
                                                      mode='lines', 
                                                      name=runName_Fig, legendgroup = runName_Fig, 
                                                      line=dict(color=cls.NB_SETTINGS['color']['dis'], 
                                                                #width=alpha, 
                                                                dash = 'dash',
                                                                ),
                                                      showlegend=False), row=r+1, col=c+1)
                
        if a_gain:
            title +=  f'   | Gain action: {a_gain_agents}' + ' | (green = K(x)_LoS, blue = K(x)_dLoS)'
       
        ## AGGREGATE REULSTS
        if agg:
            quantiles = [0.25, 0.5, 0.75]
            game_agg = np.nanquantile(game_progression, quantiles, axis = 0) 
            
        aid_aMax = np.max(a[aid], axis = 0)
                            
        for c, aid in enumerate(subplot_titles_OG):#['p0', 'e0','game']):
            if agg:
                for r in range(2):
                    showlegend = (c== 0) & (r == 0)
                    for q_i, q in enumerate(quantiles):

                        #if aid == 'game':
                        if q_i == 1:
                            line_dict = dict(color='blue')
                        else:
                            line_dict = dict(color='blue', width=alpha, dash = 'dash')
                            
                        fig.add_trace(pgo.Scatter(x=x_extended, y=game_agg[q_i,:,r,c], 
                                                 mode='lines', name=f'q{int(q*100)}', legendgroup = 'agg', line=line_dict,showlegend=showlegend), 
                                      row=r+1, col=c+1)
                
            fig.update_xaxes(title_text=xaxis_types[xaxis_type], row=1, col=c+1)
            fig.update_yaxes(title_text='Input [-]', row=1, col=c+1)
            fig.update_xaxes(title_text=xaxis_types[xaxis_type], row=2, col=c+1)
            fig.update_yaxes(title_text='Input [-]', row=2, col=c+1)
        
        
        ## LAYOUT
        fig.update_xaxes(matches='x')
        '''
        #y matching
        fig.update_yaxes(matches='y1', range = [-1,1], row = 1)
        fig.update_yaxes(matches='y4', row = 2)
        '''
        # TODO SEMINAR
        # TODO THIS DOES NOT WORK NICELY, I DONT WANT TO SET THE RANGE FOR THE SECOND ROW
        for row in range(nb_rows):
            match_y = "y" if row == 0 else f"y{row * nb_cols + 1}"
            for col in range(nb_cols):
                if row == 0:
                    fig.update_yaxes(matches=match_y, #range = [-1,1], 
                                     row=row + 1, col=col + 1)
                else:
                    fig.update_yaxes(matches=match_y, #range = [-5,5], 
                                     row=row + 1, col=col + 1)
        #''' 
        fig.update_layout(#height=800, width=2400, 
                          title_text=title + f' - I = {len(runs_outcome["I"])}/{runs_N}',
                         )
        
        fig.update_layout(xaxis3_rangeslider_visible=True, xaxis3_rangeslider_thickness=0.01)
                
        ## print outcome summary
        print(f'Filtered on condition: "{condition}", left = {count_check}/{runs_N}.\nPotential outcomes: {Analysis.OUTCOMES_LABELS}')
        for k,v in runs_outcome.items():
            print(f'{k} runs = {len(v)}/{runs_N} = {len(v)/runs_N:.2}:\n{runs_outcome[k]}')
            
        return fig, runs_outcome
    
    
    #%% Analysis.Tables

    ## class variables
    # maps
    tables_order = ['start','end','agent', 's','o','r','a','i']
    tables_NameIdxMap = {k: v for k,v in \
             zip(tables_order, range(8))} # name:idx mapping
    tables_IdxNameMap = {v:k for k,v in tables_NameIdxMap.items()} # inverse map

    # df column names

    S_NAMES = ['|x|','|dx|','|ddx|'] \
        +[v+k for v in ['','d','dd'] for k in ['x', 'y', 'z', 'phi', 'theta','psi']]

    SORAI_NAMES = {'s':S_NAMES,
                   'o':[],
                   'r':[],
                   'a':[],
                   'i':[],
                   }
    
    
    ##
    @classmethod
    def tables_get_table(cls, tables: list, name: str, aid: str):
        
        try:
            table = tables[cls.tables_NameIdxMap[name]][aid]
        except KeyError:
            warnings.warn('aid argument ignored as it does not exist (depth does not exist)')
            # in to deep! no agent specifcs exists
            table = tables[cls.tables_NameIdxMap[name]]

        return table
    
    @classmethod
    def tables_extract_runs(cls, histories: list, 
                            condition: Union[list, str] = '',
                            quantiles: np.array = np.array([0., 0.5, 1.]), 
                            sorai_names: dict = {},
                            runNames: list = None,
                            ): 
        '''
        TODO
        
        (A) SPLIT THIS FUNCTION UP INTO PARTS RELATED TO THE DIFFERENT
        PARTS OF 
            s, o, r, a , i , statics = HIS
        SO THAT WE CAN UNIT TEST THEM!!!
        
        (B) INFER OBS, ACTION & REWARD & INFO COLUMN NAMES for the 
            ultimate df's setup
        
        (C) give this function more arguments
        ''' 
        warnings.warn('Ensure that keys with agents also reflect so within sorai_names!')
        print(f'Quantiles selected {quantiles}')
        print(f'DF output order: {cls.tables_order}')
        ##
        sorai_names = {**cls.SORAI_NAMES, **sorai_names} # override old with provided names

        
        ## Part 1: initialize containers and column names
        runs_outcome = {out:[] for out in Analysis.OUTCOMES_LABELS} 
        
        s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(histories[0])
        
        #
        start_df, outcome_df = {}, {}
        agent_df = {aid:[] for aid in o.keys()}
        
        # agent ids only
        o_df = {aid:[] for aid in o.keys()}
        a_df = {aid:[] for aid in a.keys()}
        
        # agent ids & game (in that order!)
        r_df = {aid:[] for aid in r.keys()}
        i_df = {aid:[] for aid in i.keys()}
        s_df = copy.deepcopy(i_df) # reuse info for state, game agent to be appended
        
        ## Part 2: fill primitive containers with history content
        runs_N = len(histories)
        count_check = 0
        for run, run_history in enumerate(histories):
            
            s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history) 
            outcome_label = [k for k, v in statics['end'].items() if v][0] 
            
            check, outcome_label = Analysis.condition_outcome(statics['end'],
                                                              condition)
            if runNames:
                # predefined run name
                runName = runNames[run]
            else:
                # generic run name
                runName = f'R{run}'
            
            runs_outcome[outcome_label].append(runName) # track outcome
        
            ## visualize if condition is met
            if check:
                count_check += 1
                
                ##
                start_df[runName] = statics['start'].copy()
                outcome_df[runName] = statics['end'].copy()
                ##
                s_game = s[:,[1],:,:] - s[:,[0],:,:] # i.e. s_diff = s_e-s_p
                s_full = np.concatenate((s, s_game), axis = 1) # out (T,aids=2+1,D=3,s=6); game is another agent at index = -1
                s_full_norm = np.linalg.norm(s_full[...,:3], axis = -1, keepdims = False) # out (T,3,3,); x_, dx_, ddx_norm; 
                # NOTE that this dx & ddx is not necessarily v & a; we dont estimat them using (e.g. forward, backward) differences
                #s_full = np.concatenate((s_full, np.linalg.norm(s_full[...,:3], axis = -1, keepdims = True)), axis = -1) # out (T,aids=3,D=3,s=6+1); norm only for xyz
            
                s_full = np.concatenate((s_full_norm, s_full.reshape((-1, 3,3*6))), axis = -1) # out (T,3,3+(6*3))
                
                ##
                for c, aid in enumerate(i.keys()):
                    ##
                    s_df[aid].append(np.quantile(s_full[:,c,:], quantiles, axis = 0).T.flatten())
                    r_df[aid].append(np.append(np.quantile(r[aid][:-1], quantiles, axis = 0).T.flatten(),[np.sum(r[aid][:-1]), np.sum(r[aid])]))
                    i_df[aid].append(np.quantile(i[aid], quantiles, axis = 0).T.flatten())
            
                    if aid != 'game':
                        ## TODO infer this elsewhere (maybe from obs & action spaces?)
                        DOF = a[aid].shape[-1]
                        o_dim = o[aid].shape[-1]
            
                        agent_df[aid].append(statics['agent'][aid].copy())
                        
                        o_df[aid].append(np.quantile(o[aid], quantiles, axis = 0).T.flatten())
                        a_df[aid].append(np.quantile(a[aid], quantiles, axis = 0).T.flatten())
        
        ## TODO MOVE OR INFER ELSEWHERE
        warnings.warn('NOTE that o.r.a. columns are unstably inferred')
        sorai_names['o'] = [f'OBS{x}' for x in range(1,o_dim+1)]
        sorai_names['r'] = ['R']
        sorai_names['a'] = [f'DOF{x}' for x in range(1,DOF+1)]

        quantiles_names = [f'q{int(q*100)}' for q in quantiles] # generic
        quantiles_names_r = [f'{q}PreT' for q in quantiles_names]+['sumPreT','sum'] # reward specific
        
        ## Part 3: primitive containers to dataframe
        start_df = pd.DataFrame(data = start_df).T
        outcome_df = pd.DataFrame(data = outcome_df).T
        outcome_df.replace({False: 0., True: 1.}, inplace=True)
        for c, aid in enumerate(i.keys()):
            ## containers with game agent
            s_df[aid] = pd.DataFrame(s_df[aid], columns = \
                 [f'{y}-{x}' for y in sorai_names['s'] for x in quantiles_names])
            s_df[aid].index = start_df.index
            # all indexes are overriden with that of start_df which has runNames
            
            r_df[aid] = pd.DataFrame(r_df[aid], columns = \
                 [f'{y}-{x}' for y in sorai_names['r'] for x in quantiles_names_r]) # todo do something similar to info_cols ALSO remove 'Rew' 
            r_df[aid].index = start_df.index
            
            i_df[aid] = pd.DataFrame(i_df[aid], columns = \
                 [f'{y}-{x}' for y in sorai_names['i'][('game' if (aid == 'game') else 'agent')].keys()  for x in quantiles_names])
            # Note; the nested conditional statement changes aid to agent
            i_df[aid].index = start_df.index
        
            ## containers without game agent
            if aid != 'game':
                agent_df[aid] = pd.DataFrame(agent_df[aid]) # todo give obs names
                agent_df[aid].index = start_df.index
                
                o_df[aid] = pd.DataFrame(o_df[aid], columns = \
                         [f'{y}-{x}' for y in sorai_names['o'] for x in quantiles_names]) # todo give obs names
                o_df[aid].index = start_df.index
                
                a_df[aid] = pd.DataFrame(a_df[aid], columns = \
                         [f'{y}-{x}' for y in sorai_names['a'] for x in quantiles_names])
                a_df[aid].index = start_df.index
        
        
        ## finalize
        print(f'Filtered on condition: "{condition}", left = {count_check}/{runs_N}.\nPotential outcomes: {Analysis.OUTCOMES_LABELS}')
        for k,v in runs_outcome.items():
            print(f'{k} runs = {len(v)}/{runs_N} = {len(v)/runs_N:.2}:\n{runs_outcome[k]}')
        # collect
        df_all = [start_df, outcome_df, agent_df, s_df, o_df, r_df, a_df, i_df]
        return df_all, runs_outcome
    
    @classmethod
    def tables_describe_across_runs(cls, tables: list):
        warnings.warn('Note that non-numeric types are lost in aggregation! (e.g. TF dicts)')
        describe_df = []
        for data in tables:
            if isinstance(data, pd.DataFrame):
                summary_d = data.describe().round(3)
            elif isinstance(data, dict):
                summary_d = {}
                for k,v in data.items():
                    summary_d[k] = v.describe().round(3)
            else:
                raise Exception('unknown dtype')    
            describe_df.append(summary_d)
        return describe_df


    @classmethod
    def tables_plot_tables(cls, table: pd.DataFrame):
        '''
        IDEA is to choose two variables from a table and then polot them 
        against aeachother e.g. noise level vs interception rate
        '''
        
        return 
    
    
    @classmethod
    def tables_setup_latex_report(cls, tables: list):
        ''' 
        create a massive string with
        - first describe dctionaries
        - then the full dictionaries; 
        
        take df.to_latex and add the resize text before and after
        as well as label generator & simple caption
        
        ''' 
        latex_report =  'TEST' # should be a massive string
        return latex_report

    #%% QUICK METHODS
    
    @classmethod
    def quick_runs_summary(cls):
        return 
    
    @classmethod
    def quick_runs_visualization(cls):
        return 
    
    @classmethod
    def quick_single_run_summary(cls):
        return 


    @classmethod
    def quick_single_run_visalization(cls):
        return 
    
    
        

#%% VISUALIZATION TOOLS (functions called by conduct)

class Visualization:
    '''
    ANALYSIS CLASS WITH STATIC FUNCTIONALITY
    '''
    @staticmethod 
    def setup_3dplotly(fig, xyz, name, color, title, mode = 'lines',marker_size = 5, torch= False, BrowserFig= False, legendgroup= 'R0', 
                       showlegend = True, width = 3, visible = True, dxyz = None, dis = None):
        '''
        TODO; to analysis class 'visualize_simple3d'
        '''
    
        if torch:
            xyz = xyz.numpy()
        assert isinstance(xyz, np.ndarray), type(xyz)
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        
        if isinstance(dxyz, np.ndarray):
            # DOCS; https://plotly.com/python-api-reference/generated/plotly.graph_objects.Cone.html
            u, v, w = dxyz[:,0], dxyz[:,1], dxyz[:,2]
            v_norm = list(np.linalg.norm(dxyz, axis = 1).round(3))
            
            cmax, cmin = 10., 0.
            fig.add_trace(pgo.Cone(x=x,y=y,z=z, u = u, v = v, w = w,
                          #mode = mode, 
                          name = name, 
                          #marker = dict(color = color, size = marker_size), 
                          #line = dict(color= color, width  = width),
                          legendgroup=legendgroup,
                          legendgrouptitle={'text': legendgroup},
                          showlegend = showlegend,
                          visible = visible,
                          hovertemplate = f'({legendgroup})'+'<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>v: %{text}', # TODO find the v variable name
                          ## cone specific
                          colorbar = dict(title = dict(text = f'|v|(>{cmax})')),
                          colorscale = 'rdpu',#'viridis',
                          reversescale = True, # reverses colormap
                          cmax = cmax, cmin = cmin, 
                          sizeref = 0.7, # def 0.5
                          #anchor = "tip" | "tail" | "cm" | "center" 
                          text = v_norm, # text is shown in hovertemplate
                ))
        elif dis is not None:
            fig.add_trace(pgo.Scatter3d(x=x,y=y,z=z,
                          mode = mode, 
                          name = name, 
                          marker = dict(color = color, size = marker_size), 
                          line = dict(color= color, width  = width),
                          legendgroup=legendgroup,
                          legendgrouptitle={'text': legendgroup},
                          showlegend = showlegend,
                          visible = visible,
                          
                          hovertemplate = f'({legendgroup})'+'<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>d: %{text}', # TODO find the v variable name
                          #anchor = "tip" | "tail" | "cm" | "center" 
                          text = [round(dis,3)]*len(x), # text is shown in hovertemplate
                ))
        else:    
            t = (np.arange(len(x))*0.01).round(3)
            fig.add_trace(pgo.Scatter3d(x=x,y=y,z=z, mode = mode, name = name, marker = dict(color = color, size = marker_size), line = dict(color= color, width  = width),
                          legendgroup=legendgroup,
                          legendgrouptitle={'text': legendgroup},
                          showlegend = showlegend,
                          visible = visible,
                          hovertemplate = f'({legendgroup})'+'<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>t: %{text}',
                          
                          text = t, # text is shown in hovertemplate
                ))
        # hovertemplates & hovermode at https://plotly.com/python/hover-text-and-formatting/
        
        if BrowserFig:
            # for non-jupyter notebook IDE use
            fig.update_layout(legend = dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
                              showlegend = True, title = dict(text=title, y = 0.9, x = 0.5)
                              #hovermode="x unified",
                              )
        else:
            # notebook figure
            fig.update_layout(autosize = True, width = 600, height = 500, margin=dict(l=0, r=0, b=0, t=0), 
                              legend = dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
                              showlegend = True, title = dict(text=title, y = 0.9, x = 0.5),
                              #hovermode="x unified",
                              )
            
        #fig.update_traces(connectgaps=False)
        return fig # returned to allow additional traces 
    
    @staticmethod
    def setup_slider_plot():
        # TODO REMOVE
        return 
    
    @staticmethod
    def setup_animation_plot():
        # TODO REMOVE
        '''
        df has x,y,z & time
        
        # Create frames for each unique time
        frames = [go.Frame(
            data=[go.Scatter3d(
                x=df1[df1['time'] <= t]['x'],
                y=df1[df1['time'] <= t]['y'],
                z=df1[df1['time'] <= t]['z'],
                mode='lines',
                marker=dict(size=4, color='red')
            ),
            go.Scatter3d(
                x=df2[df2['time'] <= t]['x'],
                y=df2[df2['time'] <= t]['y'],
                z=df2[df2['time'] <= t]['z'],
                mode='lines',
                marker=dict(size=4, color='blue')
            )]) for t in sorted(df1['time'].unique())]
        
        # Add frames to the figure
        fig.frames = frames
        
        # Create a slider
        sliders = [dict(steps=[dict(method='animate',
                                    args=[[f.name],
                                          dict(mode='immediate',
                                               frame=dict(duration=300, redraw=True),
                                               transition=dict(duration=0))],
                                    label=f.name) for f in fig.frames],
                         active=0)]
        
        # Update the layout
        fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                            buttons=[dict(label='Play',
                                                          method='animate',
                                                          args=[None, 
                                                                dict(frame=dict(duration=300, redraw=True), 
                                                                     fromcurrent=True, 
                                                                     transition=dict(duration=0))])])],
                          sliders=sliders)
        '''
        pass
        return 
        
    
    
def analysis_trajectory_3d(traj_Z, traj_Rpred, name, interception_threshold = 1, torch = False):
    if torch:
        traj_Z = traj_Z.numpy()
        traj_Rpred = traj_Rpred.numpy()
    ## compute distance target and agressor prediction
    assert traj_Z.shape == traj_Rpred.shape, (traj_Z.shape, traj_Rpred.shape)
    distance_zrpred = np.abs(traj_Z-traj_Rpred)
    interception_bool = np.all((distance_zrpred < interception_threshold), axis = 1) #(T,)
    
    ## 3d trajectory visualization - cumulative performance 
    fig = pgo.Figure()
    fig = Analysis.setup_3dplotly(fig, traj_Z, 'z_t','purple', f'Trajectory - {name} error | Intercepted: {False}')
    fig = Analysis.setup_3dplotly(fig, traj_Rpred, '\hat{r}_t','red', f'Trajectory - {name} error | Intercepted: {False}')
    # check for interception
    if np.sum(interception_bool > 0):
        # plot
        interception_t = np.min(np.where(interception_bool)) # select first interception case
    
        fig = Analysis.setup_3dplotly(fig, traj_Z[[interception_t], :], 'interception-z_t','purple', f'Trajectory - {name} error | Intercepted: {True}',mode = 'markers')
        fig = Analysis.setup_3dplotly(fig, traj_Rpred[[interception_t],:], 'interception-\hat{r}_t','red', f'Trajectory - {name} error | Intercepted: {True}',mode = 'markers')
    
    fig.show()
    return


#%%











#%% MATPLOTLIB FUNCTIONS - DEPRECATED


class PLTfuncs:
    
    @staticmethod
    def gammaNreward(eval_histories):
            
        ##
        fig, ax = plt.subplots(2,3, figsize = (24,8))
        
        xaxis_type = 1
        xaxis_types = ['%','dur'] # dur or time
        
        ##
        runs_N = len(eval_histories) 
        t_max = max([len(x[0]) for x in eval_histories])
        game_progression = np.full((runs_N,2, t_max,3), 
                                    np.nan,
                                     np.float64)
        dur_max = 0
        for run, run_history in enumerate(eval_histories): 
            s,o,r,a,i, statics, sb = Analysis.run_safe_unpack(run_history)
            runName = f'R{run}'
            for c, aid in enumerate(i.keys()):
                r_cs = np.cumsum(r[aid][:-1,0])
                game_progression[run,0, :len(i[aid]), c] = i[aid][:,0]
                game_progression[run,1, :(len(i[aid])-1), c] = r_cs # -1: = pre-terminal reward
                if xaxis_type == 0:
                    # %; rescaled to [0,1] always
                    x, y = i['game'][:,1]/np.max(i['game'][:,1]), i[aid][:,0] # gamma
                elif xaxis_type == 1:
                    # duration wrt to end time
                    x, y = i['game'][:,1], i[aid][:,0] # gamma
                    dur_max = max(np.max(i['game'][:,0]), dur_max) # used later
                ax[0,c].plot(x,y, alpha = 0.2, label =runName, color = 'grey')
        
                ##
                if xaxis_type == 0:
                    x, y = i['game'][:-1,1]/np.max(i['game'][:-1,1]), r_cs
                elif xaxis_type == 1:
                    x, y = i['game'][:-1,1], r_cs
                ax[1,c].plot(x,y, alpha = 0.2, label =runName, color = 'grey')
        
        ##
        game_mean = np.nanmean(game_progression, axis = 0) 
        #t = np.arange(t_max)*0.1 # t_delta
        for c, aid in enumerate(i.keys()):
            if xaxis_type == 0:
                # %
                x = np.linspace(0,1,t_max) # TODO WE ACTUALLY REQUIRE INTERPOLATION
            elif xaxis_type == 1:
                # duration
                x = np.linspace(0,dur_max,t_max)
            ax[0,c].plot(x, game_mean[0,:,c], color = 'blue', label = 'mean')
            ax[1,c].plot(x, game_mean[1,:,c], color = 'blue', label = 'mean')
            ##
            ax[0,c].set_title(f'Gamma - {aid}')
            ax[0,c].set_xlabel(xaxis_types[xaxis_type])
            ax[0,c].set_ylabel('Gamma')
        
            ax[1,c].set_title(f'Reward - {aid}')
            ax[1,c].set_xlabel(xaxis_types[xaxis_type])
            ax[1,c].set_ylabel('Reward')
        fig.tight_layout()
        return fig







