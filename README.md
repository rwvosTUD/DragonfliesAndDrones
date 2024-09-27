# *Hunt like a Dragonfly and Strike like a Drone*
## DragonfliesAndDrones
Repository for MSc. thesis research in Aerospace Engineering at TU Delft titled *'**Hunt like a Dragonfly and Strike like a Drone**: learning pursuit controllers for insect interception through multi-agent deep reinforcement learning for onboard use in autonomous quadcopters'* conducted by R.W. Vos, Dr. M. Yedutenko, Prof. Dr. G.C.H.E. de Croon at Delft University of Technology, Faculty of Aerospace Engineering, Department of Control & Simulation, Micro-Air-Vehicle Laboratory (MAVLab) between November 2023- October 2024.

## Abstract
*Insect pest elimination through MAV interception can reduce the need for insecticide and can contribute to sustainable agriculture. In this research, we analyze the feasibility of such solutions through simulated two-player differential games of pursuit and evasion with agents operating on minimalistic sets of biologically-plausible observations and optimized to control constrained vehicle models through deep multi-agent reinforcement learning. Our pursuer and evader agent, representing the quadcopter drone and insect pest respectively, are asymmetric in design, capabilities and objectives. Our results show that our quadcopter pursuer is consistently able to pursue and intercept a reactive insect-inspired evader as well as recordings of actual insect targets, achieving interception rates of 55\% and 95\% on these respective tasks. In comparison, pursuers alternatively optimized against non-reactive evaders or reactive drone-like evaders with symmetric capabilities, achieve an interception rate of only 20\% for the same insect target recordings. Despite these promising results, we conclude that further research is needed to formally establish the superiority of multi-agent optimization in this asymmetric game scenario. Finally, we determine how emergent behavior and strategies resembles nature. During the confrontations, we observe that our pursuer mainly implements pure-pursuit as well as motion camouflage to some degree; drawing comparison to the hunting strategy of dragonfly.*

## Repository overview
This repository contains two main folders:
- *Code*: containing all relevant scripts to setup, optimize and evaluate agents in our pursuit-evasion scenarios using deep reinforcement learning in Pytorch and Ray RLlib. This folder also contains a yaml & requirements.txt file to replicate the conda environment. Note that the Pytorch version was chosen to correspond with CUDA 12.1. 
- *Visualizations*: containing visualized confrontations between our optimized agents and insect recordings across a selection of system configurations. Files are made with plotly and are of HTML format, downloading is required for proper viewing.

### Visualizations
This repository contains interactive visualizations of 3D trajectories collected from confrontations between our optimized pursuer and evader agents as well as between the optimized pursuer agents and the offline Opogona recordings (provided by SOURCE). Trajectories are created using Plotly, saved in HTML files and stored in the visualizations folder of this repository. 

Within these files, the pursuer and evader trajectories are blue and red respectively, with a similarly colored sphere indicating their initial position. Orange and green spheres on top of a trajectory highlight that the agent in question is in the motion-camouflage game state. The grey connecting lines define the lines-of-sight between the agents at a periodic interval. The green neon line indicates the shortest distance between agents observed within a trial.  

Example trajectories:
<br>![SA_offlineTrajectories_EXAMPLE](https://github.com/user-attachments/assets/0d1ace43-3fa9-4e95-919c-07dbebba690a)![SA_onlineTrajectories_EXAMPLE](https://github.com/user-attachments/assets/6146fc91-d8b7-49aa-988c-40d5adc286d5)



