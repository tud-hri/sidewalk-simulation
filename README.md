# Sidewalk Simulation
This project contains the code accompanying the paper **"A Model of the Sidewalk Salsa"** by Olger Siebinga. This software was created to simulate the interactive behavior of two human pedestrians approaching each other on a sidewalk. It includes code to execute the experiments, playback the gathered data, and plot and evaluate the data. The data that was gathered for the publication can be found [here](https://doi.org/10.4121/504bba81-c1fd-422a-ac21-e0dc9b8feed9.v1). The version numbers of the datasets, correspond to the versions of the software. Version 1 uses bicycle dynamics while version 2 uses pedestrian dynamics. Version 2 corresponds to the data reported in the paper. This repository was tested with Python version 3.10


## Quick Start
The main project folder contains three run-scripts: `simulate_online.py`, `simulate_offline.py`, and `playback.py`. The `simulate_offline.py` script will simulate all scenarios described in the paper. All simulated data will be stored in the data folder as pickle files, mat files (for Matlab compatability), and csv files. 

The `playback.py` script can be used to replay a single simulated trial. It also includes a live plot of the plan and belief during the interaction (as can be seen in Figure 1B in the paper). Select which trial to load by changing the file name in the script’s main block. The plotting module can be used to gain more insight into the recorded data. It contains the scripts to reproduce the figures from the paper and more. To reproduce the plots in the paper obtain all the data from the link above, extract in the data folder, and run the scripta in the plotting module.

Finally, the script `simulate_online.py` will run a single online (i.e., live displayed) simulation. The data of this simulation will be stored as `online_simulation` in the data folder, and can be plotted or replayed in the same way as offline simulations.

## The modules
The structure of this repository is based on the [`simple-merging-experiment`](https://github.com/tud-hri/simple-merging-experiment) repository. All modules will be separately discussed here

### Agents
The `agents` module contains classes that can be used to provide input to the pedestrian. The name `agents` is derived from AI terminology, where every decision-maker is considered an agent. These agents can either provide continuous or discrete input to the controllable objects. In the simulation, continuous input is used. 

### Controllable Objects
The `controllableobjects` module contains the dynamical models of controllable objects. In this simulation, only a pedestrian model is used. This pedestrian model can be controlled by continuous inputs (accelerations).  

### GUI
All files related to the graphical user interfaces are located in the GUI module. All GUI files are made with pyqt. There are a main GUI in the project; the `simulation_gui` is the user interface that is used when recorded data is played back. The GUI module also contains the worldview widget that is used to display the world. This should be self-explanatory. 

### Plotting
The plotting module contains scripts that can provide insight into the recorded data. The file `plot_overview.py` loads all experiment data and was used to create the (box) plots that give an overview of the simulated behavior in all conditions. It can plot both metrics and raw signals (trajectories). It uses Pandas dataframes (loaded with the `load_data.py`script) containing all metrics and signals so it easy to modify to plot other signals.  

The script `plot_single_trial.py` can be used to gain more insight into the behavior of the pedestrians in a single simulated trial.

### Simulation
The simulation module contains two classes that are used when running the simulation. The simmasters take care of the clock. These classes contain the loop in which all update functions are called. The normal SimMaster class runs the simulations while the PlaybackMaster is used when replaying a recorded trial. The simmasters are also used for saving the experiment data. To initialize all data storage, the simmasters have a maximum run time. If this maximum time is exceeded, the simulation will automatically stop. 

The `SimulationConstants` class can be used to create simple data objects that hold the parameters of the simulation. These are the dimensions of the sidewalk, the collision constraint, and the timing parameters.

### Track Objects
The trackobjects module contains the definition of the experiment track (a sidewalk in this case). In the simulation, the sidewalk track is used. Other track can be made by inheriting from the Track class.
