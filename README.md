# ProtStretch

ProtStretch is a tool for analyzing the data from protein stretching simulations and experiments. It consists of 6 main files:

1. Tools.py - storing functions used for numerical analysis of the data; 
2. Trajectory.py - defining an object that is a single stretching trajectory. It takes the path to our data and parameters of the protein as an input. Methods of this class compute states' boundaries of a protein, rupture forces, works, contour lengths, persistence length and spring constant. One can also plot the data and fitted curves, 
plot contour length histogram with fitted distributions and save all fitted parameters and images.
3. Theory.py - a class inheriting from Trajectory.py. It defines an object that is a single stretching trajectory in simulations. 
4. Experiment.py - a class inheriting from Trajectory.py. It defines an object that is a single stretching trajectory in experiment. 
5. Whole.py - a class storing multiple objects from class Trajectory and creating work histograms and rupture forces histograms. 
6. WholeExperiment.py - a class inheriting from Whole.py. It defines an object that is a set of different trajectories in experiment. 

