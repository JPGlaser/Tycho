# Python Classes/Functions used to Analyze Tycho's Output

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import math
import numpy as np
import time as tp
import matplotlib as plt
import random as rp
import sys
import os
import io
import unittest
import glob

# Import the Amuse Base Packages
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.datamodel import particle_attributes
from amuse.io import *
from amuse.lab import *

# Import the Amuse Gravity & Close-Encounter Packages
from amuse.community.ph4.interface import ph4 as grav
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.couple import multiples

# Import Tycho modules
from tycho import create, read, util, write

# Safety net for matplotlib
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import *
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
    
    matplotlib.rcParams['figure.figsize'] = (16, 6)
    matplotlib.rcParams['font.size'] = (14)
except ImportError:
    HAS_MATPLOTLIB = False

# Import cPickle/Pickle
try:
   import cPickle as pickle
except:
   import pickle

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #


# Most of the default values depend on naming and will chang if we change how we often we restart. 
# We saved every 10 timesteps after the first timestep so the restart files start at 0.55 and increment 
# by 10*dt or 0.50 in our case. If any of these values are different they can be changed easily, either 
# the defaults or just input the correct values. dt is the increment of the restart files not the dt 
# used in the cluster simulation. delta_t is the dt from the cluster simulation. restart_base should be 
# the restart files name minus the number. e.g. $PATH/EnergyTest_restart

def GetValues(restart_dir, cluster_name, **kwargs):
    ''' Gets Corrected/Uncorrected Total, Kinetic, & Potential Energy as well as Angular & Linear Momentum
        num_workers: Number of PH4 Workers
        use_gpu: Set to 1 (Default) to Utalize GPUs
        gpu_ID: Select the GPU to Use
        eps2: The Smoothing Parameter (Deault is 0)
    '''
# Read & Set Keyword Arguments
    num_workers = kwargs.get("num_workers", 1)
    use_gpu = kwargs.get("use_gpu", 1)
    gpu_ID = kwargs.get("gpu_ID", 0)
    eps2 = kwargs.get("eps2", 0.0 | nbody_system.length**2)
# Initialize the Require Value Arrays
    Energy = []
    UncorrectedEnergy = []
    Time = []
    Kinetic = []
    Potential = []
    L = []
    P = []
    i = 0
# Locates all Restart HDF5 Files for a Run
    file_loc = restart_dir+cluster_name
    search = glob.glob(file_loc+"*.hdf5")
# Automatically Set the Timestep Parameter
    time_grab = ((search[0])[:-11]).split("_")
    timestep = float(time_grab[-2]) | nbody_system.time
# Loops through all Restart Files & Append Energy and Momentum to Corresponding Arrays
    for key in search:
    # Only select the '*restart.stars.hdf5' Files
        if i%2==0:
        # Set the Restart File Name
            restart_file = key[:-11]
        # Retrieve the Current Time from the File Name
            time_grab = key.split("_")
            time = float(time_grab[-2]) | nbody_system.time
        # Sets the GPU Useage
            if use_gpu == 1:
                gravity = ph4(number_of_workers = num_workers, redirection = "none", mode = "gpu")
            else:
                gravity = ph4(number_of_workers = num_workers, redirection = "none")
        # Initializing PH4 with Initial Conditions
            print "Initializing gravity"
            gravity.initialize_code()
            gravity.parameters.set_defaults()
            gravity.parameters.begin_time = time
            gravity.parameters.epsilon_squared = eps2
            gravity.parameters.timestep_parameter = timestep.number
        # Setting up the Code to Run with GPUs
            gravity.parameters.use_gpu = use_gpu
            gravity.parameters.gpu_id = gpu_ID
        # Initializing Kepler and SmallN
            print "Initializing Kepler"
            kep = Kepler(None, redirection = "none")
            kep.initialize_code()
            print "Initializing SmallN"
            util.init_smalln()
        # Retrieving the Master Set & Multiples Instance
            MasterSet = []
            print "Retrieving data"
            MasterSet, multiples_code = read.read_state_from_file(restart_file, gravity, kep, util.new_smalln())
        # Setting Up the Stopping Conditions in PH4
            stopping_condition = gravity.stopping_conditions.collision_detection
            stopping_condition.enable()
            sys.stdout.flush()
        # Starting the AMUSE Channel for PH4
            grav_channel = gravity.particles.new_channel_to(MasterSet)
            print "Reload Successful! Getting Values!"
        # Calculate the Top-Level Energy
            U = multiples_code.potential_energy
            T = multiples_code.kinetic_energy
            Etop = T + U
        # Calculate the Angular & Linear Momentum (Needs Corrections)
            angular_momentum = MasterSet.total_angular_momentum()
            momentum = MasterSet.total_momentum()
        # Calculate the True Total Energy via Corrections
            Nmul, Nbin, Emul = multiples_code.get_total_multiple_energy()
            Etot = Etop + Emul
            Eext = multiples_code.multiples_external_tidal_correction
            Eint = multiples_code.multiples_internal_tidal_correction
            Eerr = multiples_code.multiples_integration_energy_error
            Edel = Eext + Eint + Eerr
            Ecor = Etot - Edel
        # Stop the Gravity Codes
            gravity.stop()
            kep.stop()
            util.stop_smalln()
        # Append the Values to Their Arrays
        # Note: These do not carry AMUSE units!
            Energy.append(Ecor.number)
            UncorrectedEnergy.append(Etop.number)
            Time.append(time.number)
            Kinetic.append(T.number)
            Potential.append(U.number)
            L.append(angular_momentum.number)
            P.append(momentum.number)
        i+=1
    return Energy, UncorrectedEnergy, Time, Kinetic, Potential, L, P

# THE BELOW GRAPHS WILL BE IN WHATEVER UNIT YOU SAVE IN. IF YOU CHANGE ANYTHING THEY WILL NOT BE NBODY UNITS

def AngularMomentumGraph(Time, L, cluster_name, dpi = 150):
# This function makes a graph of the angular momentum and saves it in the Graphs folder
    res_dir = os.getcwd()+"/Graphs"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    normL = []
    for thing in L:
        normL.append(np.linalg.norm(thing))
    plt.grid()   
#    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title('Angular Momentum Over Time', fontsize=25)
    plt.xlabel('Time (Nbody Time)', fontsize=20)
    plt.ylabel('Angular Momentum (Nbody Units)', fontsize=20)     
    plt.plot(Time, normL, color='green')
    plt.ioff()
    plt.savefig("Graphs/"+cluster_name+'AngularMomentumGraph.png', format="png", dpi=dpi)
    plt.clf()
    plt.close('all')

# Input the Time and momentum as arrays for this function followed by a string of the cluster name. You may change the dpi for resolution purposes.
def MomentumGraph(Time, P, cluster_name, dpi = 150):
# This function makes a graph of the total momentum and saves it in the Graphs folder

    res_dir = os.getcwd()+"/Graphs"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    normP = []
    for thing in P:
        normP.append(np.linalg.norm(thing))
    plt.grid()   
#    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title('Momentum Over Time', fontsize=25)
    plt.xlabel('Time (Nbody Time)', fontsize=20)
    plt.ylabel('Momentum (Nbody Units)', fontsize=20)     
    plt.plot(Time, normP, color='green')
    plt.ioff()
    plt.savefig("Graphs/"+cluster_name+'MomentumGraph.png', format="png", dpi=dpi)
    plt.clf()
    plt.close('all')

# Input the Time and the respective energies (Total then Kinetic then Potential) as arrays for this function followed by a string of the cluster name. You may change the dpi for resolution purposes.

def EnergyGraph(Time, Energy, T, U, cluster_name, dpi = 150):

# This function makes a graph of the Potential, Kinetic, and Total and saves it in the Graphs folder
    res_dir = os.getcwd()+"/Graphs"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    plt.grid()   
#    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title('Energy Over Time', fontsize=25)
    plt.xlabel('Time (Nbody Time)', fontsize=20)
    plt.ylabel('Energy (Nbody Units)', fontsize=20)     
    plt.plot(Time, Energy, color='blue', label = "Energy")
    plt.plot(Time, T, color='green', label = "Kinetic Energy")
    plt.plot(Time, U, color='red', label = "Potential Energy")
    plt.ioff()
    plt.savefig("Graphs/"+cluster_name+'EnergyGraph.png', format="png", dpi=dpi)
    plt.clf()
    plt.close('all')

def EnergyGraph2(Time, UncorrectedEnergy, Energy, T, U, cluster_name, dpi = 150):

# This function makes a graph of the Potential, Kinetic, and Total Without Multiples Correction and saves it in the Graphs folder
    res_dir = os.getcwd()+"/Graphs"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    plt.grid()
#    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title('Energy Over Time', fontsize=25)
    plt.xlabel('Time (Nbody Time)', fontsize=20)
    plt.ylabel('Energy (Nbody Units)', fontsize=20)
    plt.plot(Time, Energy, color='blue', label = "Corrected Energy")
    plt.plot(Time, T, color='green', label = "Kinetic Energy")
    plt.plot(Time, U, color='red', label = "Potential Energy")
    plt.plot(Time, UncorrectedEnergy, color = 'orange', label = "Uncorrected Energy")
    plt.legend()
    plt.ioff()
    plt.savefig("Graphs/"+cluster_name+'EnergyGraphNoCorrection.png', format="png", dpi=dpi)
    plt.clf()
    plt.close('all')

