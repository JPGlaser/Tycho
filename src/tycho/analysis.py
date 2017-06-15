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

# Import the Amuse Base Packages
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.datamodel import particle_attributes
from amuse.io import *
from amuse.lab import *

# Import the Amuse Stellar Packages
from amuse.ic.kingmodel import new_king_model
from amuse.ic.kroupa import new_kroupa_mass_distribution

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

def GetValues(restart_base, num_files, num_workers = 1, use_gpu = 1, gpu_ID = 0, dt = 0.50 | nbody_system.time, time = 0.55 | nbody_system.time, eps2 = 0.0 | nbody_system.length**2, delta_t = 0.05 | nbody_system.time):
# This function uses calls upon the restart fuction to reload the multiples from the simulation to use 
# the multiples function to get the Energy and it correction, num_files can probably be replaced if we 
# can measure how many files are in a cluster, maybe glob.glob can do that.

# This function returns the arrays below:
    Energy = []
    Time = []
    Kinetic = []
    Potential = []
    L = []
    P = []

    i = 0
    while i < num_files:
# This loop will go through the restart files of teh cluster and append energy and momentum values to the corresponding arrays
# First get the restart naming correct
        step = time.number
        restart_file = restart_base + str(step)
        print restart_file
        
        if use_gpu == 1:
            gravity = ph4(number_of_workers = num_workers, redirection = "none", mode = "gpu")
        else:
            gravity = grav(number_of_workers = num_workers, redirection = "none")


# Initializing PH4 with Initial Conditions
        gravity.initialize_code()
        gravity.parameters.set_defaults()
        gravity.parameters.begin_time = time
        gravity.parameters.epsilon_squared = eps2
        gravity.parameters.timestep_parameter = delta_t.number

# Setting up the Code to Run with GPUs Provided by Command Line
        gravity.parameters.use_gpu = use_gpu
        gravity.parameters.gpu_id = gpu_ID

# Initializing Kepler and SmallN
        kep = Kepler(None, redirection = "none")
        kep.initialize_code()
        util.init_smalln()
        MasterSet = []
        MasterSet, multiples_code = read.read_state_from_file(restart_file, gravity, kep, util.new_smalln())

# Setting Up the Stopping Conditions in PH4
        stopping_condition = gravity.stopping_conditions.collision_detection
        stopping_condition.enable()
        sys.stdout.flush()

# Starting the AMUSE Channel for PH4
        grav_channel = gravity.particles.new_channel_to(MasterSet)
        
        print "Reload Successful"
        
        U = multiples_code.potential_energy
        T = multiples_code.kinetic_energy
        Etop = T + U

        angular_momentum = MasterSet.total_angular_momentum()
        momentum = MasterSet.total_momentum()

        Nmul, Nbin, Emul = multiples_code.get_total_multiple_energy()
        tmp1,tmp2,Emul2 = multiples_code.get_total_multiple_energy2()
        Etot = Etop + Emul
        Eext = multiples_code.multiples_external_tidal_correction
        Eint = multiples_code.multiples_internal_tidal_correction
        Eerr = multiples_code.multiples_integration_energy_error
        Edel = multiples_code.multiples_external_tidal_correction \
            + multiples_code.multiples_internal_tidal_correction \
                + multiples_code.multiples_integration_energy_error
        Ecor = Etot - Edel
        T = multiples_code.kinetic_energy        
        U = multiples_code.potential_energy

        gravity.stop()
        kep.stop()
        util.stop_smalln()

        Energy.append(Ecor.number)
        Time.append(step)
        Kinetic.append(T.number)
        Potential.append(U.number)
        L.append(angular_momentum.number)
        P.append(momentum.number)

        time += dt
        i+=1
    return Energy, Time, Kinetic, Potential, L, P


# THE GRAPHS WILL BE IN WHATEVER UNIT YOU SAVE IN. IF YOU CHANGE ANYTHING THEY WILL NOT BE NBODY UNITS


# Input the Time and Angular momentum as arrays for this function followed by a string of the cluster name. You may change the dpi for resolution purposes.
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
    plt.plot(Time, Energy, color='blue')
    plt.plot(Time, T, color='green')
    plt.plot(Time, U, color='red')
    plt.ioff()
    plt.savefig("Graphs/"+cluster_name+'EnergyGraph.png', format="png", dpi=dpi)
    plt.clf()
    plt.close('all')
