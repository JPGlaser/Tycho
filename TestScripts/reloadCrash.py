# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import sys, os, math
import numpy as np
import matplotlib as plt
import time as tp
import random as rp
from optparse import OptionParser
import glob

# Importing cPickle/Pickle
try:
   import pickle as pickle
except:
   import pickle

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
from amuse.community.ph4.interface import ph4
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.community.sse.interface import SSE
from amuse.couple import multiples

# Import the Tycho Packages
from tycho import create, util, read, write 

if __name__=="__main__":

# Creating Command Line Argument Parser
    parser = OptionParser()
    parser.add_option("-g", "--no-gpu", dest="use_gpu",default=1, type="int", \
                      help="Disable GPU for computation by setting to a value other than 1.")
    parser.add_option("-i", "--gpu-id", dest="gpu_ID", default= -1, type="int", \
                      help="Select which GPU to use by device ID.")
    parser.add_option("-p", "--num-psys", dest="num_psys", default=32, type="int", \
        	          help="Enter the number of planetary systems desired.")
    parser.add_option("-s", "--num-stars", dest="num_stars", default=750, type="int", \
        	          help="Enter the number of stars desired.")
    parser.add_option("-t", "--timestep", dest="dt", default=0.05, type="float", \
        	          help="Enter the PH4 timestep in N-Body Units.")
    parser.add_option("-c", "--cluster-name", dest="cluster_name", default=None, type="str", \
        	          help="Enter the name of the Cluster (Defaults to Numerical Naming Scheme).")
    parser.add_option("-w", "--w0", dest="w0", default=2.5, type="float", \
        	          help="Enter the w0 parameter for the King's Model.")
    parser.add_option("-N", "--num-steps", dest="num_steps", default=1000, type="int", \
        	          help="Enter the total number of time-steps to take.")
    parser.add_option("-b", "--IBF", dest="IBF", default = 0.5, type ="float", \
        		      help = "Enter the initial binary fraction.")
    parser.add_option("-S", "--seed", dest="seed", default = 1234, type="int", \
                      help = "Enter a random seed for the simulation")
    parser.add_option("-R","--restart",dest="restart_file",default="_restart", type="str", \
                      help = "Enter the name for the restart_file, (Defaults to _restart.hdf5")
    (options, args) = parser.parse_args()

# Set Commonly Used Python Variables from Options
    num_stars = 100
    num_psys = 10
    cluster_name = "crashTest1"
    restart_file = "Restart/"+cluster_name+"_restart"
    write_file_base = restart_file
    crash_file = "CrashSave/"+cluster_name+"_Save"
# Try to Import the Cluster from File or Create a New One
    MasterSet, ic_array, converter = read.read_initial_state(cluster_name)
    # TODO: Create the "stars" and "planets" sets and open up their channels to the master set.
    #       It should match the style designed below.
    read_from_file = True

# Create Initial Conditions Array
    initial_conditions = util.store_ic(converter, options)
        
# Write the Initial State 
    if not read_from_file:
        write.write_initial_state(MasterSet, initial_conditions, cluster_name)

# Define PH4-Related Initial Conditions
    time = 0.0 | nbody_system.time
    delta_t = options.dt | nbody_system.time
    number_of_steps = options.num_steps
    end_time = number_of_steps*delta_t
    num_workers = 1
    eps2 = 0.0 | nbody_system.length**2
    use_gpu = options.use_gpu
    gpu_ID = options.gpu_ID
# Need to find a way to recover time on crash    

# Setting PH4 as the Top-Level Gravity Code
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

    print(time)

    time, multiples_code = read.recover_crash(crash_file, gravity, kep, util.new_smalln)
    

# Setting Up the Stopping Conditions in PH4
    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()
    sys.stdout.flush()

# Starting the AMUSE Channel for PH4
    grav_channel = gravity.particles.new_channel_to(MasterSet)
