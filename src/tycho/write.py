# Python Classes/Functions used to Export Tycho's Datasets

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import os
import numpy as np
import matplotlib as plt

# Import the Amuse Base Packages
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.io import *
from amuse.lab import *

# Import the Amuse Stellar Packages
from amuse.ic.kingmodel import new_king_model
from amuse.ic.kroupa import new_kroupa_mass_distribution

# Import cPickle/Pickle
try:
   import cPickle as pickle
except:
   import pickle

from tycho import util

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def write_initial_state(master_set, ic_array, file_prefix):
    ''' Writes out an initial state for the Tycho Module.
        master_set: The Master Amuse Particle Set used in Tycho
        ic_array: Predefined Numpy Array that Stores Initial Conditions in SI Units
        file_prefix: String Value for a Prefix to the Saved File
    '''    
# First, Define/Make the Directory for the Initial State to be Stored
    file_dir = os.getcwd()+"/InitialState"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_base = file_dir+"/"+file_prefix
# Second, Write the AMUSE Particle Set to a HDF5 File
    file_format = "hdf5"
    write_set_to_file(master_set, file_base+"_particles.hdf5", format=file_format, close_file=True)
# Third, Pickle the Initial Conditions Array
    ic_file = open(file_base+"_ic.pkl", "wb")
    pickle.dump(ic_array, ic_file)
    ic_file.close()

def write_time_step(master_set, converter, current_time, file_prefix):
    ''' Writes out necessary information for a time step.
        master_set: The Master AMUSE Particle Set used in Tycho
        multiples_code: The Multiples Instance for Tycho
        current_time: The Simulations Current Time
        file_prefix: String Value for a Prefix to the Saved File
    '''
# First, Define/Make the Directory for the Time Step to be Stored
    file_dir_MS = os.getcwd()+"/Run/MasterParticleSet"
    file_dir_CoM = os.getcwd()+"/Run/CoMSet"
    if not os.path.exists(file_dir_MS):
        os.makedirs(file_dir_MS)
    if not os.path.exists(file_dir_CoM):
        os.makedirs(file_dir_CoM)
    file_base_MS = file_dir_MS+"/"+file_prefix
    file_base_CoM = file_dir_CoM+"/"+file_prefix
# Second, Create the CoM Tree Particle Set from Multiples
# Third, Convert from NBody to SI Before Writing
    MS_SI = datamodel.ParticlesWithUnitsConverted(master_set, converter.as_converter_from_nbody_to_si())
#    CoM_SI = datamodel.ParticlesWithUnitsConverted(CoM_Set, converter.as_converter_from_nbody_to_si())
# Fourth, Write the Master AMUSE Particle Set to a HDF5 File
    file_format = "hdf5"
    write_set_to_file(MS_SI, file_base_MS+"_MS_t%.3f.hdf5" %(current_time.number), \
                      format=file_format, close_file=True)
# Fifth, Write the CoM Tree Particle Set to a HDF5 File
#    write_set_to_file(CoM_SI, file_base_CoM+"_CoM_t%.3f.hdf5" %(current_time.number), \
#                      format=file_format, close_file=True)





    
    



