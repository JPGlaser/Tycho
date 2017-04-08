# Python Classes/Functions used to Import Tycho Datasets

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import math
import numpy as np
import matplotlib as plt
import random as rp

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

# Import cPickle/Pickle
try:
   import cPickle as pickle
except:
   import pickle

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def read_initial_state(file_prefix):
    ''' Reads in an initial state for the Tycho Module.
        file_prefix: String Value for a Prefix to the Saved File
    ''' 
# TODO: Convert the saved datasets from SI to NBody. Also everything else in this function.

# First, Define the Directory where Initial State is Stored
    file_dir = os.getcwd()+"/InitialState"
    file_base = file_dir+"/"+file_prefix
# Second, Read the Master AMUSE Particle Set from a HDF5 File
    file_format = "hdf5"
    master_set = read_set_from_file(file_base+"_particles.hdf5", format=file_format, close_file=True)
# Third, unPickle the Initial Conditions Array
    ic_file = open(file_base+"_ic.pkl", "rb")
    ic_array = pickle.load(ic_file)
    ic_file.close()
# Fourth, Define the Master Set's Converter
    converter = nbody_system.nbody_to_si(ic_array.total_mass, ic_array.viral_radius)
    return master_set, ic_array, converter

