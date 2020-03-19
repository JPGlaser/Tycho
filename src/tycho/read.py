# Python Classes/Functions used to Import Tycho Datasets

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# TO-DO: Add time back to the read state function for Tyler's code

# Importing Necessary System Packages
import math
import io
import os
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
#from amuse.couple import multiples

# Import the Amuse Stellar Packages
from amuse.ic.kingmodel import new_king_model
from amuse.ic.kroupa import new_kroupa_mass_distribution

# Import cPickle/Pickle
try:
   import pickle as pickle
except:
   import pickle

# Tycho util import
from tycho import util
from tycho import multiples2 as multiples

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def read_initial_state(file_prefix):
    ''' Reads in an initial state for the Tycho Module.
        file_prefix: String Value for a Prefix to the Saved File
    ''' 
# TODO: Also everything else in this function.

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
# Fourth, convert ic_array.total_smass and viral_radius from strings to floats
    total_smass = float(ic_array.total_smass) | units.kg
    viral_radius = float(ic_array.viral_radius) | units.m
# Fifth, Define the Master Set's Converter
    converter = nbody_system.nbody_to_si(total_smass, viral_radius)
    return master_set, ic_array, converter


# ------------------------------------ #
#           RESTART FUNCTION           #
# ------------------------------------ #

def read_state_from_file(restart_file, gravity_code, kep, SMALLN):

    stars = read_set_from_file(restart_file+".stars.hdf5",'hdf5',version='2.0', close_file=True).copy()
    stars_python = read_set_from_file(restart_file+".stars_python.hdf5",'hdf5',version='2.0', close_file=True).copy()
    with open(restart_file + ".bookkeeping", "rb") as f:
        bookkeeping = pickle.load(f)
        f.close()
    print(bookkeeping)
    root_to_tree = {}
    for root in stars:
        if hasattr(root, 'components') and not root.components is None:
            root_to_tree[root] = datamodel.trees.BinaryTreeOnParticle(root.components[0])
    gravity_code.particles.add_particles(stars)
#    print bookkeeping['model_time']
#    gravity_code.set_begin_time = bookkeeping['model_time']
    multiples_code = multiples.Multiples(gravity_code, SMALLN, kep, gravity_constant=units.constants.G)
    multiples_code.neighbor_distance_factor = bookkeeping['neighbor_distance_factor']
    multiples_code.neighbor_veto = bookkeeping['neighbor_veto']
    multiples_code.multiples_external_tidal_correction = bookkeeping['multiples_external_tidal_correction']
    multiples_code.multiples_integration_energy_error = bookkeeping['multiples_integration_energy_error']
    multiples_code.multiples_internal_tidal_correction = bookkeeping['multiples_internal_tidal_correction']
    multiples.root_index = bookkeeping['root_index']
    multiples_code.root_to_tree = root_to_tree
#    multiples_code.set_model_time = bookkeeping['model_time']

    return stars_python, multiples_code

# ------------------------------------------ #
#           RESTART CRASH FUNCTION           #
# ------------------------------------------ #

def recover_crash(restart_file, gravity_code, kep, SMALLN):
# NEEDS SOME TENDER LOVE AND CARE
    stars = read_set_from_file(restart_file+".stars.hdf5",'hdf5',version='2.0', close_file=True).copy()
    stars_python = read_set_from_file(restart_file+".stars_python.hdf5",'hdf5',version='2.0', close_file=True).copy()
    with open(restart_file + ".bookkeeping", "rb") as f:
        bookkeeping = pickle.load(f)
        f.close()
    print(bookkeeping)
    root_to_tree = {}
    for root in stars:
        if hasattr(root, 'components') and not root.components is None:
            root_to_tree[root] = datamodel.trees.BinaryTreeOnParticle(root.components[0])
    #gravity_code.particles.add_particles(stars)
    #print bookkeeping['model_time']
    gravity_code.set_begin_time = bookkeeping['model_time']
    multiples_code = multiples.Multiples(gravity_code, SMALLN, kep, gravity_constant=units.constants.G)
    multiples_code.neighbor_distance_factor = bookkeeping['neighbor_distance_factor']
    multiples_code.neighbor_veto = bookkeeping['neighbor_veto']
    multiples_code.multiples_external_tidal_correction = bookkeeping['multiples_external_tidal_correction']
    multiples_code.multiples_integration_energy_error = bookkeeping['multiples_integration_energy_error']
    multiples_code.multiples_internal_tidal_correction = bookkeeping['multiples_internal_tidal_correction']
    multiples.root_index = bookkeeping['root_index']
    multiples_code.root_to_tree = root_to_tree
    #multiples_code.set_model_time = bookkeeping['model_time']

    return bookkeeping['model_time'], multiples_code    
    
