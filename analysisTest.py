# ------------------------------------- #
#        Python Package Importing	#
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
   import cPickle as pickle
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
from tycho import analysis, create, util, read, write

cluster_names = ["TestCluster4"]
runs_dir = "/home/draco/jthornton/Tycho/Run/MasterParticleSet/"

for cluster_name in cluster_names:
# Gather Information to Graph the 2D Image of the Cluster
    master_set, ic_array, converter = read.read_initial_state(cluster_name)
# The run_plotting may also take a dpi input if you want to change it from the default 150
    analysis.run_plotting(runs_dir, cluster_name, converter)

'''
# Gather Information to make the Energy Graphs
    Energy, UncorrectedEnergy, Time, T, U, L, P = analysis.GetValues(cluster_name)
# Plot the Corrected Energy, the Kinetic Energy, and the Potential Energy
    analysis.EnergyGraph(Time, Energy, T, U, cluster_name)
# Plot the Corrected energy, the Uncorrected Energy, the Kinetic Energy, and the Potential Energy
    analysis.EnergyGraph2(Time, UncorrectedEnergy, Energy, T, U, cluster_name)
'''
