import numpy as np
import random as rp
import os, sys
import scipy as sp
from optparse import OptionParser
from scipy import optimize
from scipy import special
import pickle
import glob
from collections import defaultdict

# Import the Amuse Base Packages
import amuse
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.datamodel import particle_attributes
from amuse.io import *
from amuse.lab import *

from tycho import util, create, read, write, stellar_systems

set_printing_strategy("custom", preferred_units = [units.MSun, units.AU, units.day], precision = 6, prefix = "", separator = "[", suffix = "]")

def get_first_and_last_states(bodies, end_time=(10 | units.Myr)):
    first = bodies.get_state_at_timestamp(0 | units.yr)
    last = bodies.get_state_at_timestamp(end_time)
    result = [first, last]
    for state in result:
        stellar_systems.update_host_star(state)
    return (first, last)

if __name__=="__main__":

    # ------------------------------------- #
    #      Setting up Required Variables    #
    # ------------------------------------- #
    parser = OptionParser()
    parser.add_option("-d", "--rootdirectory", dest="rootDir", default=None, type="str",
                      help="Enter the full directory of the Root .")
    parser.add_option("-S", "--serial", dest="doSerial", action="store_true",
                      help="Run the program in serial?.")
    (options, args) = parser.parse_args()
    if options.rootDir != None:
        rootDir = options.rootDir
    else:
        rootDir = '/home/draco/jglaser/Public/Tycho_Runs/MarkG/'
    doSerial = options.doSerial

    paths_of_hdf5_files = glob.glob(rootDir+'*/Encounters/*/Enc-*Rot-*.hdf5')
    cluster_names = [path.split("/")[-4] for path in paths_of_hdf5_files]
    primary_sysIDs = [path.split("/")[-2] for path in paths_of_hdf5_files]
    enc_IDs = [path.split("/")[-1].split("-")[1].split('Rot')[0] for path in paths_of_hdf5_files]
    rot_IDs = [path.split("/")[-1].split("-")[2].split('.hdf5')[0] for path in paths_of_hdf5_files]

    counter_finished = 0
    for rot_ID in rot_IDs:
        if rot_ID == '100':
            counter_finished += 1
    print "Number of Encounters Fully Simulated:", counter_finished
    print "Number of Initial States Simulated:", len(rot_IDs)
    print "Total Number of Initial States to Simulate:", int(len(glob.glob(rootDir+'*/Encounters/*/')))*100

    flDB_file = open(rootDir+"TotalI-F_DB.pkl", "w")
    total_flDB = defaultdict(list)
    for i, path in enumerate(paths_of_hdf5_files[::10]):
        system = read_set_from_file(path, 'hdf5',version='2.0', copy_history=True, close_file=True)
        f_and_l = get_first_and_last_states(system)
        total_flDB[cluster_names[i]].append(f_and_l)
        if i%10==0:
            pickle.dump(total_flDB, flDB_file)
            print "!!!!!! Percent Completed:", i*1.0/len(paths_of_hdf5_files[::10])*100

    pickle.dump(total_flDB, flDB_file)
    encounter_cut_file.close()
