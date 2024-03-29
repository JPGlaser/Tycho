import numpy as np
import random as rp
import os, sys
import scipy as sp
from optparse import OptionParser
from scipy import optimize
from scipy import special
import pickle as pickle
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

def get_first_and_last_states(bodies, end_time=(10 | units.Myr), kepler_worker=None):
    if kepler_worker == None:
        if converter == None:
            converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
        kep_p = Kepler(unit_converter = converter, redirection = 'none')
        kep_p.initialize_code()
    else:
        kep_p = kepler_worker
    first = bodies.get_state_at_timestamp(0 | units.yr)
    last = bodies.get_state_at_timestamp(end_time)
    result = [first, last]
    for state in result:
        #print state
        #print util.get_stars(state)
        #print util.get_planets(state).id
        #print util.get_planets(state).host_star
        stellar_systems.update_host_star(state, kepler_worker=kep_p)
    if kepler_worker ==None:
        kep_p.stop()
    return (first.copy(), last.copy())

if __name__=="__main__":

    # ------------------------------------- #
    #      Setting up Required Variables    #
    # ------------------------------------- #
    parser = OptionParser()
    parser.add_option("-d", "--rootdirectory", dest="rootDir", default=None, type="str",
                      help="Enter the full directory of the Root Folder.")
    parser.add_option("-S", "--serial", dest="doSerial", action="store_true",
                      help="Run the program in serial?.")
    parser.add_option("-r", "--sample", dest="sample_rate", default=10, type="int",
                      help="Sampling Rate for the clusters.")
    parser.add_option("-n", "--number_of_rotations", dest="num_rot", default=100, type="int",
                      help="Flag to turn on for running the script over a series of multiple clusters.")
    (options, args) = parser.parse_args()
    if options.rootDir != None:
        rootDir = options.rootDir
    else:
        rootDir = os.getcwd()
    doSerial = options.doSerial
    sample_rate = options.sample_rate
    num_rot = options.num_rot

    # Bring Root Directory Path Inline with os.cwd()
    if rootDir.endswith("/"):
        rootDir = rootDir[:-1]

    paths_of_hdf5_files = glob.glob(rootDir+'/*/Encounters/*/Enc-*Rot-*.hdf5')
    cluster_names = [path.split("/")[-4] for path in paths_of_hdf5_files]
    primary_sysIDs = [path.split("/")[-2] for path in paths_of_hdf5_files]
    enc_IDs = [path.split("/")[-1].split("-")[1].split('Rot')[0] for path in paths_of_hdf5_files]
    rot_IDs = [path.split("/")[-1].split("-")[2].split('.hdf5')[0] for path in paths_of_hdf5_files]

    print(cluster_names)

    counter_finished = 0
    for rot_ID in rot_IDs:
        if rot_ID == str(num_rot):
            counter_finished += 1
    print("Number of Encounters Fully Simulated:", counter_finished)
    print("Number of Initial States Simulated:", len(rot_IDs))
    if len(cluster_names) > 1:
        print("Total Number of Initial States Simulated:", int(len(glob.glob(rootDir+'/*/Encounters/*/')))*(num_rot+1))

    converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    kep = Kepler(unit_converter = converter, redirection = 'none')
    kep.initialize_code()

    total_flDB = defaultdict(list)
    for i, path in enumerate(paths_of_hdf5_files[::sample_rate]):
        system = read_set_from_file(path, 'hdf5',version='2.0', copy_history=True, close_file=True)
        try:
            util.get_planets(system).host_star
        except:
            print("!!!!!!", util.timestamp(), "Skipping", path.split("/")[-1], "for Star ID", path.split("/")[-2], "in Cluster", cluster_names[i])
            continue
        f_and_l = get_first_and_last_states(system, kepler_worker=kep)
        total_flDB[cluster_names[i]].append((path, f_and_l))
        if i%10==0:
            print("!!!!!!", util.timestamp(), "Percent Completed:", i*1.0/len(paths_of_hdf5_files[::sample_rate])*100)

    sys.setrecursionlimit(13438)
    for key in list(total_flDB.keys()):
        temp = defaultdict(list)
        temp[key] = total_flDB[key]
        flDB_file = open(rootDir+"/"+str(key)+"-Total_IF_DB.pkl", "w")
        pickle.dump(temp, flDB_file, protocol=pickle.HIGHEST_PROTOCOL)
        flDB_file.close()
    kep.stop()
