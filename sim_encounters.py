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

# Importing Multiprocessing Packages
from functools import partial
import multiprocessing as mp
import queue
import threading

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
from tycho import util, scattering

# Import the Amuse Gravity & Close-Encounter Packages
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.community.ph4.interface import ph4
from amuse.community.secularmultiple.interface import SecularMultiple
from amuse.datamodel.trees import BinaryTreesOnAParticleSet
from amuse.ext.orbital_elements import new_binary_from_orbital_elements

# Import the Tycho Packages
from tycho import create, util, read, write, stellar_systems, enc_patching

# Set Backend (For DRACO Only)
import matplotlib; matplotlib.use('agg')

# ------------------------------------- #
#   Required Non-Seperable Functions    #
# ------------------------------------- #

global job_queue
job_queue = queue.Queue()

def remote_process(desiredFunction):
    while not job_queue.empty():
        try:
            current_clusterDir = job_queue.get()
        except:
            return None
        desiredFunction(current_clusterDir)
        job_queue.task_done()
        # Announce to Terminal that the Current Task is Done
        #sys.stdout.flush()
        #print "\n", util.timestamp(), "Star ID", str(current_starID), "has finished processing!"
        #print "\n", util.timestamp(), "There are", job_queue.qsize(), "stars left to process!"
        #sys.stdout.flush()

def mpScatterExperiments(list_of_clusterDirs, desiredFunction):
    for clusterDir in list_of_clusterDirs:
        job_queue.put(clusterDir)
    num_of_cpus =  mp.cpu_count()-2
    for i in range(num_of_cpus):
        th = threading.Thread(target=remote_process, args=(desiredFunction,))
        th.daemon = True
        th.start()
    job_queue.join()

def simulate_all_close_encounters(rootExecDir, **kwargs):
    '''
    This function will run all scatters for a single cluster in serial.
    str rootExecDir -> The absolute root directory for all single cluster files.
    '''
    max_number_of_rotations = kwargs.get("maxRotations", 100)
    max_runtime = kwargs.get("maxRunTime", 10**5) # Units Years
    delta_time = kwargs.get("dt", 10) # Units Years
    # Strip off Extra '/' if added by user to bring inline with os.cwd()
    if rootExecDir.endswith("/"):
        rootExecDir = rootExecDir[:-1]
    # Define the Cluster's Name
    cluster_name = rootExecDir.split("/")[-1]
    # Generate List of Scattering IC HDF5 Paths
    enc_dict = scattering.build_ClusterEncounterHistory(rootExecDir)
    # Find all Primary Star IDs
    star_IDs = enc_dict.keys() # Integer tied to StarID
    # Set Up Output Directory Structure
    output_MainDirectory = rootExecDir+"/Encounters"
    if not os.path.exists(output_MainDirectory):
        os.mkdir(output_MainDirectory)
    # Initialize the Necessary Worker Lists
    converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    KepW = []
    for i in range(2):
        KepW.append(Kepler(unit_converter = converter, redirection = 'none'))
        KepW[-1].initialize_code()
    NBodyW = [scattering.initialize_GravCode(ph4), scattering.initialize_isOverCode()]
    SecW = SecularMultiple()

    # Loop Over the Stars
    for star_ID in star_IDs:
        # Load the Close Encounter class for the Star
        EncounterHandler = scattering.CloseEncounters(enc_dict[star_ID], KeplerWorkerList = KepW, \
                                           NBodyWorkerList = NBodyW, SecularWorker = SecW)
        # Simulate Encounter
        EncounterHandler.SimAllEncounters()
        # Prepare Data for Pickling
        file_name = output_MainDirectory+"/"+str(star_ID)+"_EncounterHandler.pk"
        p_file = open(file_name, "wb")
        # Remove Worker Lists from Class for Storage
        EncounterHandler.kep = None
        EncounterHandler.NBodyCodes = None
        EncounterHandler.SecularCode = None
        # Pickle EncounterHandler Class
        # Note: This allows for ease-of-use when you want to revisit
        #       a specific star's simulation set in detail.
        pickle.dump(EncounterHandler, p_file)
        p_file.close()
        # Note: Ensure the EncounterHandler class is deleted incase
        #       of a memory leak is possible in future updates.
        del EncounterHandler
    # Stop all Workers
    for Worker in KepW+NBodyW+[SecW]:
        Worker.stop()


# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #
if __name__=="__main__":

    import time
    s_time = tp.time()
    # ------------------------------------- #
    #      Setting up Required Variables    #
    # ------------------------------------- #
    parser = OptionParser()
    parser.add_option("-d", "--rootdirectory", dest="rootDir", default=None, type="str",
                      help="Enter the full directory of the Root Folder. Defaults to your CWD unless -M is on.")
    parser.add_option("-M", "--doMultipleClusters", dest="doMultipleClusters", action="store_true",
                      help="Flag to turn on for running the script over a series of multiple clusters.")
    parser.add_option("-S", "--serial", dest="doSerial", action="store_true",
                      help="Run the program in serial?.")
    (options, args) = parser.parse_args()

    if options.doMultipleClusters:
        if options.rootDir != None:
            rootDir = options.rootDir+'/*'
        else:
            print(util.timestamp(), "Please provide the path to your root directory which contains all cluster folders!", cluster_name,"...")
    else:
        if options.rootDir != None:
            rootDir = options.rootDir
        else:
            rootDir = os.getcwd()
    # Bring Root Directory Path Inline with os.cwd()
    if rootDir.endswith("/"):
        rootDir = rootDir[:-1]

    doSerial = options.doSerial

    base_planet_ID = 50000

    # ------------------------------------- #
    #   Defining File/Directory Structure   #
    # ------------------------------------- #
    if options.doMultipleClusters:
        all_clusterDirs = glob.glob(rootDir+"/*/")
    else:
        all_clusterDirs = [rootDir+"/"]

    # ------------------------------------- #
    #     Perform All Req. Simulations      #
    # ------------------------------------- #

    if doSerial:
        for clusterDir in all_clusterDirs:
            # Announce to Terminal that the Runs are Starting
            sys.stdout.flush()
            print(util.timestamp(), "Cluster", clusterDir.split("/")[-2], "has begun processing!")
            sys.stdout.flush()
            simulate_all_close_encounters(clusterDir)
    else:
        # Begin Looping Through Clusters (Each Cluster is a Queued Process)
        mpScatterExperiments(all_clusterDirs, simulate_all_close_encounters)

    e_time = tp.time()

    # Announce to Terminal that the Runs have Finished
    sys.stdout.flush()
    print(util.timestamp(), "All clusters have finished processing.")
    print(util.timestamp(), len(all_clusterDirs), "clusters were processed in", (e_time - s_time), "seconds!")
    sys.stdout.flush()
