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
import Queue
import threading

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
from tycho import util

# Import the Amuse Gravity & Close-Encounter Packages
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.community.ph4.interface import ph4

# Import the Tycho Packages
from tycho import create, util, read, write, stellar_systems

# Set Backend (For DRACO Only)
import matplotlib; matplotlib.use('agg')

# ------------------------------------- #
#   Required Non-Seperable Functions    #
# ------------------------------------- #

global job_queue
job_queue = Queue.Queue()

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

def do_all_scatters_for_single_cluster(rootExecDir, **kwargs):
    '''
    This function will run all scatters for a single cluster in serial.
    str rootExecDir -> The absolute root directory for all single cluster files.
    '''
    max_number_of_rotations = kwargs.get("maxRotations", 100)
    max_runtime = kwargs.get("maxRunTime", 10**5) # Units Years
    delta_time = kwargs.get("dt", 10) # Units Years
    GCodes = [initialize_GravCode(ph4), initialize_isOverCode()]
    # Strip off Extra '/' if added by user to bring inline with os.cwd()
    if rootExecDir.endswith("/"):
        rootExecDir = rootExecDir[:-1]
    # Define the Cluster's Name
    cluster_name = rootExecDir.split("/")[-1]
    # Generate List of Scattering IC HDF5 Paths
    paths_of_IC_files = glob.glob(rootExecDir+'/Scatter_IC/*/*.hdf5')
    # Find all Primary Star IDs
    star_IDs = [path.split("/")[-2] for path in paths_of_IC_files] # '1221'
    # Set Up Output Directory Structure
    output_MainDirectory = rootExecDir+"/Encounters"
    if not os.path.exists(output_MainDirectory): os.mkdir(output_MainDirectory)
    for star_ID in star_IDs:
        output_KeyDirectory = output_MainDirectory+"/"+star_ID
        if not os.path.exists(output_KeyDirectory): os.mkdir(output_KeyDirectory)
    for i, path_of_IC in enumerate(paths_of_IC_files):
        itteration_filename = path_of_IC.split('/')[-1] # 'Enc-0_Rot_0.hdf5'
        enc_bodies = read_set_from_file(path_of_IC, format="hdf5", version='2.0', close_file=True)
        output_HDF5File = output_MainDirectory+"/"+star_IDs[i]+"/"+itteration_filename
        print output_HDF5File
        if not os.path.exists(output_HDF5File):
            run_collision(enc_bodies, max_runtime, delta_time, output_HDF5File, GCodes=GCodes, doEncPatching=False)
        else:
            print util.timestamp(), "Skipping", itteration_filename.split(".hdf5")[0], "of system", star_IDs[i]

def initialize_GravCode(desiredCode, **kwargs):
    converter = kwargs.get("converter", None)
    n_workers = kwargs.get("number_of_workers", 1)
    if converter == None:
        converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    GCode = desiredCode(number_of_workers = n_workers, redirection = "none", convert_nbody = converter)
    GCode.initialize_code()
    GCode.parameters.set_defaults()
    if desiredCode == ph4:
        GCode.parameters.timestep_parameter = 0.05
    if desiredCode == SmallN:
        GCode.parameters.timestep_parameter = 0.05
    return GCode

def initialize_isOverCode(**kwargs):
    converter = kwargs.get("converter", None)
    if converter == None:
        converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    isOverCode = SmallN(redirection = "none", convert_nbody = converter)
    isOverCode.initialize_code()
    isOverCode.parameters.set_defaults()
    isOverCode.parameters.allow_full_unperturbed = 0
    return isOverCode

def run_collision(bodies, end_time, delta_time, save_file, **kwargs):
    # Define Additional User Options and Set Defaults Properly
    converter = kwargs.get("converter", None)
    doEncPatching = kwargs.get("doEncPatching", False)
    doVerboseSaves = kwargs.get("doVerboseSaves", False)
    GCodes = kwargs.get("GCodes", None)
    # Set Up the Integrators
    if GCodes == None:
        if converter == None:
            converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
        gravity = initialize_GravCode(ph4, converter=converter)
        over_grav = initialize_isOverCode(converter=converter)
    else:
        gravity = GCodes[0]
        over_grav = GCodes[1]
    # Storing Initial Center of Mass Information for the Encounter
    rCM_i = bodies.center_of_mass()
    vCM_i = bodies.center_of_mass_velocity()
    GravitatingBodies = Particles()
    for body in bodies:
        GravitatingBodies.add_particle(body.copy())
    # Fixing Stored Encounter Particle Set to Feed into SmallN
    #GravitatingBodies = Particles(particles=GravitatingBodies)
    if 'child1' in GravitatingBodies.get_attribute_names_defined_in_store():
        del GravitatingBodies.child1, GravitatingBodies.child2
    # Moving the Encounter's Center of Mass to the Origin and Setting it at Rest
    GravitatingBodies.position -= rCM_i
    GravitatingBodies.velocity -= vCM_i
    # Add and Commit the Scattering Particles
    gravity.particles.add_particles(GravitatingBodies) # adds bodies to gravity calculations
    gravity.commit_particles()
    # Create the Channel to Python Set & Copy it Over
    channel_from_grav_to_python = gravity.particles.new_channel_to(GravitatingBodies)
    channel_from_grav_to_python.copy()
    # Get Free-Fall Time for the Collision
    s = util.get_stars(GravitatingBodies)
    t_freefall = s.dynamical_timescale()
    # Setting Coarse Timesteps
    list_of_times = np.arange(0., end_time, delta_time) | units.yr
    stepNumber = 0
    # Integrate the Encounter Until Over ...
    for current_time in list_of_times:
        # Evolve the Model to the Desired Current Time
        gravity.evolve_model(current_time)
        # Update Python Set in In-Code Set
        channel_from_grav_to_python.copy() # original
        channel_from_grav_to_python.copy_attribute("index_in_code", "id")
        # Handle Writing Output of Integration
        if doVerboseSaves:
            # Write a Save Every Coarse Timestep
            write_set_to_file(GravitatingBodies.savepoint(current_time), save_file, 'hdf5', version='2.0')
        else:
            # Write a Save at the Begninning, Middle & End Times
            if stepNumber%25 == 0:
                # Write Set to File
                gravity.particles.synchronize_to(GravitatingBodies)
                write_set_to_file(GravitatingBodies.savepoint(current_time), save_file, 'hdf5', version='2.0')
        # Check to See if the Encounter is Declared "Over" Every 50 Timesteps
        if current_time > 1.25*t_freefall and stepNumber%25 == 0: #and len(list_of_times)/3.- stepNumber <= 0:
            over = util.check_isOver(gravity.particles, over_grav)
            print over
            if over:
                current_time += 100 | units.yr
                # Get to a Final State After Several Planet Orbits
                gravity.evolve_model(current_time)
                # Update all Particle Sets
                #gravity.update_particle_tree()
                gravity.update_particle_set()
                gravity.particles.synchronize_to(GravitatingBodies)
                channel_from_grav_to_python.copy()
                # Removes the Heirarchical Particle from the HDF5 File
                # This is done for personal convience.
                #for body in GravitatingBodies:
                #    if body.child1 != None or body.child1 != None:
                #        GravitatingBodies.remove_particle(body)
                write_set_to_file(GravitatingBodies.savepoint(current_time), save_file, 'hdf5', version='2.0')
                #print "Encounter has finished at Step #", stepNumber, '. Final Age:', current_time.in_(units.yr)
                break
            #else:
                #print "Encounter has NOT finished at Step #", stepNumber
                #t_freefall = util.get_stars(gravity.particles).dynamical_timescale()
        stepNumber +=1
    if GCodes == None:
        # Stop the Gravity Code Once the Encounter Finishes
        gravity.stop()
        over_grav.stop()
    else:
        gravity.reset()
        over_grav.reset()
    # Seperate out the Systems to Prepare for Encounter Patching
    if doEncPatching:
        ResultingPSystems = stellar_systems.get_heirarchical_systems_from_set(GravitatingBodies, converter=converter, RelativePosition=True)
    else:
        ResultingPSystems = GravitatingBodies
    return ResultingPSystems

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
                      help="Enter the full directory of the Cluster's Folder.")
    parser.add_option("-S", "--serial", dest="doSerial", action="store_true",
                      help="Run the program in serial?.")
    (options, args) = parser.parse_args()

    if options.rootDir != None:
        rootDir = options.rootDir
    else:
        rootDir = '/home/draco/jglaser/Public/Tycho_Runs/MarkG'
    # Bring Root Directory Path Inline with os.cwd()
    if rootDir.endswith("/"):
        rootDir = rootDir[:-1]

    doSerial = options.doSerial

    base_planet_ID = 50000

    # ------------------------------------- #
    #   Defining File/Directory Structure   #
    # ------------------------------------- #

    all_clusterDirs = glob.glob(rootDir+"/*/")

    # ------------------------------------- #
    #     Perform All Req. Simulations      #
    # ------------------------------------- #

    if doSerial:
        for clusterDir in all_clusterDirs:
            # Announce to Terminal that the Runs are Starting
            sys.stdout.flush()
            print util.timestamp(), "Cluster", clusterDir.split("/")[-2], "has begun processing!"
            sys.stdout.flush()
            do_all_scatters_for_single_cluster(clusterDir)
    else:
        # Begin Looping Through Clusters (Each Cluster is a Queued Process)
        mpScatterExperiments(all_clusterDirs, do_all_scatters_for_single_cluster)

    e_time = tp.time()

    # Announce to Terminal that the Runs have Finished
    sys.stdout.flush()
    print util.timestamp(), "Cluster", cluster_name, "is finished processing!"
    print util.timestamp(), "The Cluster was processed in", (e_time - s_time), "seconds!"
    sys.stdout.flush()
