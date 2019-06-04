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
            current_starID = job_queue.get()
        except:
            return None
        desiredFunction(current_starID)
        job_queue.task_done()
        # Announce to Terminal that the Current Task is Done
        #sys.stdout.flush()
        #print "\n", util.timestamp(), "Star ID", str(current_starID), "has finished processing!"
        #print "\n", util.timestamp(), "There are", job_queue.qsize(), "stars left to process!"
        #sys.stdout.flush()

def mpScatterExperiments(star_ids, desiredFunction):
    for starID in star_ids:
        job_queue.put(starID)
    num_of_cpus =  mp.cpu_count()-2
    for i in range(num_of_cpus):
        th = threading.Thread(target=remote_process, args=(desiredFunction,))
        th.daemon = True
        th.start()
    job_queue.join()

def bulk_run_for_star(star_id, encounter_db, dictionary_for_results, **kwargs):
    max_number_of_rotations = kwargs.get("maxRotations", 100)
    max_runtime = kwargs.get("maxRunTime", 10**5) # Units Years
    delta_time = kwargs.get("dt", 10) # Units Years
    # Set Up Output Directory Structure
    output_KeyDirectory = os.getcwd()+"/Encounters/"+str(star_id)
    # Set Up the Results Dictionary to Store Initial and Final ParticleSets for this Star
    dictionary_for_results.setdefault(star_id, {})
    encounter_id = 0
    for encounter in encounter_db[star_id]:
        # Set Up Subdirectory for this Specific Encounter
        output_EncDirectory = output_KeyDirectory+"/Enc-"+str(encounter_id)
        if not os.path.exists(output_EncDirectory): os.mkdir(output_EncDirectory)
        # Set up Encounter Key for this Specific Encounter for this Specific Star
        dictionary_for_results[star_id].setdefault(encounter_id, {})
        rotation_id = 0
        while rotation_id <= max_number_of_rotations:
            # Set Up Output Directory for this Specific Iteration
            output_HDF5File = output_EncDirectory+"Rot-"+str(rotation_id)+'.hdf5'
            # Remove Jupiter and Add Desired Planetary System
            enc_bodies = replace_planetary_system(encounter.copy())
            #print enc_bodies
            # Set up Rotation Key for this Specific Iteration for this Specific Encounter for this Specific Star
            dictionary_for_results[star_id][encounter_id].setdefault(rotation_id, [])
            # Store Initial Conditions
            dictionary_for_results[star_id][encounter_id][rotation_id].append(enc_bodies.copy())
            # Run Encounter
            # TODO: Finalize Encounter Patching Methodology with SecularMultiples
            enc_bodies = run_collision(enc_bodies, max_runtime, delta_time, output_HDF5File, doEncPatching=False)
            # Store Final Conditions
            dictionary_for_results[star_id][encounter_id][rotation_id].append(enc_bodies.copy())
            # Pickle Dictionary Every 10 Rotations
            if rotation_id%10 == 0:
                pickle.dump(resultDict, open(os.getcwd()+"/"+cluster_name+"_resultDB.pkl", "wb"))
            rotation_id += 1
        encounter_id += 1

def run_collision(bodies, end_time, delta_time, save_file, **kwargs):
    # Define Additional User Options and Set Defaults Properly
    converter = kwargs.get("converter", None)
    doEncPatching = kwargs.get("doEncPatching", False)
    doVerboseSaves = kwargs.get("doVerboseSaves", False)
    if converter == None:
        converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
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
    # Setting Up Gravity Code
    gravity = SmallN(redirection = 'none', convert_nbody = converter)
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.allow_full_unperturbed = 0
    gravity.parameters.timestep_parameter = 0.05
    gravity.particles.add_particles(GravitatingBodies) # adds bodies to gravity calculations
    gravity.commit_particles()
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
        if current_time > t_freefall and stepNumber%25 == 0: #and len(list_of_times)/3.- stepNumber <= 0:
            over = gravity.is_over()
            if over:
                current_time += 100 | units.yr
                # Get to a Final State After Several Planet Orbits
                gravity.evolve_model(current_time)
                # Update all Particle Sets
                gravity.update_particle_tree()
                gravity.update_particle_set()
                gravity.particles.synchronize_to(GravitatingBodies)
                channel_from_grav_to_python.copy()
                # Removes the Heirarchical Particle from the HDF5 File
                # This is done for personal convience.
                for body in GravitatingBodies:
                    if body.child1 != None or body.child1 != None:
                        GravitatingBodies.remove_particle(body)
                write_set_to_file(GravitatingBodies.savepoint(current_time), save_file, 'hdf5', version='2.0')
                #print "Encounter has finished at Step #", stepNumber, '. Final Age:', current_time.in_(units.yr)
                break
            #else:
                #print "Encounter has NOT finished at Step #", stepNumber
        stepNumber +=1
    # Stop the Gravity Code Once the Encounter Finishes
    gravity.stop()
    # Seperate out the Systems to Prepare for Encounter Patching
    if doEncPatching:
        ResultingPSystems = stellar_systems.get_heirarchical_systems_from_set(GravitatingBodies, converter=converter, RelativePosition=True)
    else:
        ResultingPSystems = stellar_systems.get_heirarchical_systems_from_set(GravitatingBodies, converter=converter, RelativePosition=False)
    return ResultingPSystems

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def replace_planetary_system(bodies, base_planet_ID=50000, converter=None):
    enc_systems = stellar_systems.get_heirarchical_systems_from_set(bodies, converter=converter)
    sys_with_planets = []
    # Remove Any Tracer Planets in the Encounter and Adds the Key to Add in the New System
    for sys_key in enc_systems.keys():
        for particle in enc_systems[sys_key]:
            if particle.id >= base_planet_ID:
                enc_systems[sys_key].remove_particle(particle)
                sys_with_planets.append(sys_key)
    # Allows for Planets to be Added to Single Stars
    for sys_key in enc_systems.keys():
        if (len(enc_systems[sys_key]) == 1) and (sys_key not in sys_with_planets):
            sys_with_planets.append(sys_key)
    #print sys_with_planets
    # Add in a New Planetary System
    for sys_key in sys_with_planets:
        planets = create.planetary_systems_v2(enc_systems[sys_key], 1, Jupiter=True, Earth=True, Neptune=True)
        enc_systems[sys_key].add_particles(planets)
    new_bodies = Particles()
    for sys_key in enc_systems:
        new_bodies.add_particles(enc_systems[sys_key])
    return new_bodies

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
    parser.add_option("-c", "--cluster-name", dest="cluster_name", default=None, type="str",
                      help="Enter the name of the cluster with suffixes.")
    parser.add_option("-S", "--serial", dest="doSerial", action="store_true",
                      help="Run the program in serial?.")
    (options, args) = parser.parse_args()
    if options.cluster_name != None:
        cluster_name = options.cluster_name
    else:
        directory = os.getcwd()
        cluster_name = directory.split("/")[-1]
    doSerial = options.doSerial
    base_planet_ID = 50000

    # ------------------------------------- #
    #   Defining File/Directory Structure   #
    # ------------------------------------- #

    # Read in Encounter Directory
    encounter_file = open(os.getcwd()+"/"+cluster_name+"_encounters_cut.pkl", "rb")
    encounter_db = pickle.load(encounter_file)
    encounter_file.close()

    # ------------------------------------- #
    #      Perform All Necessary Cuts       #
    # ------------------------------------- #


    print "Estimated Number of Encounters to Process:", len(encounter_db.keys())*100

    # Set Up Final Dictionary to Record Initial and Final States
    resultDict = {}

    # ------------------------------------- #
    #     Perform All Req. Simulations      #
    # ------------------------------------- #

    # Announce to Terminal that the Runs are Starting
    sys.stdout.flush()
    print util.timestamp(), "Cluster", cluster_name, "has begun processing!"
    sys.stdout.flush()

    # Set Up the Multiprocessing Pool Environment Using Partial to Send Static Variables
    process_func = partial(bulk_run_for_star, encounter_db=encounter_db, dictionary_for_results=resultDict)
    star_ids = encounter_db.keys()

    # Set Up Output Directory Structure
    output_MainDirectory = os.getcwd()+"/Encounters"
    if not os.path.exists(output_MainDirectory): os.mkdir(output_MainDirectory)
    for starID in star_ids:
        output_KeyDirectory = output_MainDirectory+"/"+str(starID)
        if not os.path.exists(output_KeyDirectory): os.mkdir(output_KeyDirectory)

    if doSerial:
        for starID in star_ids:
            print len(encounter_db[starID])
            process_func(starID)
    else:
        # Begin Looping Through Star IDs (Each Star is a Queued Process)
        mpScatterExperiments(star_ids, process_func)

    # Picke the Resulting Database of Initial and Final Conditions
    pickle.dump(resultDict, open(os.getcwd()+"/"+cluster_name+"_resultDB.pkl", "wb"))

    e_time = tp.time()

    # Announce to Terminal that the Runs have Finished
    sys.stdout.flush()
    print util.timestamp(), "Cluster", cluster_name, "is finished processing!"
    print util.timestamp(), "The Cluster was processed in", (e_time - s_time), "seconds!"
    sys.stdout.flush()
