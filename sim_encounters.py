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
    current_starID = job_queue.get()
    desiredFunction(current_starID)
    job_queue.task_done()
    # Announce to Terminal that the Current Task is Done
    sys.stdout.flush()
    print util.timestamp(), "Star ID", current_starID, "has finished processing!"
    print util.timestamp(), "There are", job_queue.qsize(), "stars left to process!"
    sys.stdout.flush()

def mpScatterExperiments(star_ids, desiredFunction):
    for starID in star_ids:
        job_queue.put(starID)
    num_of_cpus = mp.cpu_count()-2
    for i in range(num_of_cpus):
        th = threading.Thread(target=remote_process, args=(desiredFunction,))
        th.daemon = True
        th.start()
    job_queue.join()

def bulk_run_for_star(star_id, encounter_db, dictionary_for_results, **kwargs):
    max_number_of_rotations = kwargs.get("maxRotations", 10)
    max_runtime = kwargs.get("maxRunTime", 1 | units.Myr)
    delta_time = kwargs.get("dt", 0.1 | units.yr)
    # Set Up Output Directory Structure
    output_KeyDirectory = os.getcwd()+"/Encounters/"+str(star_id)
    # Set Up the Results Dictionary to Store Initial and Final ParticleSets for this Star
    dictionary_for_results.setdefault(star_id, {})
    encounter_id = 0
    for encounter in encounter_db[star_id]:
        # Set Up Subdirectory for this Specific Encounter
        output_EncDirectory = output_KeyDirectory+"/Enc-"+str(encounter_number)
        if not os.path.exists(output_EncDirectory): os.mkdir(output_EncDirectory)
        # Set up Encounter Key for this Specific Encounter for this Specific Star
        dictionary_for_results[star_id].setdefault(encounter_id, {})
        rotation_id = 0
        while rotation_id <= max_number_of_rotations:
            # Set Up Output Directory for this Specific Iteration
            output_HDF5File = output_EncDirectory+"Rot-"+str(rotation_id)+'.hdf5'
            # Remove Jupiter and Add Desired Planetary System
            enc_bodies = replace_planetary_system(encounter.copy())
            print enc_bodies()
            # Set up Rotation Key for this Specific Iteration for this Specific Encounter for this Specific Star
            dictionary_for_results[star_id][encounter_id].setdefault(rotation_id, [])
            # Store Initial Conditions
            dictionary_for_results[star_id][encounter_id][rotation_id].append(enc_bodies.copy())
            # Run Encounter
            # TODO: Finalize Encounter Patching Methodology with SecularMultiples
            enc_bodies = run_collision(enc_bodies, max_runtime, delta_time, output_HDF5File, doEncPatching=False)
            # Store Final Conditions
            dictionary_for_results[star_id][encounter_id][rotation_id].append(enc_bodies.copy())
            rotation_id += 1
        encounter_id += 1
    sys.stdout.flush()
    print util.timestamp(), "All Encounters Simulated for Star ID:", star_id
    sys.stdout.flush()

def run_collision(GravitatingBodies, end_time, delta_time, save_file, **kwargs):
    # Define Additional User Options and Set Defaults Properly
    converter = kwargs.get("converter", None)
    doEncPatching = kwargs.get("doEncPatching", False)
    doVerboseSaves = kwargs.get("doVerboseSaves", False)
    if converter == None:
        converter = nbody_system.nbody_to_si(GravitatingBodies.mass.sum(), 2 * np.max(GravitatingBodies.radius.number) | GravitatingBodies.radius.unit)
    # Storing Initial Center of Mass Information for the Encounter
    rCM_i = GravitatingBodies.center_of_mass()
    vCM_i = GravitatingBodies.center_of_mass_velocity()
    # Fixing Stored Encounter Particle Set to Feed into SmallN
    GravitatingBodies = Particles(particles=GravitatingBodies)
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
    gravity.particles.add_particles(GravitatingBodies) # adds bodies to gravity calculations
    gravity.commit_particles()
    channel_from_grav_to_python = gravity.particles.new_channel_to(GravitatingBodies)
    channel_from_grav_to_python.copy()
    # Setting Coarse Timesteps
    list_of_times = np.arange(0. | units.yr, end_time, delta_time)
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
            write_set_to_file(GravitatingBodies.savepoint(current_time), save_file, 'hdf5')
        else:
            # Write a Save at the Begninning, Middle & End Times
            if stepNumber==0 or stepNumber==len(list_of_times) or stepNumber==len(list_of_times)/2:
                # Write Set to File
                write_set_to_file(GravitatingBodies.savepoint(current_time), save_file, 'hdf5')
        # Check to See if the Encounter is Declared "Over" Every 50 Timesteps
        if stepNumber%50:
            over = gravity.is_over()
            if over:
                gravity.update_particle_tree()
                gravity.update_particle_set()
                gravity.particles.synchronize_to(GravitatingBodies)
                channel_from_grav_to_python.copy()
                print "Encounter has finished at Step #", stepNumber
                break
            else:
                print "Encounter has NOT finished at Step #", stepNumber
        stepNumber +=1
    # Stop the Gravity Code Once the Encounter Finishes
    gravity.stop()
    # Seperate out the Systems to Prepare for Encounter Patching
    if doEncPatching:
        ResultingPSystems = stellar_systems.get_planetary_systems_from_set(GravitatingBodies, converter=converter, RelativePosition=True)
    else:
        ResultingPSystems = stellar_systems.get_planetary_systems_from_set(GravitatingBodies, converter=converter, RelativePosition=False)
    return ResultingPSystems

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def CutOrAdvance(enc_bodies, primary_sysID, converter=None):
    bodies = enc_bodies.copy()
    if converter==None:
        converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
    systems = stellar_systems.get_planetary_systems_from_set(bodies, converter=converter, RelativePosition=False)
    # As this function is pulling from Multiples, there should never be more than 2 "Root" Particles ...
    if len(systems) > 2:
        print "Error: Encounter has more roots than expected! Total Root Particles:", len(systems)
        #print bodies
        return None
    # Assign the Primary System to #1 and Perturbing System to #2
    sys_1 = systems[int(primary_sysID)]
    secondary_sysID = [key for key in systems.keys() if key!=int(primary_sysID)][0]
    sys_2 = systems[secondary_sysID]
    # Calculate Useful Quantities
    mass_ratio = sys_2.mass.sum()/sys_1.mass.sum()
    total_mass = sys_1.mass.sum() + sys_2.mass.sum()
    rel_pos = sys_1.center_of_mass() - sys_2.center_of_mass()
    rel_vel = sys_1.center_of_mass_velocity() - sys_2.center_of_mass_velocity()
    # Initialize Kepler Worker
    kep = Kepler(unit_converter = converter, redirection = 'none')
    kep.initialize_code()
    kep.initialize_from_dyn(total_mass, rel_pos[0], rel_pos[1], rel_pos[2], rel_vel[0], rel_vel[1], rel_vel[2])
    # Check to See if the Periastron is within the Ignore Distance for 10^3 Perturbation
    p = kep.get_periastron()
    ignore_distance = mass_ratio**(1./3.) * 600 | units.AU
    if p > ignore_distance:
        print "Encounter Ignored due to Periastron of", p, "and an IgnoreDistance of",ignore_distance
        return None
    # Move the Particles to be Relative to their Respective Center of Mass
    cm_sys_1, cm_sys_2 = sys_1.center_of_mass(), sys_2.center_of_mass()
    cmv_sys_1, cmv_sys_2 = sys_1.center_of_mass_velocity(), sys_2.center_of_mass_velocity()
    for particle in sys_1:
        particle.position -= cm_sys_1
        particle.velocity -= cmv_sys_1
    for particle in sys_2:
        particle.position -= cm_sys_2
        particle.velocity -= cmv_sys_2
    # Check to See if the Planets are Closer than the Ignore Distance
    # Note: This shouldn't happen in the main code, but this prevents overshooting the periastron in debug mode.
    if kep.get_separation() > ignore_distance:
        kep.advance_to_radius(ignore_distance)
    # Advance the Center of Masses to the Desired Distance in Reduced Mass Coordinates
    x, y, z = kep.get_separation_vector()
    rel_pos_f = rel_pos.copy()
    rel_pos_f[0], rel_pos_f[1], rel_pos_f[2] = x, y, z
    vx, vy, vz = kep.get_velocity_vector()
    rel_vel_f = rel_vel.copy()
    rel_vel_f[0], rel_vel_f[1], rel_vel_f[2] = vx, vy, vz
    # Transform to Absolute Coordinates from Kepler Reduced Mass Coordinates
    cm_pos_1, cm_pos_2 = sys_2.mass.sum() * rel_pos_f / total_mass, -sys_1.mass.sum() * rel_pos_f / total_mass
    cm_vel_1, cm_vel_2 = sys_2.mass.sum() * rel_vel_f / total_mass, -sys_1.mass.sum() * rel_vel_f / total_mass
    # Move the Particles to the New Postions of their Respective Center of Mass
    for particle in sys_1:
        particle.position += cm_pos_1
        particle.velocity += cm_vel_1
    for particle in sys_2:
        particle.position += cm_pos_2
        particle.velocity += cm_vel_2
    # Stop Kepler and Return the Systems as a Particle Set
    kep.stop()
    return ParticlesSuperset([sys_1, sys_2])

def replace_planetary_system(bodies, base_planet_ID=50000, converter=None):
    enc_systems = stellar_systems.get_planetary_systems_from_set(bodies, converter=converter)
    sys_with_planets = []
    # Remove Any Tracer Planets in the Encounter
    for sys_key in enc_systems:
        for particle in enc_systems[sys_key]:
            if particle.id >= base_planet_ID:
                enc_systems[sys_key].remove_particle(particle)
                sys_with_planets.append(sys_key)
    print sys_with_planets
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

    # ------------------------------------- #
    #      Setting up Required Variables    #
    # ------------------------------------- #
    parser = OptionParser()
    parser.add_option("-c", "--cluster-name", dest="cluster_name", default=None, type="str",
                      help="Enter the name of the cluster (Defaults to Numerical Naming Scheme).")
    (options, args) = parser.parse_args()
    cluster_name = options.cluster_name
    base_planet_ID = 50000

    # ------------------------------------- #
    #   Defining File/Directory Structure   #
    # ------------------------------------- #

    # Read in Encounter Directory
    encounter_file = open(os.getcwd()+"/"+cluster_name+"_encounters.pkl", "rb")
    encounter_db = pickle.load(encounter_file)
    encounter_file.close()

    # ------------------------------------- #
    #      Perform All Necessary Cuts       #
    # ------------------------------------- #

    sys.stdout.flush()
    print util.timestamp(), "Performing First Cut on Encounter Database ..."
    sys.stdout.flush()

    # Perform a Cut on the Encounter Database
    for star_ID in encounter_db.keys():
        # Cut Out Stars Recorded with Only Initialization Pickups
        if len(encounter_db[star_ID]) <= 1:
            del encounter_db[star_ID]
            continue
    for star_ID in encounter_db.keys():
        # Cut Out Stars with No Planets
        enc_id_to_cut = []
        for enc_id, encounter in enumerate(encounter_db[star_ID]):
            # Refine "No Planet" Cut to Deal with Hierarchical Stellar Systems
            # We are Looping Through Encounters to Deal with Rogue Jupiter Captures
            if len([ID for ID in encounter.id if ID >= 50000]) == 0:
                enc_id_to_cut.append(enc_id)
            elif len([ID for ID in encounter.id if ID >= 50000]) > 0:
                if len([ID for ID in encounter.id if ID <= 50000]) == 1:
                    enc_id_to_cut.append(enc_id)
        for enc_id in sorted(enc_id_to_cut, reverse=True):
            del encounter_db[star_ID][enc_id]

    sys.stdout.flush()
    print util.timestamp(), "Performing Second Cut on Encounter Database ..."
    sys.stdout.flush()

    # Perform Cut & Advancement on Systems to Lower Integration Time
    for star_ID in encounter_db.keys():
        enc_id_to_cut = []
        for enc_id, encounter in enumerate(encounter_db[star_ID]):
            PeriastronCut = CutOrAdvance(encounter, star_ID)
            if PeriastronCut != None:
                encounter_db[star_ID][enc_id] = PeriastronCut
            elif PeriastronCut == None:
                enc_id_to_cut.append(enc_id)
        for enc_id in sorted(enc_id_to_cut, reverse=True):
            del encounter_db[star_ID][enc_id]

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

    # Begin Looping Through Star IDs (Each Star is a Pool Process)
    mpScatterExperiments(star_ids, process_func)

    # Picke the Resulting Database of Initial and Final Conditions
    pickle.dump(resultDict, open(os.getcwd()+"/"+cluster_name+"_resultDB.pkl", "wb"))

    # Announce to Terminal that the Runs have Finished
    sys.stdout.flush()
    print util.timestamp(), "Cluster", cluster_name, "is finished processing!"
    sys.stdout.flush()
