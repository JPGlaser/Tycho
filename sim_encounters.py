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

def run_collision(GravitatingBodies, end_time, delta_time, save_dir, save_id, **kwargs):
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
    stepNumber +=0

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
        else:
            # Write a Save at the Begninning, Middle & End Times
            if stepNumber==0 or stepNumber==len(list_of_times) or stepNumber==len(list_of_times)/2:
                # Write Set to File
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
        ResultingPSystems = get_planetary_systems_from_set(GravitatingBodies, converter=None, RelativePosition=True)
    else:
        ResultingPSystems = None
    return ResultingPSystems

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def CutOrAdvance(enc_bodies, primary_sysID, converter=None):
    bodies = enc_bodies.copy()
    systems = stellar_systems.get_planetary_systems_from_set(bodies, converter=converter, RelativePosition=False)
    # As this function is pulling from Multiples, there should never be more than 2 "Root" Particles ...
    if len(systems) > 2:
        print "Error: Encounter has more roots than expected! Total Root Particles:", len(systems)
        return None
    # Assign the Primary System to #1 and Perturbing System to #2
    sys_1 = systems[int(primary_sysID)]
    secondary_sysID = [key for key in systems.keys() if key!=int(primary_sysID)][0]
    sys_2 = systems[secondary_sysID]

    mass_ratio = sys_2.mass.sum()/sys_1.mass.sum()
    total_mass = sys_1.mass.sum() + sys_2.mass.sum()
    rel_pos = sys_1.center_of_mass() - sys_2.center_of_mass()
    rel_vel = sys_1.center_of_mass_velocity() - sys_2.center_of_mass_velocity()

    kep = Kepler(unit_converter = converter, redirection = 'none')
    kep.initialize_code()
    kep.initialize_from_dyn(total_mass, rel_pos[0], rel_pos[1], rel_pos[2], rel_vel[0], rel_vel[1], rel_vel[2])

    p = kep.get_periastron()
    ignore_distance = mass_ratio**(1./3.) * 600 | units.AU
    if p > ignore_distance:
        print "Encounter Ignored due to Periastron of", p, "and an IgnoreDistance of", ignore_distance
        return None
    # Move the Particles to be Relative to their Respective Center of Mass
    for particle in sys_1:
        particle.position -= sys_1.center_of_mass()
        particle.velocity -= sys_1.center_of_mass_velocity()
    for particle in sys_2:
        particle.position -= sys_2.center_of_mass()
        particle.velocity -= sys_2.center_of_mass_velocity()

    # Advance the Center of Masses to the Desired Distance in Reduced Mass Coordinates
    kep.advance_to_radius(ignore_distance)
    x, y, z = kep.get_separation_vector()
    rel_pos_f = rel_pos.copy()
    rel_pos_f[0], rel_pos_f[1], rel_pos_f[2] = x, y, z
    vx, vy, vz = kep.get_velocity_vector()
    rel_vel_f = rel_vel.copy()
    rel_vel_f[0], rel_vel_f[1], rel_vel_f[2] = vx, vy, vz

    # Transform to Absolute Coordinates from Kepler Reduced Mass Coordinates
    cm_pos_1, cm_pos_2 = -sys_2.mass.sum() * rel_pos_f / total_mass, sys_1.mass.sum() * rel_pos_f / total_mass
    cm_vel_1, cm_vel_2 = -sys_2.mass.sum() * rel_vel_f / total_mass, sys_1.mass.sum() * rel_vel_f / total_mass
    # Move the Particles to the New Postions of their Respective Center of Mass
    for particle in sys_1:
        particle.position += cm_pos_1
        particle.velocity += cm_vel_1
    for particle in sys_2:
        particle.position += cm_pos_2
        particle.velocity += cm_vel_2
    kep.stop()
    return ParticlesSuperset([sys_1, sys_2])

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #
if __name__=="__main__":

    # ------------------------------------- #
    #      Setting up Required Variables    #
    # ------------------------------------- #
    cluster_name =
    base_planet_ID = 50000
    max_number_of_rotations = 100



    # ------------------------------------- #
    #   Defining File/Directory Structure   #
    # ------------------------------------- #
    output_MainDirectory = os.getcwd()+"/Encounters"
    if not os.path.exists(output_MainDirectory): os.mkdir(output_MainDirectory)


    # Read in Encounter Directory
    encounter_file = os.getcwd()+cluster_name+"_encounters.pkl"
    encounter_db = pickle.load(encounter_file)
    encounter_file.close()

    # Perform a Cut on the Encounter Database
    for star_ID in encounter_db.keys():
        # Cut Out Stars Recorded with Only Initialization Pickups
        if len(encounter_db[star_ID]) <= 1:
            del encounter_db[star_ID]
            continue
        # Cut Out Stars with No Planets
        for encounter in encounter_db[star_ID]:
            # Refine "No Planet" Cut to Deal with Hierarchical Stellar Systems
            # We are Looping Through Encounters to Deal with Rogue Jupiter Captures
            if len([ID for ID in encounter.id if ID <= base_planet_ID]) == 0:
                del encounter
                continue

    # Perform Cut & Advancement on Systems to Lower Integration Time
    for star_ID in encounter_db.keys():
        for enc_id, encounter in enumerate(encounter_db[star_ID]):
            PeriastronCut = CutOrAdvance(encounter)
            if PeriastronCut != None:
                encounter_db[star_ID][enc_id] = PeriastronCut
            elif PeriastronCut == None:
                del encounter_db[star_ID][enc_id]

    # Begin Looping Through Stars (Each Star is a Pool Process)
    for star_id in encounter_db.keys():
        output_KeyDirectory = output_MainDirectory+"/"+str(star_id)
        if not os.path.exists(output_KeyDirectory): os.mkdir(output_KeyDirectory)
        for encounter in encounter_db[star_id]:
            encounter_id = 0
            rotation_id= 0
            output_EncDirectory = output_KeyDirectory+"/Enc-"+str(encounter_number)+"Rot-"+str(rotation_id)
            if not os.path.exists(output_EncDirectory): os.mkdir(output_EncDirectory)
            while rotation_id <= max_number_of_rotations:
                # Add Other Planets
                # Preform Random Rotation
                # Store Initial Conditions
                # Run Encounter
                # Store Final Conditions
                rotation_id += 1
            encounter_id += 1
