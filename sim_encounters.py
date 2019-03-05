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



# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

if __name__=="__main__":

# ------------------------------------- #
#      Setting up Required Variables    #
# ------------------------------------- #
cluster_name =
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

# Preform a Cut on the Encounter Database


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
