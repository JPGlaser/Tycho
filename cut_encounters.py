# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import sys, os, math
import numpy as np
import time as tp
from optparse import OptionParser
import glob

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
from tycho import util

# Import the Amuse Gravity & Close-Encounter Packages
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler

# Import the Tycho Packages
from tycho import create, util, read, write, stellar_systems

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def CutOrAdvance(enc_bodies, primary_sysID, converter=None, **kwargs):
    bodies = enc_bodies.copy()
    KeplerWorkerList = kwargs.get("kepler_workers", None)
    # Initialize Kepler Workers if they Don't Exist
    if KeplerWorkerList == None:
        if converter == None:
            converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
        KeplerWorkerList = []
        for i in range(3):
            KeplerWorkerList.append(Kepler(unit_converter = converter, redirection = 'none'))
            KeplerWorkerList[i].initialize_code()
    systems = stellar_systems.get_heirarchical_systems_from_set(bodies, \
                                    kepler_workers=KeplerWorkerList[:2], \
                                    RelativePosition=False)
    # Deal with Possible Key Issues with Encounters with 3+ Star Particles Being Run More than Other Systems ...
    if int(primary_sysID) not in list(systems.keys()):
        print("...: Error: Previously run binary system has been found! Not running this system ...")
        print(primary_sysID)
        print(list(systems.keys()))
        print("---------------------------------")
        return None
    # As this function is pulling from Multiples, there should never be more or less than 2 "Root" Particles ...
    if len(systems) != 2:
        print("...: Error: Encounter has more roots than expected! Total Root Particles:", len(systems))
        print(bodies)
        print("---------------------------------")
        return None
    # Assign the Primary System to #1 and Perturbing System to #2
    sys_1 = systems[int(primary_sysID)]
    secondary_sysID = [key for key in list(systems.keys()) if key!=int(primary_sysID)][0]
    sys_2 = systems[secondary_sysID]
    print('All System Keys:', list(systems.keys()))
    print('Primary System Key:', primary_sysID)
    print('System 1 IDs:', sys_1.id)
    print('System 2 IDs:', sys_2.id)
    # Calculate Useful Quantities
    mass_ratio = sys_2.mass.sum()/sys_1.mass.sum()
    total_mass = sys_1.mass.sum() + sys_2.mass.sum()
    rel_pos = sys_1.center_of_mass() - sys_2.center_of_mass()
    rel_vel = sys_1.center_of_mass_velocity() - sys_2.center_of_mass_velocity()
    # Initialize Kepler Worker
    kep = KeplerWorkerList[-1]
    kep.initialize_from_dyn(total_mass, rel_pos[0], rel_pos[1], rel_pos[2], rel_vel[0], rel_vel[1], rel_vel[2])
    # Check to See if the Periastron is within the Ignore Distance for 10^3 Perturbation
    p = kep.get_periastron()
    ignore_distance = mass_ratio**(1./3.) * 600 | units.AU
    if p > ignore_distance:
        print("Encounter Ignored due to Periastron of", p.in_(units.AU), "and an IgnoreDistance of",ignore_distance)
        kep.stop()
        print("---------------------------------")
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
    # If not provided, stop Kepler and return the Systems as a Particle Set
    if KeplerWorkerList == None:
        for K in KeplerWorkerList:
            K.stop()
    # Collect the Collective Particle Set to be Returned Back
    final_set = Particles()
    final_set.add_particles(sys_1)
    final_set.add_particles(sys_2)
    print("---------------------------------")
    return final_set

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", "--cluster-name", dest="cluster_name", default=None, type="str",
                      help="Enter the name of the cluster with suffixes.")
    (options, args) = parser.parse_args()
    if options.cluster_name != None:
        cluster_name = options.cluster_name
    else:
        directory = os.getcwd()
        cluster_name = directory.split("/")[-1]

    base_planet_ID = 50000

    orig_stdout = sys.stdout
    log_file = open(os.getcwd()+"/cut_encounters.log","w")
    sys.stdout = log_file

    # Create the Kepler Workers
    KeplerWorkerList = []
    converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    for i in range(3):
        KeplerWorkerList.append(Kepler(unit_converter = converter, redirection = 'none'))
        KeplerWorkerList[i].initialize_code()

    # Read in Encounter Directory
    encounter_file = open(os.getcwd()+"/"+cluster_name+"_encounters.pkl", "rb")
    encounter_db = pickle.load(encounter_file)
    encounter_file.close()

    sys.stdout.flush()
    print(util.timestamp(), "Performing First Cut on Encounter Database ...")
    print(len(encounter_db.keys()))
    sys.stdout.flush()
    # Perform a Cut on the Encounter Database
    for star_ID in list(encounter_db.keys()):
        # Cut Out Stars Recorded with Only Initialization Pickups
        if len(encounter_db[star_ID]) == 0:
            del encounter_db[star_ID]
        if len(encounter_db[star_ID]) == 1:
            # Check to Ensure it is an Actual Multiples Initialization (AKA: 1 System)
            temp = stellar_systems.get_heirarchical_systems_from_set(encounter_db[star_ID][0], kepler_workers=KeplerWorkerList[:2])
            print(temp)
            if len(temp.keys()) <= 1:
                print(encounter_db[star_ID][0].id)
                del encounter_db[star_ID]
    print("After Removal of Just Initializations", len(encounter_db.keys()))
    for star_ID in list(encounter_db.keys()):
        # Cut Out Stars with No Planets
        enc_id_to_cut = []
        for enc_id, encounter in enumerate(encounter_db[star_ID]):
            # Refine "No Planet" Cut to Deal with Hierarchical Stellar Systems
            # We are Looping Through Encounters to Deal with Rogue Jupiter Captures
            print(star_ID, encounter.id)
            sys.stdout.flush()
            if len([ID for ID in encounter.id if ID >= base_planet_ID]) == 0:
                enc_id_to_cut.append(enc_id)
            elif len([ID for ID in encounter.id if ID >= base_planet_ID]) > 0:
                if len([ID for ID in encounter.id if ID <= base_planet_ID]) == 1:
                    enc_id_to_cut.append(enc_id)
        for enc_id in sorted(enc_id_to_cut, reverse=True):
            del encounter_db[star_ID][enc_id]
    print("After no planet encounters are removed", len(encounter_db.keys()))
    sys.stdout.flush()
    print(util.timestamp(), "Performing Second Cut on Encounter Database ...")
    sys.stdout.flush()

    star_id_to_cut = []
    for star_ID in list(encounter_db.keys()):
        if len(encounter_db[star_ID]) == 0:
            star_id_to_cut.append(star_ID)
    print(len(star_id_to_cut))
    for star_ID in sorted(star_id_to_cut, reverse=True):
        del encounter_db[star_ID]

    # Perform Cut & Advancement on Systems to Lower Integration Time
    for star_ID in list(encounter_db.keys()):
        enc_id_to_cut = []
        for enc_id, encounter in enumerate(encounter_db[star_ID]):
            PeriastronCut = CutOrAdvance(encounter, star_ID, kepler_workers=KeplerWorkerList)
            if PeriastronCut != None:
                encounter_db[star_ID][enc_id] = PeriastronCut
            elif PeriastronCut == None:
                enc_id_to_cut.append(enc_id)
        for enc_id in sorted(enc_id_to_cut, reverse=True):
            del encounter_db[star_ID][enc_id]

    star_id_to_cut = []
    for star_ID in list(encounter_db.keys()):
        if len(encounter_db[star_ID]) == 0:
            star_id_to_cut.append(star_ID)
    print("Star IDs to Cut:", star_id_to_cut)
    for star_ID in sorted(star_id_to_cut, reverse=True):
        del encounter_db[star_ID]
    print(encounter_db.keys())
    encounter_cut_file = open(os.getcwd()+"/"+cluster_name+"_encounters_cut.pkl", "wb")
    pickle.dump(encounter_db, encounter_cut_file)
    encounter_cut_file.close()

    sys.stdout = orig_stdout
    log_file.close()

    for K in KeplerWorkerList:
        K.stop()
    print("Finished cutting encounter database.")
