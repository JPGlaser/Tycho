# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Add a as a read in from the restart and crash function for Tyler's code!

# Importing Necessary System Packages
import sys, os, math
import numpy as np
import matplotlib as plt
import time as tp
import random as rp
from optparse import OptionParser
import glob

# Tyler's imports
import hashlib
import copy
import traceback
import signal

from time import gmtime
from time import mktime
from time import clock

from collections import defaultdict

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
from amuse.community.seba.interface import SeBa
from amuse.couple.bridge import Bridge
#from amuse.couple import multiples

# Import the Tycho Packages
from tycho import create, util, read, write, encounter_db
from tycho import multiples2 as multiples

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

def print_diagnostics(grav, E0=None):

    # Simple diagnostics.

    ke = grav.kinetic_energy
    pe = grav.potential_energy
    Nmul, Nbin, Emul = grav.get_total_multiple_energy()
    print ''
    print 'Time =', grav.get_time().in_(units.Myr)
    print '    top-level kinetic energy =', ke
    print '    top-level potential energy =', pe
    print '    total top-level energy =', ke + pe
    print '   ', Nmul, 'multiples,', 'total energy =', Emul
    E = ke + pe + Emul
    print '    uncorrected total energy =', E
    
    # Apply known corrections.
    
    Etid = grav.multiples_external_tidal_correction \
            + grav.multiples_internal_tidal_correction  # tidal error
    Eerr = grav.multiples_integration_energy_error	# integration error

    E -= Etid + Eerr
    print '    corrected total energy =', E

    if E0 is not None: print '    relative energy error=', (E-E0)/E0
    
    return E

# Tyler uses a start time variable for his naming

global Starting

Starting = mktime(gmtime())

if __name__=="__main__":

    start_time = tp.time()

# Creating Command Line Argument Parser
    parser = OptionParser()
    parser.add_option("-g", "--no-gpu", dest="use_gpu",default=1, type="int", \
                      help="Disable GPU for computation by setting to a value other than 1.")
    parser.add_option("-i", "--gpu-id", dest="gpu_ID", default= -1, type="int", \
                      help="Select which GPU to use by device ID.")
    parser.add_option("-p", "--num-psys", dest="num_psys", default=32, type="int", \
                      help="Enter the number of planetary systems desired.")
    parser.add_option("-s", "--num-stars", dest="num_stars", default=750, type="int", \
                      help="Enter the number of stars desired.")
    parser.add_option("-t", "--timestep", dest="dt", default=0.05, type="float", \
                      help="Enter the PH4 timestep in N-Body Units.")
    parser.add_option("-c", "--cluster-name", dest="cluster_name", default=None, type="str", \
                      help="Enter the name of the Cluster (Defaults to Numerical Naming Scheme).")
    parser.add_option("-w", "--w0", dest="w0", default=2.5, type="float", \
                      help="Enter the w0 parameter for the King's Model.")
    parser.add_option("-N", "--num-steps", dest="num_steps", default=1000, type="int", \
                      help="Enter the total number of time-steps to take.")
    parser.add_option("-b", "--IBF", dest="IBF", default = 0.5, type ="float", \
       		      help = "Enter the initial binary fraction.")
    parser.add_option("-S", "--seed", dest="seed", default = "1234", type="str", \
                      help = "Enter a random seed for the simulation")
    parser.add_option("-R","--restart",dest="restart_file",default="_restart", type="str", \
                      help = "Enter the name for the restart_file, (Defaults to _restart_")
    parser.add_option("-D", "--database", dest="database", default="cluster_db", type="str")
    parser.add_option("-r", "--doRestart", dest="doRestart", action="store_true")
    parser.add_option("-P", "--pregen-flag", dest="pregen", default=0, type ="int", \
		              help = "If set to 1 will attempt to load an amuse file containing a cluster")

    (options, args) = parser.parse_args()

# Set Commonly Used Python Variables from Options
    num_stars = options.num_stars
    num_psys = options.num_psys
    cluster_name = options.cluster_name
    restart_end = options.restart_file
    restart_file = "Restart/"+cluster_name+"_time_"
    write_file_base = restart_file
    database = options.database
    crash_base = "CrashSave/"+cluster_name
    pregen = options.pregen
    
# Eventually Change this to find the pregen file
    pregen_file = "/home/draco/jthornton/Tycho/PregenClusters/DracoM3V02AnewSFplt0975.amuse"

# Set up loading in a pregeerated cluster from a gas cloud simulation
# TODO FIX THIS.
    if pregen == 1:
        files = glob.glob("PregenClusters/"+".amuse")
        print files
        pregen_cluster_load = files

# If Pregenerated Cluster IS NOT Wanted, Try to Import the Cluster from File or Create a New One
    if pregen != 1:
        try:
            MasterSet, ic_array, LargeScaleConverter = read.read_initial_state(cluster_name)
        # TODO: Create the "stars" and "planets" sets and open up their channels to the master set.
        #       It should match the style designed below.
        # Set the Boolean Check for Reading a File to True 
            read_from_file = True
        except:
        # Initilize the Master Particle Set
            MasterSet = datamodel.Particles()
        # Create the Stellar Cluster in SI Units & Create the Large-Scale Converter
            stars_SI, LargeScaleConverter = create.king_cluster(num_stars, num_binaries = 
                                                    int(num_stars*options.IBF), seed=options.seed)
            MasterSet.add_particles(stars_SI)
            channel_stars_master = stars_SI.new_channel_to(MasterSet)
        # Create the Planetary Systems in SU Units
            systems_SI = create.planetary_systems(stars_SI, num_psys, 'test_planets', Jupiter=True)
            MasterSet.add_particles(systems_SI)
            channel_planets_master = systems_SI.new_channel_to(MasterSet)
        # Set the Boolean Check for Reading a File to False         
            read_from_file = False
        # Create Initial Conditions Array
            initial_conditions = util.store_ic(LargeScaleConverter, options)
# If Pregenerated Cluster IS Wanted, Load the Pregenerated Cluster.
    if pregen == 1:
        MasterSet = read_set_from_file(pregen_file, format='amuse')
        MasterSet.type = "star"
        num_stars = len(MasterSet)
        MasterSet.id = np.arange(num_stars) + 1
        read_from_file = False
        num_stars = len(MasterSet)
        MStars = sum(MasterSet.mass.number) | units.MSun
        MasterSet.radius = 5000*MasterSet.mass/(1.0 | units.MSun) | units.AU
        virial_radius = MasterSet.virial_radius()
        LargeScaleConverter = nbody_system.nbody_to_si(MStars, virial_radius)
        initial_conditions = util.store_ic(LargeScaleConverter, options)
    # Add Planets to the Pregenerated Cluster
        systems_SI = create.planetary_systems(MasterSet, num_psys, 'test_planets', Jupiter=True)
        MasterSet.add_particles(systems_SI)
        channel_planets_master = systems_SI.new_channel_to(MasterSet)

# Attempt to Restart a Crash if it Exists in the CrashSave Directory
    try:
        search = glob.glob("CrashSave/"+cluster_name+"*.hdf5")[-1]
        crash_file = search[:-18]
        crash = True
    except:
        crash = False

# Write the Initial State if Using New IC
    if not read_from_file:
        write.write_initial_state(MasterSet, initial_conditions, cluster_name)

# Define PH4-Related Initial Conditions
    time = 0.0 | units.Myr
    delta_t = 0.002 | units.Myr
    number_of_steps = options.num_steps
    end_time = number_of_steps*delta_t
    num_workers = 1
    eps2 = 1 | units.AU**2
    use_gpu = options.use_gpu
    gpu_ID = options.gpu_ID

# Setting PH4 as the Top-Level Gravity Code
    if use_gpu == 1:
        gravity = ph4(number_of_workers = num_workers, redirection = "none", mode = "gpu", 
                      convert_nbody=LargeScaleConverter)
    else:
        gravity = ph4(number_of_workers = num_workers, redirection = "none",
                      convert_nbody=LargeScaleConverter)

# Initializing PH4 with Initial Conditions
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.begin_time = time
    gravity.parameters.epsilon_squared = eps2
    gravity.parameters.timestSep_parameter = options.dt

# Setting up the Code to Run with GPUs Provided by Command Line
    gravity.parameters.use_gpu = use_gpu
    gravity.parameters.gpu_id = gpu_ID

# Setting Up the Stopping Conditions in PH4
    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()
    sys.stdout.flush()

# Adding and Committing Particles to PH4
    gravity.particles.add_particles(MasterSet)
    gravity.commit_particles()
    #print gravity.particles[-5:]

# Starting the AMUSE Channel for PH4
    grav_to_MS_channel = gravity.particles.new_channel_to(MasterSet)
    MS_to_grav_channel = MasterSet.new_channel_to(gravity.particles, 
                            attributes=["x","y","z","vx","vy","vz"])

    SmallScaleConverter = nbody_system.nbody_to_si(2*np.mean(MasterSet.mass), 2*np.mean(MasterSet.radius))
# Initializing Kepler and SmallN
    kep = Kepler(unit_converter=SmallScaleConverter, redirection = "none")
    kep.initialize_code()
    util.init_smalln(unit_converter=SmallScaleConverter)

# Initializing MULTIPLES, Testing to See if a Crash Exists First
    if read_from_file and crash:
        time, multiples_code = read.recover_crash(crash_file, gravity, kep, util.new_smalln)
    else:
        multiples_code = multiples.Multiples(gravity, util.new_smalln, kep, 
                                             gravity_constant=units.constants.G)
        multiples_code.neighbor_perturbation_limit = 0.05
        #multiples_code.neighbor_distance_factor = 1.0
        multiples_code.neighbor_veto = True

# Initializing Stellar Evolution (SeBa)
    sev_code = SeBa()
    sev_code.particles.add_particles(MasterSet)

# Starting the AMUSE Channel for Stellar Evolution
    sev_to_MS_channel = sev_code.particles.new_channel_to(MasterSet, 
                            attributes=["mass", "luminosity", "stellar_type", "temperature", "age"])

# Initializing Galactic Background
    Rgal=1. | units.kpc
    Mgal=1.6e10 | units.MSun
    alpha=1.2
    galactic_code = create.GalacticCenterGravityCode(Rgal, Mgal, alpha)

# Move the Cluster into a 1 kpc Orbit
    rinit_from_galaxy_core = 5.0 | units.kpc
    galactic_code.move_particles_into_ellipictal_orbit(MasterSet, rinit_from_galaxy_core)
    MS_to_grav_channel.copy()
    #print gravity.particles.x[0].in_(units.kpc)
    #print MasterSet.x[0].in_(units.kpc)

# Initializing the Bridge
    bridge_code = Bridge(verbose=False)
    bridge_code.add_system(multiples_code, (galactic_code,))
    #bridge_code.add_system(gravity, (galactic_code,))
    #bridge_code.add_system(multiples_code, ())
    bridge_code.add_system(sev_code, (multiples_code,))

# Stating the AMUSE Channel for Bridge to Have SeBa and Multiples Interact
    bridge_code.channels.add_channel(sev_code.particles.new_channel_to(multiples_code.particles, 
                                     attributes=["mass"]))
    bridge_code.timestep = delta_t

# Alerts the Terminal User that the Run has Started!
    print '\n [UPDATE] Run Started at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    print '-------------'
    sys.stdout.flush()
    E0 = print_diagnostics(multiples_code)

# Creates the Log File and Redirects all Print Statements
    orig_stdout = sys.stdout
    log_dir = os.getcwd()+"/Logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    f = file(log_dir+"/%s_%s.log" %(cluster_name, tp.strftime("%y%m%d", tp.gmtime())), 'w')
    sys.stdout = f
    
    step_index = 0

# Make the Encounters directory to store encounter information in
    enc_dir = "Encounters"
    if not os.path.exists(enc_dir):
        os.makedirs(enc_dir)

# So far this puts one empty list, encounters, into the dictionary
# The particle list is made in the EncounterHandler class
# The particle list will contain all the information about the particles involved in an encounter
    encounterInformation = defaultdict(list)
    for particle in MasterSet:
        if particle.type == "star":
            key = str(particle.id)
            encounterInformation[key] = []

    class EncounterHandler(object):
        def handle_encounter_v5(self, time, star1, star2):
        # Synchronize Gravity for Safety
            #gravity.synchronize_model()

        # Create the Scattering CoM Particle Set
            scattering_com = Particles(particles = (star1, star2))
            com_pos = scattering_com.center_of_mass()
            com_vel = scattering_com.center_of_mass_velocity()
            
        # Expand enconter returns a particle set with all of the children 
        # when given a particle set of two objects involved in an encounter
            enc_particles = multiples_code.expand_encounter(scattering_com, delete=False)[0]
            
        # Assign the time of the encounter to the Encounter Particle Set.
            enc_particles.time = time
        
        # Set the Origin to be the Center of Mass for the Encounter Particle Set.
            enc_particles.position -= com_pos
            enc_particles.velocity -= com_vel
            
       # Retrieve Star IDs to Use as Dictionary Keys, and Loop Over Those IDs
       # to Add Encounter Information to Each Star's Dictionary Entry.
            for star_id in [str(key) for key in enc_particles.id if key<=len(MasterSet)]:
                encounterInformation[star_id].append(enc_particles)

       # Return True is Necessary for the Multiples Code
            return True

# Setting Up the Encounter Handler
    encounters = EncounterHandler()
    multiples_code.callback = encounters.handle_encounter_v5

# Variable used for saving the dictionary at resets
    encounter_file = None
    
# Copy values from the module to the set in memory.
    grav_to_MS_channel.copy()
    
# Send the Initial Stellear Mass to init_mass Attribute of the Master Set
# This is for FRESCO Compatibility as SEBA is Odd about Ages ...
    MasterSet.init_mass = MasterSet.mass
    
# Copies over the SEV Desired Stellar Traits to the Master Set
    sev_code.evolve_model(1 | units.yr)
    sev_to_MS_channel.copy()
    bridge_code.channels.copy()

# Begin Evolving the Cluster
    while time < end_time:
        sys.stdout.flush()
    # Kick the Gravity Bridge
        time += delta_t
        bridge_code.evolve_model(time)
        util.update_MasterSet(gravity, multiples_code, sev_code)
    # Copy the index (ID) as used in the module to the id field in
    # memory.  The index is not copied by default, as different
    # codes may have different indices for the same particle and
    # we don't want to overwrite silently.
        grav_to_MS_channel.copy_attribute("index_in_code", "id")

    # Copy values from the module to the set in memory.
        grav_to_MS_channel.copy()
    # Copies over the SEV Desired Stellar Traits to the Master Set
        sev_to_MS_channel.copy()

    # Write Out the Data Every 5 Time Steps
        if step_index%5 == 0:
            write.write_time_step(MasterSet, time, cluster_name)

    # Write out a crash file every 50 steps
        #if step_index%50 == 0:
        #    step = str(time.number)
        #    crash_file = crash_base+"_t"+step
        #    write.write_crash_save(time, MasterSet, gravity, multiples_code, crash_file)
            
    # Write out the restart file and restart from it every 10 time steps
        doRestart = False
        if step_index%10 == 0: #CHANGE LATER
            if doRestart:
                # TODO: Need to figure out how this works with the new Bridge.
                reset_flag=1
            # Log that a Reset happened		
                print '\n-------------'
                print '[UPDATE] Reset at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
                print '-------------\n'
            sys.stdout.flush()

        # Save the Encounters Dictionary Thus Far (If Not the First Reset)
            if step_index != 0:
                if encounter_file != None:
                # First Try to Delete the Previous Backup File
                # Its in a try because there will not be a backup the first time this runs
                # Could replace try loop with a counter if desired
                    try:
                        os.remove("Encounters/"+cluster_name+"_encounters_backup.pkl")
                    except:
                        pass
                # Then Rename the Previous Encounter Dictionary
                    os.rename("Encounters/"+cluster_name+"_encounters.pkl", 
                              "Encounters/"+cluster_name+"_encounters_backup.pkl")		
            # Finally, Save the Encounter Dictionary!
                encounter_file = None		
                encounter_file = open("Encounters/"+cluster_name+"_encounters.pkl", "wb")
                pickle.dump(encounterInformation, encounter_file)		
                encounter_file.close()
            # Log that a the Encounters have been Saved!		
                print '\n-------------'
                print '[UPDATE] Encounters Saved at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
                print '-------------\n'
                sys.stdout.flush()

    # Increase the Step Counter
        step_index += 1
    # Log that a Step was Taken
        print '\n-------------'
        print '[UPDATE] Step Taken at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
        print '-------------\n'
        sys.stdout.flush()
    
# Log that the Simulation Ended & Switch to Terminal Output
    print '\n[UPDATE] Run Finished at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    sys.stdout = orig_stdout
    f.close()

# Pickle the Final Encounter Information Dictionary
    encounter_file = open("Encounters/"+cluster_name+"_encounters.pkl", "wb")
    pickle.dump(encounterInformation, encounter_file)
    encounter_file.close()

# Alerts the Terminal User that the Run has Ended!
    print '\n[UPDATE] Run Finished at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    sys.stdout.flush()
    print_diagnostics(multiples_code, E0)

# Closes PH4, Kepler & SmallN Instances
    sev_code.stop()
    gravity.stop()
    kep.stop()
    util.stop_smalln()
    bridge_code.stop()
