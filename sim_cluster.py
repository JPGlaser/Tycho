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
from amuse.community.sse.interface import SSE
from amuse.couple import multiples

# Import the Tycho Packages
from tycho import create, util, read, write, encounter_db

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

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
    parser.add_option("-S", "--seed", dest="seed", default = 1234, type="int", \
                      help = "Enter a random seed for the simulation")
    parser.add_option("-R","--restart",dest="restart_file",default="_restart", type="str", \
                      help = "Enter the name for the restart_file, (Defaults to _restart.hdf5")
    parser.add_option("-D", "--database", dest="database", default="cluster_db", type="str")

    (options, args) = parser.parse_args()

# Set Commonly Used Python Variables from Options
    num_stars = options.num_stars
    num_psys = options.num_psys
    cluster_name = options.cluster_name
    restart_file = "Restart/"+cluster_name+options.restart_file
    write_file_base = restart_file
    database = options.database
    crash_base = "CrashSave/"+cluster_name

# Try to Import the Cluster from File or Create a New One
    try:
        MasterSet, ic_array, converter = read.read_initial_state(cluster_name)
    # TODO: Create the "stars" and "planets" sets and open up their channels to the master set.
    #       It should match the style designed below.
        read_from_file = True
    except:
    # Initilize the Master Particle Set
        MasterSet = datamodel.Particles()
    # Create the Stellar Cluster, Shift from SI to NBody, Add the Particles to MS, & Open a Channel to the MS
        stars_SI, converter = create.king_cluster(num_stars, num_binaries=int(num_stars*options.IBF), seed=options.seed)
        stars_NB = datamodel.ParticlesWithUnitsConverted(stars_SI, converter.as_converter_from_nbody_to_si())
        MasterSet.add_particles(stars_NB)
        #channel_stars_master = stars_NB.new_channel_to(MasterSet)
    # Create Planetary Systems, Shift from SI to NBody, Add the Particles to MS, & Open a Channel to the MS
        systems_SI = create.planetary_systems(stars_SI, converter, num_psys, 'test_planets', Jupiter=True)
        systems_NB = datamodel.ParticlesWithUnitsConverted(systems_SI, converter.as_converter_from_nbody_to_si())
        MasterSet.add_particles(systems_NB)
        #channel_planets_master = systems_NB.new_channel_to(MasterSet)
        read_from_file = False

    # Create Initial Conditions Array
        initial_conditions = util.store_ic(converter, options)

    try:
        search = glob.glob("CrashSave/"+cluster_name+"*.hdf5")[-1]
        crash_file = search[:-18]
        crash = True
    except:
        crash = False



# Write the Initial State 
    if not read_from_file:
        write.write_initial_state(MasterSet, initial_conditions, cluster_name)

# Define PH4-Related Initial Conditions
    time = 0.0 | nbody_system.time
    delta_t = options.dt | nbody_system.time
    number_of_steps = options.num_steps
    end_time = number_of_steps*delta_t
    num_workers = 1
    eps2 = 0.0 | nbody_system.length**2
    use_gpu = options.use_gpu
    gpu_ID = options.gpu_ID

    # Setting PH4 as the Top-Level Gravity Code
    if use_gpu == 1:
        gravity = ph4(number_of_workers = num_workers, redirection = "none", mode = "gpu")
    #try:
        #gravity = ph4(number_of_workers = num_workers, redirection = "none", mode = "gp$
    #except Exception as ex:
    #    gravity = ph4(number_of_workers = num_workers, redirection = "none")
    #    print "*** GPU worker code not found. Reverting to non-GPU code. ***"
    else:
        gravity = grav(number_of_workers = num_workers, redirection = "none")

# Initializing PH4 with Initial Conditions
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.begin_time = time
    gravity.parameters.epsilon_squared = eps2
    gravity.parameters.timestep_parameter = delta_t.number

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

# Starting the AMUSE Channel for PH4
    grav_channel = gravity.particles.new_channel_to(MasterSet)

# Initializing Kepler and SmallN
    kep = Kepler(None, redirection = "none")
    kep.initialize_code()
    util.init_smalln()

# Initializing MULTIPLES
    if read_from_file and crash:
        time, multiples_code = read.recover_crash(crash_file, gravity, kep, util.new_smalln)
    else:
        multiples_code = multiples.Multiples(gravity, util.new_smalln, kep)
        multiples_code.neighbor_distance_factor = 1.0
        multiples_code.neighbor_veto = True

# Alerts the Terminal User that the Run has Started!
    print '\n [UPDATE] Run Started at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    print '-------------'
    sys.stdout.flush()
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

# This is where I added Tyler's stuff

# So far this puts one empty list, encounters, into the dictionary
# The particle list is made in the EncounterHandler class
# The particle list will contain all the information about the particles involved in an encounter

    encounterInformation = defaultdict(list)

    for star in MasterSet:
        if star.type == "star":
            name = str(star.id)
            encounterInformation[name] = []



    class EncounterHandler(object):
        def __init__(self):
            self.run_id = hashlib.sha256(str(Starting)).hexdigest()

        # I need to change this from writing into Tyler's database files to writing the infomation into structured arrays

        def handle_encounter_v2(self, kepler, conv, time, star1, star2):

            encounters1 = []
            encounters2 = []
            particles1 = []
            particles2 = []
            name1 = str(star1.id)
            name2 = str(star2.id)

            M,a,e,r,E,t = multiples.get_component_binary_elements(star1, star2, kepler)
            peri = abs(conv.to_si(a * (1.0 - e)).value_in(units.AU))
            apo = abs(conv.to_si(a * (1.0 + e)).value_in(units.AU))

            r_au = conv.to_si(r).value_in(units.AU)
            real_time = conv.to_si(time).value_in(units.Myr)

            # OrbitParams returns the peri and apo and convets all of the parameters to SI
            orbit = encounter_db.OrbitParams.from_nbody_params(M, a, e, r, E, time, conv)

            # from_particle returns the id, mass, radius, position, and velocity in SI
            star1_params = encounter_db.EncounterBody.from_particle(star1, conv)
            star2_params = encounter_db.EncounterBody.from_particle(star2, conv)

            '''
            Encounter returns: 
            
            [<Encounter None @ t=1.25509037737 
            peri=1.60014590665e+15 AU, r_init=3.44462289279e+15 m, ecc=3.32880142942
	    Body 82: <Body 82: mass=1.57566086421e+30 kg>
            Body 1: <Body 1: mass=1.50990530471e+29 kg>
            '''

            encounter = encounter_db.Encounter([star1_params, star2_params], orbit, conv.to_si(time))
        
            # Here I need to add all of this information into the structured arrays

            # -------- #
            #   FLAG   #
            # -------- #

            # First I put everything into the two particles list. 
            
            particles1.append(encounter)
            particles1.append(star1_params)
            particles1.append(star2_params)

            particles2.append(encounter)
            particles2.append(star2_params)
            particles2.append(star1_params)

            encounters1.append(particles1)
            encounters2.append(particles2)

            encounterInformation[name1].append(encounters1)
            encounterInformation[name2].append(encounters2)

            return True

#    cluster_params = encounter_db.ClusterParameters(num_stars, end_time)
#    db_writer = encounter_db.EncounterDbWriter(database, cluster_params)
    encounters = EncounterHandler()

# Begin Evolving the Cluster
    while time < end_time:
        sys.stdout.flush()
        time += delta_t
        
        def encounter_callback(time, s1, s2):
            return encounters.handle_encounter_v2(kep, converter, time, s1, s2)
        

        multiples_code.evolve_model(time, callback=encounter_callback)
        #multiples_code.evolve_model(time)
        gravity.synchronize_model()

    # Copy values from the module to the set in memory.
        grav_channel.copy()
    
    # Copy the index (ID) as used in the module to the id field in
    # memory.  The index is not copied by default, as different
    # codes may have different indices for the same particle and
    # we don't want to overwrite silently.
        grav_channel.copy_attribute("index_in_code", "id")

    # Write Out the Data Every 5 Time Steps
        if step_index%5 == 0:
            #CoMSet = datamodel.Particles()
            #for root, tree in multiples_code.root_to_tree.iteritems():
            #    multi_systems = tree.get_tree_subset().copy_to_new_particles()
            #    CoMSet.add_particle(multi_systems)
            write.write_time_step(MasterSet, converter, time, cluster_name)

    # Write out a crash file every 50 steps
        if step_index%50 == 0:
            step = str(time.number)
            crash_file = crash_base+"_t"+step
 	    write.write_crash_save(time, MasterSet, gravity, multiples_code, crash_file)
            
    # Write out the restart file and restart from it every 10 time steps
        if step_index%10 == 0:
            step = str(time.number)
            write_file=write_file_base+step
            write.write_state_to_file(time, MasterSet, gravity, multiples_code, write_file)
            gravity.stop()
            kep.stop()
            util.stop_smalln()
            restart_file=write_file

        # Setting PH4 as the Top-Level Gravity Code
            if use_gpu == 1:
                gravity = ph4(number_of_workers = num_workers, redirection = "none", mode = "gpu")
            else:
                gravity = grav(number_of_workers = num_workers, redirection = "none")


# Initializing PH4 with Initial Conditions
            gravity.initialize_code()
            gravity.parameters.set_defaults()
            gravity.parameters.begin_time = time
            gravity.parameters.epsilon_squared = eps2
            gravity.parameters.timestep_parameter = delta_t.number

# Setting up the Code to Run with GPUs Provided by Command Line
            gravity.parameters.use_gpu = use_gpu
            gravity.parameters.gpu_id = gpu_ID

# Initializing Kepler and SmallN
            kep = Kepler(None, redirection = "none")
            kep.initialize_code()
            util.init_smalln()

            MasterSet = []
            MasterSet, multiples_code = read.read_state_from_file(restart_file, gravity, kep, util.new_smalln)
            write_file = ""
            restart_file = ""

# Setting Up the Stopping Conditions in PH4
            stopping_condition = gravity.stopping_conditions.collision_detection
            stopping_condition.enable()
            sys.stdout.flush()

# Starting the AMUSE Channel for PH4
            grav_channel = gravity.particles.new_channel_to(MasterSet)


            print '\n [UPDATE] Reset at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
            print '-------------'
            sys.stdout.flush()

        step_index += 1

    # Log that a Step was Taken
        print '-------------'
        print '[UPDATE] Step Taken at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
        print '-------------'
        sys.stdout.flush()
    
# Log that the simulation Ended & Switch to Terminal Output
    print '[UPDATE] Run Finished at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    sys.stdout = orig_stdout
    f.close()

# Tyler's finalize code should produce the json files and cluster file we need added here
#    def finish(time, end_time, ex=None):
#        db_writer.finalize(time, end_time, wall_time=time.time()-start_time, ex=ex)

#    finish(time.number, end_time.number)

# Pickle the encounter information dictionary
    enc_dir = os.getcwd()+"/Encounters"
    if not os.path.exists(enc_dir):
        os.makedirs(enc_dir)
    encounter_file = open("Encounters/"+cluster_name+"_encounters.pkl", "wb")
    pickle.dump(encounterInformation, encounter_file)
    encounter_file.close()

# Alerts the Terminal User that the Run has Ended!
    print '[UPDATE] Run Finished at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    sys.stdout.flush()

# Closes PH4, Kepler & SmallN Instances
    gravity.stop()
    kep.stop()
    util.stop_smalln()    






