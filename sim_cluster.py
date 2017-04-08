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
from tycho import create, util, read, write 

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

if __name__=="__main__":

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
    parser.add_option("-S", "--seed", dest="seed", default = "tycho", type="str", \
                      help = "Enter a random seed for the simulation")
    (options, args) = parser.parse_args()

# Set Commonly Used Python Variables from Options
    num_stars = options.num_stars
    num_psys = options.num_psys
    cluster_name = options.cluster_name



# Try to Import the Cluster from File or Create a New One
    try:
        MasterSet, ic_array, converter = read.read_initial_state(options.cluster_name)
    # TODO: Create the "stars" and "planets" sets and open up their channels to the master set.
    #       It should match the style designed below.
        read_from_file = True
    except:
    # Initilize the Master Particle Set
        MasterSet = datamodel.Particles()
    # Create the Stellar Cluster, Shift from SI to NBody, Add the Particles to MS, & Open a Channel to the MS
        stars_SI, converter = create.king_cluster(num_stars, 'test_stars', rand_seed=options.seed, IBF=options.IBF)
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
		#    gravity = ph4(number_of_workers = num_workers, redirection = "none", mode = "gpu")
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
    print '\n [UPDATE] Run Started at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    print '-------------'
    sys.stdout.flush()

    step_index = 0

# Begin Evolving the Cluster
    while time < end_time:
        time += delta_t
        multiples_code.evolve_model(time)
        gravity.synchronize_model()
    # Copy values from the module to the set in memory.
        grav_channel.copy()
    
    # Copy the index (ID) as used in the module to the id field in
    # memory.  The index is not copied by default, as different
    # codes may have different indices for the same particle and
    # we don't want to overwrite silently.
        grav_channel.copy_attribute("index_in_code", "id")

    # Write Out the Data Every 5 Time Steps
        if step_index%1 == 0:
            #CoMSet = datamodel.Particles()
            #for root, tree in multiples_code.root_to_tree.iteritems():
            #    multi_systems = tree.get_tree_subset().copy_to_new_particles()
            #    CoMSet.add_particle(multi_systems)
            write.write_time_step(MasterSet, converter, time, cluster_name)
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

# Alerts the Terminal User that the Run has Ended!
    print '[UPDATE] Run Finished at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
    sys.stdout.flush()

# Closes PH4, Kepler & SmallN Instances
    gravity.stop()
    kep.stop()
    SMALLN.stop()    






