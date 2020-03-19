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

num_stars = 10
num_psys = 0
cluster_name = "workingAsIntendedTest"
restart_file = cluster_name+"_restart"
write_file = restart_file

def write_initial_state(master_set, ic_array, file_prefix):
    ''' Writes out an initial state for the Tycho Module.
        master_set: The Master Amuse Particle Set used in Tycho
        ic_array: Predefined Numpy Array that Stores Initial Conditions in SI Units
        file_prefix: String Value for a Prefix to the Saved File
    '''    
# First, Define/Make the Directory for the Initial State to be Stored
    file_dir = os.getcwd()+"/InitialState"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_base = file_dir+"/"+file_prefix
# Second, Write the AMUSE Particle Set to a HDF5 File
    file_format = "hdf5"
    write_set_to_file(master_set, file_base+"_particles.hdf5", format=file_format, close_file=True)
# Third, Pickle the Initial Conditions Array
    ic_file = open(file_base+"_ic.pkl", "wb")
    pickle.dump(ic_array, ic_file)
    ic_file.close()

def write_time_step(master_set, converter, current_time, file_prefix):
    ''' Writes out necessary information for a time step.
        master_set: The Master AMUSE Particle Set used in Tycho
        multiples_code: The Multiples Instance for Tycho
        current_time: The Simulations Current Time
        file_prefix: String Value for a Prefix to the Saved File
    '''
# First, Define/Make the Directory for the Time Step to be Stored
    file_dir_MS = os.getcwd()+"/Run/MasterParticleSet"
    file_dir_CoM = os.getcwd()+"/Run/CoMSet"
    if not os.path.exists(file_dir_MS):
        os.makedirs(file_dir_MS)
    if not os.path.exists(file_dir_CoM):
        os.makedirs(file_dir_CoM)
    file_base_MS = file_dir_MS+"/"+file_prefix
    file_base_CoM = file_dir_CoM+"/"+file_prefix
# Second, Create the CoM Tree Particle Set from Multiples
# Third, Convert from NBody to SI Before Writing
    MS_SI = datamodel.ParticlesWithUnitsConverted(master_set, converter.as_converter_from_nbody_to_si())
#    CoM_SI = datamodel.ParticlesWithUnitsConverted(CoM_Set, converter.as_converter_from_nbody_to_si())
# Fourth, Write the Master AMUSE Particle Set to a HDF5 File
    file_format = "hdf5"
    write_set_to_file(MS_SI, file_base_MS+"_MS_t%.3f.hdf5" %(current_time.number), \
                      format=file_format, close_file=True)
# Fifth, Write the CoM Tree Particle Set to a HDF5 File
#    write_set_to_file(CoM_SI, file_base_CoM+"_CoM_t%.3f.hdf5" %(current_time.number), \
#                      format=file_format, close_file=True)

# ------------------------------------ #
#        WRITING  RESTART FILE         #
# ------------------------------------ #

def write_state_to_file(time, stars_python,gravity_code, multiples_code, write_file, cp_hist=False, backup = 0 ):
    print("Writing state to write file: ", write_file,"\n\n")
    if write_file is not None:
        particles = gravity_code.particles.copy()
        write_channel = gravity_code.particles.new_channel_to(particles)
        write_channel.copy_attribute("index_in_code", "id")
        bookkeeping = {'neighbor_veto': multiples_code.neighbor_veto,
            'neighbor_distance_factor': multiples_code.neighbor_distance_factor,
                'multiples_external_tidal_correction': multiples_code.multiples_external_tidal_correction,
                    'multiples_integration_energy_error': multiples_code.multiples_integration_energy_error,
                        'multiples_internal_tidal_correction': multiples_code.multiples_internal_tidal_correction,
                        'model_time': multiples_code.model_time,
                        'root_index': multiples.root_index
        }
        
        '''
            bookkeeping.neighbor_veto =
            bookkeeping.multiples_external_tidal_correction = multiples_code.multiples_external_tidal_correction
            bookkeeping.multiples_integration_energy_error = multiples_code.multiples_integration_energy_error
            bookkeeping.multiples_internal_tidal_correction = multiples_code.multiples_internal_tidal_correction
            bookkeeping.model_time = multiples_code.model_time
            '''
        for root, tree in multiples_code.root_to_tree.items():
            #multiples.print_multiple_simple(tree,kep)
            root_in_particles = root.as_particle_in_set(particles)
            subset = tree.get_tree_subset().copy()
            if root_in_particles is not None:
                root_in_particles.components = subset
        io.write_set_to_file(particles,write_file+".stars.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist)
        io.write_set_to_file(stars_python,write_file+".stars_python.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist)
        config = {'time' : time,
            'py_seed': pickle.dumps(random.getstate()),
                'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                    'options': pickle.dumps(options)
        }
        with open(write_file + ".conf", "wb") as f:
            pickle.dump(config, f)
        with open(write_file + ".bookkeeping", "wb") as f:
            pickle.dump(bookkeeping, f)
        print("\nState successfully written to:  ", write_file)
        print(time)
        if backup > 0:
            io.write_set_to_file(particles,write_file+".backup.stars.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            io.write_set_to_file(stars_python,write_file+".backup.stars_python.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            config2 = {'time' : time,
                'py_seed': pickle.dumps(random.getstate()),
                    'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                        'options': pickle.dumps(options)
            }
            with open(write_file + ".backup.conf", "wb") as f:
                pickle.dump(config2, f)
                f.close()
            with open(write_file + ".backup.bookkeeping", "wb") as f:
                pickle.dump(bookkeeping, f)
                f.close()
            print("\nBackup write completed.\n")
        
        if backup > 2:
            io.write_set_to_file(particles,write_file+"."+str(int(time.number))+".stars.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            io.write_set_to_file(stars_python,write_file+"."+str(int(time.number))+".stars_python.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            config2 = {'time' : time,
                'py_seed': pickle.dumps(random.getstate()),
                    'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                        'options': pickle.dumps(options)
            }
            with open(write_file + "." +str(int(time.number))+".conf", "wb") as f:
                pickle.dump(config2, f)
                f.close()
            with open(write_file + "."+str(int(time.number))+".bookkeeping", "wb") as f:
                pickle.dump(bookkeeping, f)
                f.close()
            print("\nBackup write completed.\n")
            
def read_state_from_file(restart_file, gravity_code, kep, SMALLN):

    stars = read_set_from_file(restart_file+".stars.hdf5",'hdf5',version='2.0', close_file=True).copy()
#    single_stars = read_set_from_file(restart_file+".singles.hdf5",'hdf5',version='2.0')
#    multiple_stars = read_set_from_file(restart_file+".coms.hdf5",'hdf5',version='2.0')
    stars_python = read_set_from_file(restart_file+".stars_python.hdf5",'hdf5',version='2.0', close_file=True).copy()
    with open(restart_file + ".bookkeeping", "rb") as f:
        bookkeeping = pickle.load(f)
        f.close()
    print(bookkeeping)
    root_to_tree = {}
    for root in stars:
        if hasattr(root, 'components') and not root.components is None:
            root_to_tree[root] = datamodel.trees.BinaryTreeOnParticle(root.components[0])
    gravity_code.particles.add_particles(stars)
#    print bookkeeping['model_time']
#    gravity_code.set_begin_time = bookkeeping['model_time']


    multiples_code = multiples.Multiples(gravity_code, SMALLN, kep)
#    multiples_code.neighbor_distance_factor = 1.0
#    multiples_code.neighbor_veto = False
#    multiples_code.neighbor_distance_factor = 2.0
#    multiples_code.neighbor_veto = True
    multiples_code.neighbor_distance_factor = bookkeeping['neighbor_distance_factor']
    multiples_code.neighbor_veto = bookkeeping['neighbor_veto']
    multiples_code.multiples_external_tidal_correction = bookkeeping['multiples_external_tidal_correction']
    multiples_code.multiples_integration_energy_error = bookkeeping['multiples_integration_energy_error']
    multiples_code.multiples_internal_tidal_correction = bookkeeping['multiples_internal_tidal_correction']
    multiples.root_index = bookkeeping['root_index']
    multiples_code.root_to_tree = root_to_tree
#    multiples_code.set_model_time = bookkeeping['model_time']

    return stars_python, multiples_code
    
def king_cluster(num_stars, filename_cluster, w0=2.5, IBF=0.5, rand_seed=0):
    ''' Creates an open cluster according to the King Model & Kroupa IMF
        num_stars: The total number of stellar systems.
        seed: The random seed used for cluster generation.
        filename_cluster: The filename used when saving the cluster.
        w0: The King density parameter.	
        IBF: The Initial Binary Fraction.
    '''
# TODO: Have rand_seed actually do something ...

# Creates a List of Masses (in SI Units) Drawn from the Kroupa IMF
    masses_SI = new_kroupa_mass_distribution(num_stars)
# Creates the SI-NBody Converter
    converter = nbody_system.nbody_to_si(masses_SI.sum(), 1 | units.parsec)
# Creates a AMUS Particle Set Consisting of Positions (King) and Masses (Kroupa)
    stars_SI = new_king_model(num_stars, w0, convert_nbody=converter)
    stars_SI.mass = masses_SI
    print(stars_SI.mass.as_quantity_in(units.MSun))
# Assigning IDs to the Stars
    stars_SI.id = np.arange(num_stars) + 1
    stars_SI.type = "star"
# Moving Stars to Virial Equilibrium
    stars_SI.move_to_center()
    if num_stars == 1:
        pass
    else:
        stars_SI.scale_to_standard(convert_nbody=converter)
    if int(IBF) != 0:
# Creating a Temporary Particle Set to Store Binaries
        binaries=Particles()
# Selects the Indices of Stars that will be converted to Binaries
        num_binaries = int(IBF*num_stars)
        select_stars_indices_for_binaries = rp.sample(range(0, num_stars), num_binaries)
        select_stars_indices_for_binaries.sort()
        delete_star_indices = []
# Creates a decending list of Star Indicies to make deletion easier.
        for y in range(0, num_binaries):
            delete_star_indices.append(select_stars_indices_for_binaries[(num_binaries-1)-y])
# Creates the Binaries and assigns IDs
        for j in range(0, num_binaries):
            q = select_stars_indices_for_binaries[j]
            binaries.add_particles(binary_system(stars_SI[q], converter))
        binaries.id = np.arange(num_binaries*2) + 2000000
# Deletes the Stars that were converted to Binaries
        for k in range(0, num_binaries):
            b = delete_star_indices[k]
            stars_SI.remove_particle(stars_SI[b])
# Merges the Binaries into to the Master Particle set
        stars_SI.add_particle(binaries)
# Assigning SOI Estimate for Interaction Radius
    if num_stars == 1:
        stars_SI.radius = 1000 | units.AU
    else:
        stars_SI.radius = util.calc_SOI(stars_SI.mass, np.var(stars_SI.velocity), G=units.constants.G)
# Returns the Cluster & Converter
    return stars_SI, converter


def binary_system(old_star, converter):
    ''' Creates a Binary System from an Original Star, given an NBody-to-SI converter.
        old_star: The original star that is being replaced.
        converter: The AMUSE unit converter to keep scaling correct.
    '''
# Define Original Star's Information
    rcm = old_star.position
# Define Binary System Elements
# TO DO: Make this fit observations!
    r = 500 | units.AU
    r1 = 0.5*r
    r2 = -0.5*r
# Create a Temporary 
    binary = Particles(2)
    star1 = binary[0]
    star2 = binary[1]
    star1.type = 'star'
    star2.type = 'star'
    star1.mass = 0.5*old_star.mass
    star2.mass = 0.5*old_star.mass
    star1.radius = old_star.radius/2.0
    star2.radius = old_star.radius/2.0
    star1.position = [r1.number,0.0,0.0] | units.AU #nbody_system.length
    star2.position = [r2.number,0.0,0.0] | units.AU
# Calculate the Orbit Velocities for the Pair
    vel1 = np.sqrt(units.constants.G*star2.mass*r1/r**2)
    vel2 =-1*np.sqrt(units.constants.G*star1.mass*(np.absolute(r2))/r**2)
# Define New Velocities
    star1.vx = old_star.vx
    star1.vy = vel1 + old_star.vy
    star1.vz = old_star.vz
    star2.vx = old_star.vx
    star2.vy = vel2 + old_star.vy
    star2.vz = old_star.vz
# Preform the Euler Rotation and Replace the Binary
    util.preform_EulerRotation(binary)
    star1.position = star1.position+rcm
    star2.position = star2.position+rcm
    return binary


def planetary_systems(stars, converter, num_systems, filename_planets, Earth=False, Jupiter=False, Neptune=False):
    ''' Creates several mock planetary systems around random stars in the provided set.
        stars: The AMUSE Particle Set containing stellar information.
        num_systems: The number of planetary systems requested.
        filename_planets: Filename for the Initial Planetary System HDF5 Archive.
        Earth, Jupiter, Neptune: Booleans asking if they should be included.
    '''
# Sets Initial Parameters
    num_stars = len(stars)
    select_stars_indices = rp.sample(range(0, num_stars), num_systems)
    i = 0
    ID_Earth = 3000000
    ID_Jupiter = 5000000
    ID_Neptune = 8000000
    systems = datamodel.Particles()
# Begins to Build Planetary Systems According to Provided Information
    for system in range(num_systems):
        planets = datamodel.Particles()
        j = select_stars_indices[system]
        if Earth:
            init_a = 1.000 | units.AU
            init_e = 0.016
            mass_E = 0.003 | units.MJupiter
            planets.add_particle(planet(ID_Earth+system, stars[j], mass_E, init_a, init_e))
        if Jupiter:
            init_a = 5.454 | units.AU
            init_e = 0.048
            mass_J = 1 | units.MJupiter
            planets.add_particle(planet(ID_Jupiter+system, stars[j], mass_J, init_a, init_e))
        if Neptune:
            init_a = 30.110 | units.AU
            init_e = 0.009
            mass_N = 0.054 | units.MJupiter
            planets.add_particle(planet(ID_Neptune+system, stars[j], mass_N, init_a, init_e))
    # Moves Planetary System to the Origin and Applies a Random Euler Rotation
        for p in planets:
            p.position = p.position - stars[j].position
            p.velocity = p.velocity - stars[j].velocity
        util.preform_EulerRotation(planets)
        for p in planets:
            p.position = p.position + stars[j].position
            p.velocity = p.velocity + stars[j].velocity
    # Adds the System to the Provided AMUSE Particle Set
        systems.add_particles(planets)
    return systems
    

def planet(ID, host_star, planet_mass, init_a, init_e, random_orientation=False):
    ''' Creates a planet as an AMUSE Particle with provided characteristics.
        ID: Identifying number unique to this planet.
        host_star: The AMUSE Particle that is the host star for the planet.
        planet_mass: The mass of the planet (in the nbody units).
        init_a: Initial semi-major axis (in nbody units).
        init_e: Initial eccentricity (in nbody units).
        random_orientation: Boolean to incline the planet in a random fashion.
    '''
# Sets Planet Values to Provided Conditions
    p = datamodel.Particle()
    p.id = ID
    p.type = "planet"
    p.host_star = host_star.id
    p.mass = planet_mass
# Calculates a Random Position in the Orbit & Gets the Orbital Speed
    cosTheta = math.cos(np.random.random()*math.pi*2)
    init_x = init_a*(1-init_e**2)/(1+init_e*cosTheta) # Need to Double Check ... Thornton 302
    init_vy = np.sqrt(units.constants.G*host_star.mass*(2/init_x-1/init_a))
# Sets the Dynamical Radius to the Hill Sphere Approx.
    p.radius = util.calc_HillRadius(init_a, init_e, p.mass, host_star.mass)
# Sets the Particle in "Orbit" Around the Origin
# May Need to Correct This Later to Accurately Depict Position in Orbit
    p.position = [init_x.number, 0.0 , 0.0] | init_x.unit
    p.velocity = [0.0 , init_vy.number, 0.0] | init_vy.unit
# Preforms a Euler Rotation on the Planet if Requested
# Not Working Properly due to Data Structure, Need to Fix
    if random_orientation:
        temp = datamodel.Particles().add_particle(p)
        util.preform_EulerRotation(temp)
        p = temp[0]
# Moves the Planet to Orbit the Host Star
    p.position = p.position + host_star.position
    p.velocity = p.velocity + host_star.velocity
# Returns the Created AMUSE Particle
    return p    
    
    
    
MasterSet = datamodel.Particles()
# Create the Stellar Cluster, Shift from SI to NBody, Add the Particles to MS, & Open a Channel to the MS
stars_SI, converter = king_cluster(num_stars, 'test_stars', rand_seed=0, IBF=0.5)
stars_NB = datamodel.ParticlesWithUnitsConverted(stars_SI, converter.as_converter_from_nbody_to_si())
MasterSet.add_particles(stars_NB)
#channel_stars_master = stars_NB.new_channel_to(MasterSet)
# Create Planetary Systems, Shift from SI to NBody, Add the Particles to MS, & Open a Channel to the MS
systems_SI = planetary_systems(stars_SI, converter, num_psys, 'test_planets', Jupiter=True)
systems_NB = datamodel.ParticlesWithUnitsConverted(systems_SI, converter.as_converter_from_nbody_to_si())
MasterSet.add_particles(systems_NB)

print(MasterSet)

time = 0.0 | nbody_system.time
delta_t = 0.05 | nbody_system.time
number_of_steps = 10
end_time = number_of_steps*delta_t
num_workers = 1
eps2 = 0.0 | nbody_system.length**2
use_gpu = 1
gpu_ID = -1

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
multiples_code = multiples.Multiples(gravity, util.new_smalln, kep)
multiples_code.neighbor_distance_factor = 1.0
multiples_code.neighbor_veto = True



multiples_code.evolve_model(time)
gravity.synchronize_model()
# Copy values from the module to the set in memory.
grav_channel.copy()

# Copy the index (ID) as used in the module to the id field in
# memory.  The index is not copied by default, as different
# codes may have different indices for the same particle and
# we don't want to overwrite silently.
grav_channel.copy_attribute("index_in_code", "id")



write.write_state_to_file(time, MasterSet, gravity, multiples_code, write_file)


gravity.stop()
kep.stop()
util.stop_smalln()

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

MasterSet2 = []
MasterSet2, multiples_code = read.read_state_from_file(restart_file, gravity, kep, util.new_smalln)

if np.array_equal(MasterSet, MasterSet2):
    print("It worked!")
print("Finished!")

print(MasterSet)
print(MasterSet2)
