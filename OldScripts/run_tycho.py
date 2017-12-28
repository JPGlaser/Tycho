# -*- coding: utf-8 -*-

# Planetary Systems within Clusters via AMUSE's Multiples & PH4 Packages

# TODO:
# 
#  - Add in stellar evolution.
#  - Create Additional Options (H.Z., elliptical orbits, etc) in add_planets()
#  - Test Multi-Planet systems in Multiples.

# ---------------------------
# Importing Necessary Packages
# ----------------------------

# Import Required Python Base Packages
import sys
import unittest
import numpy as np
import random
import collections
import os
import time as tp
from optparse import OptionParser
import glob
import math
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a Safety Mechanism for Not Having MatPlotLib Installed
#try:
    #%matplotlib inline
    #from matplotlib.pyplot import *
    #from mpl_toolkits.mplot3d import Axes3D
    #HAS_MATPLOTLIB = True
    
    #matplotlib.rcParams['figure.figsize'] = (16, 6)
    #matplotlib.rcParams['font.size'] = (14)
#except ImportError:
    #HAS_MATPLOTLIB = False

# Import cPickle/Pickle
try:
   import cPickle as pickle
except:
   import pickle

# Import the Decimal Package
from decimal import Decimal

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
from amuse.ic.salpeter import new_salpeter_mass_distribution_nbody

# Import the Amuse Gravity & Close-Encounter Packages
from amuse.community.ph4.interface import ph4 as grav
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.couple import multiples

# ----------------------
# Defining New Functions
# ----------------------

# Function to Create a King's Model Cluster
def Create_King_Cluster(number_of_stars, number_of_binaries, w, write_file=True, cluster_name=None):
    """
    This is a function created to create a King's Model Cluster and write the HDF5 File if requested.
    The Function Returns: The Set of Stars, AMUSE Nbody Converter, Cluster Name, & King's Model W0 Value
    """
    # Creates a List of Masses Drawn from the Salpeter Mass Distribution & Prints the Total Mass
    salpeter_masses = new_salpeter_mass_distribution(number_of_stars, mass_max = (15 | units.MSun))
    total_mass = salpeter_masses.sum()
    print "Total Mass of the Cluster:", total_mass.value_in(units.MSun), "Solar Masses"

    # This converter is set so that [1 mass = Cluster's Total Mass] & [1 length = 1 parsec].
    converter = nbody_system.nbody_to_si(total_mass, 1 | units.parsec)

    # Creates a AMUSE Particle Set Consisting of Stars from the King Model & Sets Their Masses
    stars = new_king_model(number_of_stars, w)
    stars.mass = converter.to_nbody(salpeter_masses)
    #stars.radius = stars.mass.number | nbody_system.length
    stars.radius = converter.to_nbody(1000 | units.AU)
    stars.id = np.arange(number_of_stars) + 1

    # Moving Stars to Virial Equilibrium
    stars.move_to_center()
    if number_of_stars == 1:
        pass
    else:
        stars.scale_to_standard()
    
    # Creates the AMUSE Paricle set for the Binaries
    binaries = Particles()
	
	# Selects the Indices of Stars that will be converted to Binaries
    select_stars_indices_for_binaries = random.sample(xrange(0, number_of_stars), number_of_binaries)
    select_stars_indices_for_binaries.sort()
    delete_star_indices = []
	
	# Creates a decending list of Star Indicies to make deletion easier.
    for y in xrange(0, number_of_binaries):
		delete_star_indices.append(select_stars_indices_for_binaries[(number_of_binaries-1)-y])

	# Creates the Binaries and assigns IDs
    for j in xrange(0, number_of_binaries):
		q = select_stars_indices_for_binaries[j]
		binaries.add_particles(makeBinary(stars[q], converter))
    binaries.id = np.arange(number_of_binaries*2) + 2000000
    
    # Deletes the Stars that were converted to Binaries
    for k in xrange(0, number_of_binaries):
		b = delete_star_indices[k]
		stars.remove_particle(stars[b])
	
	# Adds in the Binaries to the Master Particle set
    stars.add_particle(binaries)
		
    # Assigns a Name to the Cluster if Not Provided
    if cluster_name == None:
        cluster_name = "OC"+tp.strftime("%y%m%d-%H%M%S", tp.gmtime())
    
    # Optionally Write the Cluster to a File
    if write_file:
        file_format = "hdf5"
        # Generates a File Name Given the Cluster Name
        file_name = "kingcluster_%s_w%.2f_N%i_M%.6f.%s" %(cluster_name, w, number_of_stars, total_mass.value_in(units.MSun), file_format)
        # Sets the File Path and Writes the Particle Set to a hdf5 File
        file_dir = os.getcwd()+"/Clusters"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = file_dir+"/"+file_name
        write_set_to_file(stars, file_path, format=file_format, close_file=True)
    else:
        pass
    
    # Return the Particle Set of Stars and the Unit Converter
    return stars, converter, cluster_name, w

def Import_Cluster(file_path)
    """
    This is a function created to read in a Cluster's Stellar Parcticle Set.
    The Function Returns: The Set of Stars, AMUSE Nbody Converter, Cluster Name, & King's Model W0 Value
    """
    # Parses Variables from File_Name
    file_name = (file_path.split("/")[-1])[:-5]
    file_format = (file_path.split("/")[-1])[-4:]
    cluster_name = file_name.split("_")[-4]
    w = float((file_name.split("_")[-3])[1:])
    N = int((file_name.split("_")[-2])[1:])
    total_mass = float((file_name.split("_")[-1])[1:])
    
    # Reads in the Set of Stars from the Specified File Path
    stars = read_set_from_file(file_path, format=file_format, close_file=True)
    number_of_stars = len(stars)
    
    # This converter is set so that [1 mass = Cluster's Total Mass] & [1 length = 1 parsec].
    converter = nbody_system.nbody_to_si(total_mass, 1 | units.parsec)
    return stars, converter, cluster_name, w

def add_planets(stars, number_of_planets, converter, cluster_name, Neptune=True, Double=False, write_file=True):
    """
    This is a function created to add planets to a set of stars in a cluster.
    The Function Returns: Nothing. It appends the planets to provided dataset.
    """
    # Creates the empty AMUSE Particle Set for the Planets
    planets = datamodel.Particles(number_of_planets)
    
    # Creates the List of Star Indices that will Host Plantes
    number_of_stars = len(stars)
    select_stars_indices = random.sample(xrange(0, number_of_stars), number_of_planets)
    i = 0
    
    # Creates One Jupiter-Like Planet Around the Each Star in the Subset
    if Neptune:
        for planet  in planets:
            j = select_stars_indices[i]
            planet.id = 5000000 + i + 1
            planet.host_star = stars[j].id
            planet.mass = converter.to_nbody(0.05 | units.MJupiter)
            planet.radius = planet.mass.number | nbody_system.length
            initial_orbit_radius = converter.to_nbody(30 | units.AU)
            planet.x = stars[j].x + initial_orbit_radius
            planet.y = stars[j].y
            planet.z = stars[j].z
            initial_orbit_speed = np.sqrt(stars[j].mass*nbody_system.G/initial_orbit_radius)
            planet.vx = stars[j].vx
            planet.vy = stars[j].vy + initial_orbit_speed
            planet.vz = stars[j].vz
            eulerAngle(planet, stars[j])
            i+=1
        stars.add_particles(planets)
        
    # Creates two Jupiters on Opposite Sides of Each Star in the Subset
    if Double:
        for planet in planets[:number_of_planets/2]:
            j = select_stars_indices[i]
            planet.id = 5000000 + i + 1
            planet.host_star = stars[j].id
            planet.mass = converter.to_nbody(1.0 | units.MJupiter)
            planet.radius = planet.mass.number | nbody_system.length
            initial_orbit_radius = converter.to_nbody(2 | units.AU)
            planet.x = stars[j].x + initial_orbit_radius
            planet.y = stars[j].y
            planet.z = stars[j].z
            initial_orbit_speed = np.sqrt(stars[j].mass*nbody_system.G/initial_orbit_radius)
            planet.vx = stars[j].vx
            planet.vy = stars[j].vy + initial_orbit_speed
            planet.vz = stars[j].vz
            i+=1
        i = 0
        for planet in planets[number_of_planets/2:]:
            j = select_stars_indices[i]
            planet.id = 5000000 + number_of_planets/2+i + 1
            planet.host_star = stars[j].id
            planet.mass = converter.to_nbody(0.053953 | units.MJupiter)
            planet.radius = planet.mass.number | nbody_system.length
            initial_orbit_radius = converter.to_nbody(30 |  units.AU)
            planet.x = stars[j].x - initial_orbit_radius
            planet.y = stars[j].y
            planet.z = stars[j].z
            initial_orbit_speed = np.sqrt(stars[j].mass*nbody_system.G/initial_orbit_radius)
            planet.vx = stars[j].vx
            planet.vy = stars[j].vy - initial_orbit_speed
            planet.vz = stars[j].vz
            i+=1
        stars.add_particles(planets)
    print planets

SMALLN = None
# Creates a New SmallN Instance by Resetting the Previous
def new_smalln():
    SMALLN.reset()
    return SMALLN

# Initalizes a SmallN Instance
def init_smalln():
    global SMALLN
    SMALLN = SmallN(redirection="none")
    SMALLN.parameters.timestep_parameter = 0.05

# Creates the Multiples CoM Particle Set with Children Information
def get_multiples_set(Multiples_Grav):
    """
    This is a function created to create a particle set of CoM Particles.
    The Function Returns: AMUSE Particle set of CoM & Children Particles.
    """
    MULTIPLES_Set = datamodel.Particles()
    for root, tree in Multiples_Grav.root_to_tree.iteritems():
        multi_systems = tree.get_tree_subset().copy()
        MULTIPLES_Set.add_particle(multi_systems)
    return MULTIPLES_Set

# Finds the Hoststar of a Planet by searching through the Multiples Tree
def get_hoststar(planet, MULTIPLES_Set, Python_Set):
	"""
	This is a function created to retrieve the host star given a planet id.
	The Function Returns: The AMUSE particle for the host star.
	"""
	for x in MULTIPLES_Set:
		try:
			if (x.child1.id == planet.id or x.child2.id == planet.id):
				parent = x
				if (parent.child1.id < 5000000):
					parent = parent.child1
				elif (parent.child2.id < 5000000):
					parent = parent.child2
				elif (parent.child1.id == 0):
					parent = parent.child1
				elif (parent.child2.id == 0):
					parent = parent.child2
				while parent.id == 0:
					if parent.child1.id < 5000000:
						print parent.child1
						parent = parent.child1
					elif parent.child2.id < 5000000:
						print parent.child2
						parent = parent.child2
				hoststar = next((x for x in Python_Set if (x.id == parent.id)), None)
				return hoststar
		except:
			pass

# Alternate Method to Find the Host Star | NOT FINISHED
def get_hoststar2(planet, Python_Set):
	hoststar = Python_Set.find_closest_particle_to(planet.x, planet.y, planet.z) 
	return hoststar

# Creates an AMUSE Particle Set Containing all of the Releveant Planetary System Information
def get_planet_systems(MULTIPLES_Set, Python_Set, number_of_planets, grav_kep):
	"""
    This is a function created to make a particle set of planets and their hoststars.
    The Function Returns: The AMUSE particle set containing all planets and their hoststars.
    """
	planets = Python_Set.sorted_by_attribute('id')[-number_of_planets:].copy()
	stars = Python_Set.sorted_by_attribute('id')[:-number_of_planets].copy()
	systems = datamodel.Particles()
	for planet in planets:
		hoststar = get_hoststar(planet, MULTIPLES_Set, Python_Set)
		planet.hoststar = hoststar
		if hoststar != None:
			systems.add_particle(hoststar)
			TMass = hoststar.mass + planet.mass
        		rel_pos = planet.position - hoststar.position
        		rel_vel = planet.velocity - hoststar.velocity
        		kep.initialize_from_dyn(TMass, rel_pos[0], rel_pos[1], rel_pos[2], \
						rel_vel[0], rel_vel[1], rel_vel[2])
			planet.TMass = TMass
        		planet.a, planet.e = kep.get_elements()
        		planet.sep = kep.get_separation()
       		 	planet.T = kep.get_period()
                        # Calculating Angular Momentum Vector, h = r x v
			h = np.cross(rel_pos, rel_vel)
			# Calculating the Inclination in Radians
			# https://en.wikibooks.org/wiki/Astrodynamics/Classical_Orbit_Elements#Inclination_.28.29
       		 	planet.I = np.arccos(h[2]/np.sqrt(h.dot(h)))
			# Calculating the Longitude of the Ascending Node
			n = np.cross(np.array([0,0,1]), h)
			planet.LoAN = np.arccos(n[0]/np.sqrt(n.dot(n)))
			# Calculating the Argument of the Perihelion
			mu = nbody.G*TMass
			E = np.array((np.cross(rel_vel,h)[0]/mu, np.cross(rel_vel,h)[1]/mu, \
				np.cross(rel_vel,h)[2]/mu))-rel_pos/np.sqrt(rel_pos.dot(rel_pos))
			planet.AoP = np.arccos(n.dot(E)/(np.sqrt(n.dot(n))*np.sqrt(E.dot(E))))
		systems.add_particle(planet)
		print systems
	return systems

# Function that Writes a Particle set in the appropriate Directory.
def write_set(Particle_Set, run_subdir, file_prefix, current_time):
    """
    This is a function created to write a particle set to a file with a default format.
    The Function Returns: Writes an HDF5 file with all particle information.
    """
    # Sets the File Format
    file_format = "hdf5"
    # Sets the File Path and Writes the Particle Set to a hdf5 File
    file_dir = "%s/Runs/%s" %(os.getcwd(), run_subdir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    # Generates a File Name Given the Cluster Name
    file_name = "%s_t%.3f.%s" %(file_prefix, current_time.number, file_format)
    file_path = file_dir+"/"+file_name
    # Writes the Master Set to a File
    print '-------------'
    write_set_to_file(Particle_Set, file_path, version='2.0', \
                      append_to_file=False, format=file_format)
    print "[UPDATE] The Main Particle Set and the MULTIPLES Set are Stored!"
    print '-------------'

def append_op_step(op_array, planets_set, converter, step_index, timestep):
    """
    This is a function created to append orbital parameters for planets into a numpy array.
    This will allow storage and graphing of time-series data to be significantly easier.
    """
    for i in xrange(len(op_array)):
        (op_array[i].t)[step_index] = (step_index+1.0)*(converter.to_si(timestep).value_in(units.Myr))
        (op_array[i].TMass)[step_index] = converter.to_si(planets_set[i].TMass).value_in(units.MSun)
        if planets_set[i].hoststar != None:
             (op_array[i].hsID)[step_index] = planets_set[i].hoststar.id
        else:
             (op_array[i].hsID)[step_index] = -1
        (op_array[i].a)[step_index] = converter.to_si(planets_set[i].a).value_in(units.AU)
        (op_array[i].e)[step_index] = planets_set[i].e
        (op_array[i].sep)[step_index] = converter.to_si(planets_set[i].sep).value_in(units.AU)
        (op_array[i].Per)[step_index] = converter.to_si(planets_set[i].T).value_in(units.day)
        (op_array[i].I)[step_index] = planets_set[i].I
	(op_array[i].LoAN)[step_index] = planets_set[i].LoAN
	(op_array[i].AoP)[step_index] = planets_set[i].AoP

#	this function returns a euler matrix to use in setting the inclination of a planet
def eulerAngle(planet, hoststar):
    """
    This is a function returns an euler roatation matrix used to set the inclination of a planet.
    This Function Returns: Position and Velocity particle attributes for the planet. 
    """
	# Get the three Random Angles (Uniform Distribution)	
	angle1 = np.random.random()*math.pi*2
	angle2 = np.random.random()*math.pi*2
	angle3 = np.random.random()*math.pi*2

	# Calculate the Rotation Matrix Elements
	cosz=math.cos(angle1)
	cosy=math.cos(angle2)
	cosx=math.cos(angle3)
	sinz=math.sin(angle1)
	siny=math.sin(angle2)
	sinx=math.sin(angle3)

	# Moves the Planet to the Center of Mass Position
	planet.position = planet.position - hoststar.position
	planet.velocity = planet.velocity - hoststar.velocity
	
	# First Roatation: About the Z-Axis
	z_rotation = np.matrix([[cosz,-sinz,0], 
							[sinz,cosz,0], 
							[0,0,1]])

	# Second Rotation: About the Y-Axis					
	y_rotation = np.matrix([[cosy,0,siny],
							[0,1,0],
							[-siny,0,cosy]])

	# Third Rotation: About the X-Axis
	x_rotation = np.matrix([[1,0,0],
							[0,cosx,-sinx],
							[0,sinx,cosx]])

	# Preform the Matrix Multiplication
	rotate = np.dot(y_rotation, z_rotation)
	rotate = np.dot(x_rotation, rotate)
	
	# Apply the Rotation Matrix to the Planet Position and Velocity
	planetpos = np.matrix(([[planet.x.number],
							[planet.y.number],
							[planet.z.number]]))
	planetvel = np.matrix(([[planet.vx.number],
							[planet.vy.number],
							[planet.vz.number]]))				
	rotationpos = np.dot(rotate, planetpos)
	rotationvel = np.dot(rotate, planetvel)
	
	# Shift the planet back to its proper position.
	planet.x = rotationpos[0] + hoststar.x.number | nbody_system.length
	planet.y = rotationpos[1] + hoststar.y.number | nbody_system.length
	planet.z = rotationpos[2] + hoststar.z.number | nbody_system.length
	
	# Shifts the planet back to its proper velocity
	planet.vx = rotationvel[0] + hoststar.vx.number | nbody_system.length / nbody_system.time
	planet.vy = rotationvel[1] + hoststar.vy.number | nbody_system.length / nbody_system.time
	planet.vz = rotationvel[2] + hoststar.vz.number | nbody_system.length / nbody_system.time
	# Returns the Position and Velocity Elements 
	return planet.position, planet.velocity
	
	
def makeBinary(originalStar, converter):
	originalVelocity = originalStar.velocity
	rcm = originalStar.position
	mass = originalStar.mass/2
#	r will depend on the nbody_converter
	r = converter.to_nbody(500 | units.AU)
	r1 = 0.5*r
	r2 = -0.5*r
	
	binary = Particles(2)
	star1 = binary[0]
	star2 = binary[1]
	
	star1.mass = mass
	star2.mass = mass
	
	star1.radius = originalStar.radius
	star2.radius = originalStar.radius
	
	star1.position = [r1.number,0.0,0.0] | nbody_system.length
	star2.position = [r2.number,0.0,0.0] | nbody_system.length

	vel1 = np.sqrt(nbody_system.G*mass*r1/r**2)
	vel2 =-1*np.sqrt(nbody_system.G*mass*(np.absolute(r2))/r**2)
	
	star1.vx = originalVelocity[0]
	star1.vy = vel1 + originalVelocity[1]
	star1.vz = originalVelocity[2]

	star2.vx = originalVelocity[0]
	star2.vy = vel2 + originalVelocity[1]
	star2.vz = originalVelocity[2]

#	execute the rotation	
	angle1 = np.random.random()*math.pi*2
	angle2 = np.random.random()*math.pi*2
	angle3 = np.random.random()*math.pi*2
	
	cosz=math.cos(angle1)
	cosy=math.cos(angle2)
	cosx=math.cos(angle3)

	sinz=math.sin(angle1)
	siny=math.sin(angle2)
	sinx=math.sin(angle3)
	
#	the first rotation about the z-axis
	z_rotation = np.matrix([[cosz,-sinz,0], 
							[sinz,cosz,0], 
							[0,0,1]])

#	the second rotation about the y-axis					
	y_rotation = np.matrix([[cosy,0,siny],
							[0,1,0],
							[-siny,0,cosy]])

#	the final rotation about the x-axis
	x_rotation = np.matrix([[1,0,0],
							[0,cosx,-sinx],
							[0,sinx,cosx]])

	# Put values into the translation matrix
	rotate = np.dot(y_rotation, z_rotation)
	rotate = np.dot(x_rotation, rotate)	
	
	star1pos = np.matrix(([[star1.x.number],
							[star1.y.number],
							[star1.z.number]]))
			
	star2pos = np.matrix(([[star2.x.number],
							[star2.y.number],
							[star2.z.number]]))
							
	star1vel = np.matrix(([[star1.vx.number],
							[star1.vy.number],
							[star1.vz.number]]))
							
	star2vel = np.matrix(([[star2.vx.number],
							[star2.vy.number],
							[star2.vz.number]]))
							
	rotationpos1 = np.dot(rotate, star1pos) | nbody_system.length
	rotationvel1 = np.dot(rotate, star1vel)  | nbody_system.length / nbody_system.time
	
	rotationpos2 = np.dot(rotate, star2pos) | nbody_system.length
	rotationvel2 = np.dot(rotate, star2vel)  | nbody_system.length / nbody_system.time
	
	star1.x = rotationpos1[0] + rcm[0]
	star1.y = rotationpos1[1] + rcm[1]
	star1.z = rotationpos1[2] + rcm[2]
	
	star2.x = rotationpos2[0] + rcm[0]
	star2.y = rotationpos2[1] + rcm[1]
	star2.z = rotationpos2[2] + rcm[2]
	
	star1.vx = rotationvel1[0]
	star1.vy = rotationvel1[1]  
	star1.vz = rotationvel1[2]  
	
	star2.vx = rotationvel2[0]  
	star2.vy = rotationvel2[1]  
	star2.vz = rotationvel2[2]  

	return binary
	

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

# Creating Command Line Argument Parser
parser = OptionParser()
parser.add_option("-g", "--no-gpu", dest="use_gpu",default=1, type="int", \
                      help="Disable GPU for computation by setting to a value other than 1.")
parser.add_option("-i", "--gpu-id", dest="gpu_ID", default= -1, type="int", \
                      help="Select which GPU to use by device ID")
parser.add_option("-p", "--num-planets", dest="num_planets", default=32, type="int", \
		      help="Enter the number of planets desired")
parser.add_option("-s", "--num-stars", dest="num_stars", default=750, type="int", \
		      help="Enter the number of stars desired")
parser.add_option("-t", "--timestep", dest="dt", default=0.05, type="float", \
		      help="Enter the PH4 timestep in N-Body Units")
parser.add_option("-c", "--cluster-name", dest="cluster_name", default=None, type="str", \
		      help="Enter the name of the Cluster (Defaults to Numerical Naming Scheme)")
parser.add_option("-w", "--w0", dest="w0", default=2.5, type="float", \
		      help="Enter the w0 parameter for the King's Model")
parser.add_option("-N", "--num-steps", dest="num_steps", default=1000, type="int", \
		      help="Enter the total number of time-steps to take")
parser.add_option("-b", "--num-binaries", dest="num_binaries", default = 375, type ="int", \
			  help = "Enter the number of binaries in the cluster")
(options, args) = parser.parse_args()

# Setting Cluster Parameters 
number_of_stars = options.num_stars
w0 = options.w0
cluster_name = options.cluster_name
number_of_binaries = options.num_binaries

# Getting the Full Path to the Cluster File (If the File Exists)
search_path = "%s/Clusters/kingcluster_%s_w%.2f_N%i_*.hdf5" %(os.getcwd, cluster_name, w0, number_of_stars)
try:
    full_path = glob.glob(search_path)[0]
except:
    full_path = ""

# Loading if the Cluster Already Exists or Creating a New Cluster
if not os.path.isfile(full_path):
    objects, kc_converter, cluster_name, w0 = Create_King_Cluster(number_of_stars, number_of_binaries, w0, write_file=True, cluster_name=cluster_name)
else:
    objects, kc_converter, cluster_name, w0 = Import_Cluster(full_path)

# Adding Planets to the Cluster
number_of_planets = options.num_planets
#add_planets(objects, number_of_planets, kc_converter, cluster_name, Jupiter=False, Double=True)
add_planets(objects, number_of_planets, kc_converter, cluster_name, Neptune=True, Double=False)

# Defining Initial Conditions for PH4
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
	try:
		gravity = grav(number_of_workers = num_workers, redirection = "none", mode = "gpu")
	except Exception as ex:
		gravity = grav(number_of_workers = num_workers, redirection = "none")
        print "*** GPU worker code not found. Reverting to non-GPU code. ***"
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
gravity.particles.add_particles(objects)
gravity.commit_particles()

# Starting the AMUSE Channel for PH4
channel = gravity.particles.new_channel_to(objects)

# Initializing Kepler and SmallN
kep = Kepler(None, redirection = "none")
kep.initialize_code()
init_smalln()

# Initializing MULTIPLES
multiples_code = multiples.Multiples(gravity, new_smalln, kep)
multiples_code.neighbor_distance_factor = 1.0
multiples_code.neighbor_veto = True

# Alerts the Terminal User that the Run has Started!
print '[UPDATE] Run Started at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))

# Creates the Log File and Redirects all Print Statements
orig_stdout = sys.stdout
log_dir = "%s/Runs/%s/%s" %(os.getcwd(), cluster_name, tp.strftime("%y%m%d", tp.gmtime()))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
f = file(log_dir+"/output.log", 'w')
sys.stdout = f
print '[UPDATE] Run Started at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime()))
print '-------------'
sys.stdout.flush()

# Initialize Easy Oribital Parameter Time-Series Storage
step_index = 0
op_dtype = np.dtype({'names': ['t','TMass','hsID','a','e','sep','Per','I', 'LoAN', 'AoP'], \
					 'formats': ['f8', 'f8', 'i4', 'f8','f8','f8','f8','f8','f8','f8']})
planets_op = [np.recarray((number_of_steps+1,), dtype=op_dtype) for i in xrange(number_of_planets)]

# Sets the "Runs" Directory
runs_dir = "%s/Runs" %(os.getcwd())
runs_current_dir = "/%s/%s" %(cluster_name, tp.strftime("%y%m%d", tp.gmtime()))

# Begin Evolving the Model!
while time < end_time:
    time += delta_t
    multiples_code.evolve_model(time)
    gravity.synchronize_model()

	# Update the bookkeeping: synchronize stars with the module data.
    #try:
        #gravity.update_particle_set()
        #gravity.particles.synchronize_to(objects)
    #except:
        #pass
    
    # Copy values from the module to the set in memory.
    channel.copy()
    
    # Copy the index (ID) as used in the module to the id field in
    # memory.  The index is not copied by default, as different
    # codes may have different indices for the same particle and
    # we don't want to overwrite silently.
    channel.copy_attribute("index_in_code", "id")
    
    # Creating the Output Sets
    Python_Set = multiples_code.stars
    MULTIPLES_Set = get_multiples_set(multiples_code)
    PSystems_Set = get_planet_systems(MULTIPLES_Set, Python_Set, number_of_planets, kep)
    Planets_Set = PSystems_Set.sorted_by_attribute('id')[-number_of_planets:].copy()

    # Appends Orbital Evolution Data to Planets_OP Array
    append_op_step(planets_op, Planets_Set, kc_converter, step_index, delta_t)

    # Writes the Orbital Evolution Data to a Pickle File
    op_pickle_file = open(runs_dir+runs_current_dir+"/orbital_evolution.pkl", "wb")
    pickle.dump(planets_op, op_pickle_file)
    op_pickle_file.close()
    print '-------------'
    print '[UPDATE] Orbital Parameters Pickled for Analysis Later!'
    print '-------------'

    # Writes the Particle Set Containing Planetary System Information
    run_subdir_psystem = runs_current_dir+"/PSystem"
    file_prefix_psystem = "%s_PSystem" %(cluster_name)
    write_set(PSystems_Set, run_subdir_psystem, file_prefix_psystem, time)

    # Writes the Master Particle Set Every 0.25 Myr
    if step_index%5 == 0:
	run_subdir_master = runs_current_dir+"/Master"
	file_prefix_master = "%s_Master" %(cluster_name)
        write_set(Python_Set, run_subdir_master, file_prefix_master, time)
    
    # Increase the Step_Index
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
