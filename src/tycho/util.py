# Python Classes/Functions containing Utility Functions for Tycho
# Keep This Class Unitless!

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import sys, os, math
import numpy as np
import matplotlib as plt
import time as tp
import random as rp
import hashlib

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
from amuse.community.seba.interface import SeBa

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def new_seed_from_string(string):
    ''' Creates a seed for Numpy.RandomState() usin a string.
        string: The provided string to use.
    '''
    hash_md5 = hashlib.md5(str(string)).hexdigest()
    hash_int = ""
    for c in hash_md5:
        if c.isalpha():
            hash_int += str(ord(c))
        else:
            hash_int += c
    seed = int(hash_int) % (2**32 -1)
    return seed

def store_ic(converter, options):
    ''' Creates a Structured Numpy Array to Store Initial Conditions.
        converter: AMUSE NBody Converter Used in Tycho.
        options: Commandline Options Set by User.
    '''
    ic_dtype = np.dtype({'names': ['cluster_name','seed','num_stars','num_planets','total_smass','viral_radius','w0','IBF'], \
					     'formats': ['S8', 'S8', 'i8', 'i8','f8','f8','f8','f4']})
    ic_array = np.recarray(1, dtype=ic_dtype)
    ic_array[0].cluster_name = options.cluster_name
    ic_array[0].seed = options.seed
    ic_array[0].num_stars = options.num_stars
    ic_array[0].num_planets = options.num_psys
    tsm = converter.to_si(converter.values[1]).number
    vr = converter.to_si(converter.values[2]).number
    ic_array[0].total_smass = tsm
    ic_array[0].viral_radius = vr
    ic_array[0].w0 = options.w0
    #ic_array[0].IBF = options.IBF
    return ic_array[0]

def preform_EulerRotation(particle_set):
    ''' Preforms a randomly oriented Euler Transformation to a set of AMUSE Particles.
        particle_set: AMUSE particle set which it will preform the transform on.

        !! Based on James Arvo's 1996 "Fast Random Rotation Matrices"
        !! https://pdfs.semanticscholar.org/04f3/beeee1ce89b9adf17a6fabde1221a328dbad.pdf  
    '''
# First: Generate the three Uniformly Distributed Numbers (Two Angles, One Decimal)
    n_1 = np.random.uniform(0.0, math.pi*2.0)
    n_2 = np.random.uniform(0.0, math.pi*2.0)
    n_3 = np.random.uniform(0.0, 1.0)
# Second: Calculate Matrix & Vector Values
    c1 = np.cos(n_1)
    c2 = np.cos(n_2)
    s1 = np.sin(n_1)
    s2 = np.cos(n_2)
    r3 = np.sqrt(n_3)
    R = [[  c1,  s1, 0.0],
         [ -s1,  c1, 0.0],
         [ 0.0, 0.0, 1.0]]
    V = [[c2*r3],
         [s2*r3],
         [np.sqrt(1-n_3)]]
# Third: Create the Rotation Matrix
    rotate = (np.outer(V, V) - np.dot(np.eye(3),(R)))	
# Forth: Preform the Rotation & Update the Particle
    for particle in particle_set:
        pos = np.matrix(([[particle.x.number], [particle.y.number], [particle.z.number]]))
        vel = np.matrix(([[particle.vx.number], [particle.vy.number], [particle.vz.number]]))
        particle.position = np.dot(rotate,pos) | particle.position.unit  # nbody_system.length
        particle.velocity = np.dot(rotate,vel)  | particle.velocity.unit


def calc_HillRadius(a, e, m_planet, m_star):
    ''' Calculates the Hill Radius for a planet given a, e, and the two masses.
        a: The semi-major axis of the planet's orbit.
        e: The eccentricity of the planet's orbit.
        m_planet: The mass of the planet.
        m_star: The mass of the star.
    '''
    return a*(1.0-e)*(m_planet/(3*m_star))**(1.5)


def calc_SOI(m_star, vel_var, G):
    ''' Approximates the Sphere of Influence for a singlular stellar object.
        m_star: The mass of the stellar object.
        vel_var: The velocity dispursion of the system.

        !! https://en.wikipedia.org/wiki/Sphere_of_influence_(black_hole)
    '''
    return G*m_star/vel_var


SMALLN = None
# Creates a New SmallN Instance by Resetting the Previous
def new_smalln():
    SMALLN.reset()
    return SMALLN

# Initalizes a SmallN Instance
def init_smalln(unit_converter = None):
    global SMALLN
    if unit_converter is None:
        SMALLN = SmallN(redirection="none")
    else:
        SMALLN = SmallN(redirection="none", convert_nbody=unit_converter)
    SMALLN.parameters.timestep_parameter = 0.05

def stop_smalln():
    SMALLN.stop()

# Take a particle set and a time, and perfrom stellar evolution on the particle set until that time
def StellarEvolution(star_list, age):
    se = SeBa()
    # Looping through the particle set for asynchronous stellar evolution
    for star in star_list:
        # Immediately check that the particle is a star
       	# Pass if it is NOT a star, using SE on a non-star will error
        if star.type!="star":
            pass
        # A few options here:
        # 1 figure out how often the code 'loses connection' when evolving 10k stars
        #   and then reset the SE every x stars
        # 2 try: evolve stars, except: when it crashes reset SE and goto try
        # 

        else:
            end_time = 0
            end_time= age+star.age
            # Add the single star to the chosen SE code
            se.particles.add_particle(star)
            # Attempt to evolve the star to a stored age value
            # If no value is stored evolve the star to 8 Myrs
            '''
            try:
                # Use the star's stored age if it is greater than 0 otherwise evolve 8 years
                if star.age > 0 |units.yr: age = star.age
                else: age = 8 | units.Myr
            except:
                print "No stored age attribute for star id: "+str(star.id)
                age = 8 | units.Myr
            '''

            # This part handles the actually stellar evolution
            if end_time > 0 | units.Myr:
                # not sure about using se.evolve_model or individually evolving parts
                #se.evolve_model(end_time)
                star, se = singleStarEvolution(star, end_time, se)

    # Return the modified Particle set    
    return star_list

# The reason we separate the actual evolution into a separate class is to use a try except method on restarting SeBa
# If you don't restart SeBa often enough the code crashes, but if you restart too often the overhead becomes very high
# There could possibly be a better way to manage this but I was unaware of a better way.
def singleStarEvolution(star, end_time, se):
    # First attempt to evolve the star regularly
    try:
        star.mass = se.evolve_star(star.mass,end_time,0.02)[1]
        star.luminosity = se.evolve_star(star.mass,end_time,0.02)[3]
        star.physical_radius = se.evolve_star(star.mass,end_time,0.02)[2]
        star.age = se.evolve_star(star.mass,end_time,0.02)[0]
        star.temperature = se.evolve_star(star.mass,end_time,0.02)[4]

    # If that fails for some reason reset the stellar evolution
    except:
	se.stop()
        se = SeBa()
        star.mass = se.evolve_star(star.mass,end_time,0.02)[1]
        star.luminosity = se.evolve_star(star.mass,end_time,0.02)[3]
        star.physical_radius = se.evolve_star(star.mass,end_time,0.02)[2]
        star.age = se.evolve_star(star.mass,end_time,0.02)[0]
        star.temperature = se.evolve_star(star.mass,end_time,0.02)[4]
    # return the star and stellar evolution code
    return star, se
