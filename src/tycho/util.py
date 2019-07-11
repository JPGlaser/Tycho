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

import time
import datetime

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

from amuse.ic.brokenimf import MultiplePartIMF

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
    s2 = np.sin(n_2)
    r3 = np.sqrt(n_3)
    R = [[  c1,  s1, 0.0],
         [ -s1,  c1, 0.0],
         [ 0.0, 0.0, 1.0]]
    V = [[c2*r3],
         [s2*r3],
         [np.sqrt(1-n_3)]]
# Third: Create the Rotation Matrix

    # This was the old rotation matrix calculation...
    #rotate = (np.outer(V, V) - np.dot(np.eye(3),(R)))

    # But here is the new one which more correctly implements the equations from the paper referenced above...
    rotate = (2 * np.dot(np.outer(V, V), R) - np.dot(np.eye(3), R))

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

def calc_SnowLine(host_star):
    ''' Calculates the Snow Line (Ida & Lin 2005, Kennedy & Kenyon 2008)
    '''
    return 2.7*(host_star.mass/ (1.0 | units.MSun))**2.0 | units.AU

def calc_JovianPlacement(host_star):
    ''' Calculates the placement of a Jovian, scaling Jupiter's location based
        on the host star's mass.
    '''
    a_jupiter = 5.454 | units.AU
    return a_jupiter*(host_star.mass/ (1.0 | units.MSun))**2.0

def calc_PeriodRatio(planet1_a, planet2_a, mu):
    period_1 = 2*np.pi*np.sqrt(planet1_a**3/mu)
    period_2 = 2*np.pi*np.sqrt(planet2_a**3/mu)
    return period_1/period_2

def calc_RelativePlanetPlacement(host_star, planet_mass, period_ratio_to_jovian):
    ''' Calculates the relative placement of a planet based on a stable period
        ratio with the orbital period of the system's jovian.
    '''
    a_jovian = calc_JovianPlacement(host_star)
    mu = constants.G*host_star.mass
    period_jovian = 2*np.pi*np.sqrt(a_jovian**3/mu)
    period_planet = period_ratio_to_jovian*period_jovian
    init_a = ((period_planet**2.)/(4.*np.pi**2.)*mu)**(1./3.)
    return init_a

def calc_SOI(m_star, vel_var, G):
    ''' Approximates the Sphere of Influence for a singlular stellar object.
        m_star: The mass of the stellar object.
        vel_var: The velocity dispursion of the system.

        !! https://en.wikipedia.org/wiki/Sphere_of_influence_(black_hole)
    '''
    return G*m_star/vel_var

def new_truncated_kroupa(number_of_stars, **kwargs):
    """ Returns a Kroupa (2001) Mass Distribution in SI units, with Mass Ranges:
            [Min_Mass, 0.08, 0.5, Max_Mass] MSun,
        and power-law exponents of each mass range:
            [-0.3, -1.3, -2.3]
        min_mass: the low-mass cut-off (Defaults to 0.1 MSun)
        max_mass: the high-mass cut-off (Defaults to 10.0 MSun)
    """
    min_mass = kwargs.get("min_mass", 0.1)
    max_mass = kwargs.get("max_mass", 10)
    return MultiplePartIMF(
        mass_boundaries = [min_mass, 0.08, 0.5, max_mass] | units.MSun,
        alphas = [-0.3, -1.3, -2.3], random=True
    ).next_mass(number_of_stars)

def get_stars(bodies):
    # anything with a Mass >= 13 Jupiter masses is a star
    limiting_mass_for_planets = 13 | units.MJupiter
    stars = bodies[bodies.mass > limiting_mass_for_planets]
    return stars

# if you apply this function to a particle set, it will return the planets
def get_planets(bodies):
    # anything with a Mass <= 13 Jupiter masses is a planet
    limiting_mass_for_planets = 13 | units.MJupiter
    planets = bodies[bodies.mass <= limiting_mass_for_planets]
    return planets

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

def check_isOver(bodies, smallN_worker=None):
    if smallN_worker == None:
        smallN_worker = SmallN(redirection = 'none', convert_nbody = converter)
        smallN_worker.initialize_code()
        smallN_worker.parameters.set_defaults()
        smallN_worker.parameters.allow_full_unperturbed = 0
    stars = get_stars(bodies.copy())
    smallN_worker.reset()
    smallN_worker.particles.add_particles(stars)
    smallN_worker.commit_particles()
    return smallN_worker.is_over()

def timestamp():
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return '['+st+']'
