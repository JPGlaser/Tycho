# Python Classes/Functions used to Creating Tycho's Elements

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import math
import numpy as np
import matplotlib as plt
import random as rp

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

from tycho import util

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #


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
    print stars_SI.mass.as_quantity_in(units.MSun)
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
        select_stars_indices_for_binaries = rp.sample(xrange(0, num_stars), num_binaries)
        select_stars_indices_for_binaries.sort()
        delete_star_indices = []
# Creates a decending list of Star Indicies to make deletion easier.
        for y in xrange(0, num_binaries):
            delete_star_indices.append(select_stars_indices_for_binaries[(num_binaries-1)-y])
# Creates the Binaries and assigns IDs
        for j in xrange(0, num_binaries):
            q = select_stars_indices_for_binaries[j]
            binaries.add_particles(binary_system(stars_SI[q], converter))
        binaries.id = np.arange(num_binaries*2) + 2000000
# Deletes the Stars that were converted to Binaries
        for k in xrange(0, num_binaries):
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
    select_stars_indices = rp.sample(xrange(0, num_stars), num_systems)
    i = 0
    ID_Earth = 3000000
    ID_Jupiter = 5000000
    ID_Neptune = 8000000
    systems = datamodel.Particles()
# Begins to Build Planetary Systems According to Provided Information
    for system in xrange(num_systems):
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
