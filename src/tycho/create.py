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


def king_cluster(num_stars, **kwargs):
    ''' Creates an open cluster according to the King Model & Kroupa IMF
        num_stars: The total number of stellar systems.
        w0: The King density parameter.	
        seed: The random seed used for cluster generation.
        num_binaries: The number of binary systems desired.
    '''
# Check Keyword Arguments
    w0 = kwargs.get("w0", 2.5)
    rand_seed = int(str(kwargs.get("seed", 7)).encode('hex'), 32)
    num_binaries = kwargs.get("num_binaries", int(num_stars*0.25))

# Apply the Seed for the Cluster
    np.random.seed(rand_seed)
# Creates a List of Masses (in SI Units) Drawn from the Kroupa IMF
    masses_SI = new_kroupa_mass_distribution(num_stars, mass_max = 5 | units.MSun)
# Creates the SI-NBody Converter
    converter = nbody_system.nbody_to_si(masses_SI.sum(), 1 | units.parsec)
# Creates a AMUS Particle Set Consisting of Positions (King) and Masses (Kroupa)
    stars_SI = new_king_model(num_stars, w0, convert_nbody=converter)
    stars_SI.mass = masses_SI
    #print stars_SI.mass.as_quantity_in(units.MSun)
# Assigning IDs to the Stars
    stars_SI.id = np.arange(num_stars) + 1
    stars_SI.type = "star"
# Moving Stars to Virial Equilibrium
    stars_SI.move_to_center()
    if num_stars == 1:
        pass
    else:
        stars_SI.scale_to_standard(convert_nbody=converter)
        
# Create Binary Systems if Requested
    if int(num_binaries) != 0:
    # Search for all Stars Able to Become Binary Systems
        stars_to_become_binaries = find_possible_binaries(stars_SI, num_binaries)
    # Update the Number of Binaries (Safety Check)
        num_binaries = len(stars_to_become_binaries)
    # Loop the Creation of Binary Systems
        binaries=Particles()
        for j in xrange(0, num_binaries):
            binaries.add_particles(binary_system(stars_to_become_binaries[j], converter))
    # Adjust the ID for each Binary Component Particle
        binaries.id = np.arange(num_binaries*2) + 2000000
    # Remove Old Place-Holder Particles
        stars_SI.remove_particles(stars_to_become_binaries)
    # Merge the Two Sets
        stars_SI.add_particles(binaries)

# Assigning SOI Estimate for Interaction Radius
    if num_stars == 1:
        stars_SI.radius = 1000 | units.AU
    else:
        stars_SI.radius = util.calc_SOI(stars_SI.mass, np.var(stars_SI.velocity), G=units.constants.G)
        
# Returns the Cluster & Converter
    return stars_SI, converter


def find_possible_binaries(stars_SI, num_binaries):
    stars_to_become_binaries = Particles()
    while len(stars_to_become_binaries) < num_binaries:
        for star in stars_SI:
        # Place-Holder Selection Criteria
            assigned_p = rp.uniform(0,1)
            print assigned_p
            if not star in stars_to_become_binaries:
                if star.mass <= 50 | units.MSun and assigned_p<=0.25:
                    stars_to_become_binaries.add_particle(star)# Only adds the Star if it has NOT been selected before.
                    stars_to_become_binaries[-1].mass = 2*star.mass # Rough Correction
        # Check to Break the While Loop if the Number of Binaries is Reached
            if len(stars_to_become_binaries) >= num_binaries:
                break
    return stars_to_become_binaries


def binary_system(star_to_become_binary, converter, **kwargs):
# Check Keyword Arguments
    doFlatEcc = kwargs.get("FlatEcc",True) # Apply Uniform Eccentricity Distribution
    doBasic = kwargs.get("Basic", False) # Apply a Basic Binary Distribution
    doFlatQ = kwargs.get("FlatQ",True) # Apply a Uniform Mass-Ratio Distribution
    doRag_P = kwargs.get("RagP",True) # Apply Raghavan et al. (2010) Period Distribution
    doSana_P = kwargs.get("SanaP", True) # Apply Sana et al. (2012) Period Distribution
    Pcirc = kwargs.get("Pcirc", 6 | units.day ) # Circularization Period 
    Pmin = kwargs.get("Pmin", 3. | units.day ) # Min Orbital Period Allowed
    Pmax = kwargs.get("Pmax", 10.**5. | units.day ) # Max Orbital Period Allowed
    
# Define Original Star's Information
    rCM = star_to_become_binary.position
    print rCM
    vCM = star_to_become_binary.velocity
# Define Initial Binary Particle Set
    binary = Particles(2)
    star1 = binary[0]
    star2 = binary[1]
    star1.type = 'star'
    star2.type = 'star'

# If Desired, Apply a Basic Binary Distribution
    if (not doBasic):
        semi_major_axis = 500. | units.AU
        eccentricity = 0.
        star1.mass = 0.5*star_to_become_binary.mass
        star2.mass = 0.5*star_to_become_binary.mass

# If Desired, Apply the Uniform Mass-Ratio Distribution (Goodwin, 2012)
    if (doFlatQ):
            q = np.random.random()
            star1.mass = star_to_become_binary.mass / (1. + q)
            star2.mass =  q * star1.mass
        
# If Desired, Apply Raghavan et al. (2010) Period Distribution
    if (doRag_P):
        sigma = 2.28
        mu = 5.03
        period = 2.*Pmax
        while (period > Pmax or period < Pmin):
            logP = sigma * np.random.randn() + mu
            period = 10.**logP | units.day
            semi_major_axis = ((period**2.)/(4.*np.pi**2.)*constants.G*(star1.mass+star2.mass))**(1./3.)
        

# If Desired & Applicable, Apply Sana et al. (2012) Period Distribution
    if (doSana_P and star1.mass > 15 | units.MSun):
            maxLogP = np.log10(Pmax.value_in(units.day))
            minLogP = np.log10(Pmin.value_in(units.day))
            pMod = -0.55 + 1.
            x1 = np.random.random()
            logP = ((maxLogP**pMod-minLogP**pMod)*x1 + minLogP**pMod)**(1./pMod)
            period = 10.**logP | units.day
            semi_major_axis = ((period**2.)/(4.*np.pi**2.)*constants.G*(star1.mass+star2.mass))**(1./3.)
        
# If Desired, Apply Uniform Eccentricity Distribution
    if (doFlatEcc):
        e = rp.uniform(0.0,1.0)
    if (period < Pcirc):
        e = 0.0
        
# Create the New Binary
    newBinary = new_binary_from_orbital_elements(star1.mass, star2.mass, semi_major_axis, eccentricity = e, G = constants.G)
# Rotate the System
    util.preform_EulerRotation(newBinary)
    star1.position = rCM + newBinary[0].position
    star1.velocity = vCM + newBinary[0].velocity
    star2.position = rCM + newBinary[1].position
    star2.velocity = vCM + newBinary[1].velocity
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
