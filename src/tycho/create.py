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
from amuse.ext.orbital_elements import new_binary_from_orbital_elements

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
    #rand_seed = int(str(kwargs.get("seed", 7)).encode('hex'), 32)
    rand_seed = util.new_seed_from_string(kwargs.get("seed", 7))
    num_binaries = kwargs.get("num_binaries", int(num_stars*0.25))

# Apply the Seed for the Cluster
    np.random.seed(rand_seed)
    #rp.seed(rand_seed)
# Creates a List of Primary Masses (in SI Units) Drawn from the Kroupa IMF
    SMasses_SI = new_kroupa_mass_distribution(num_stars, mass_max = 10 | units.MSun)
# Creates a List of Binary Companion Masses (in SI_Units) Drawn from the Kroupa IMF
    BMasses_SI = new_kroupa_mass_distribution(num_binaries, mass_max = 10 | units.MSun)
# Creates the SI-NBody Converter
    converter = nbody_system.nbody_to_si(SMasses_SI.sum()+BMasses_SI.sum(), 1 | units.parsec)
# Creates a AMUS Particle Set Consisting of Positions (King) and Masses (Kroupa)
    stars_SI = new_king_model(num_stars, w0, convert_nbody=converter)
    stars_SI.mass = SMasses_SI
    #print stars_SI.mass.as_quantity_in(units.MSun)
# Assigning IDs to the Stars
    stars_SI.id = np.arange(num_stars) + 1
    stars_SI.type = "star"

# If Requested, Search for Possible Binary Systems
    if int(num_binaries) != 0:
    # Search for all Stars Able to Become Binary Systems & Update stars_SI Masses for Correct Scaling
        stars_SI, stars_to_become_binaries = find_possible_binaries(stars_SI, num_binaries, BMasses_SI)
    # Update the Number of Binaries (Safety Check)
        num_binaries = len(stars_to_become_binaries)

# Moving Stars to Virial Equilibrium
    stars_SI.move_to_center()
    if num_stars == 1:
        pass
    else:
        stars_SI.scale_to_standard(convert_nbody=converter)

# Assigning SOI Estimate for Interaction Radius
    if num_stars == 1:
        stars_SI.radius = 2000*stars_SI.mass/(1.0 | units.MSun) | units.AU
    else:
        stars_SI.radius = 2000*stars_SI.mass/(1.0 | units.MSun) | units.AU # Temporary Solution
        # Need to think of a better way to calculate the SOI
        # stars_SI.radius = 100*util.calc_SOI(stars_SI.mass, np.var(stars_SI.velocity), G=units.constants.G)

# If Requested, Creates Binary Systems
    if int(num_binaries) != 0:
    # Loop the Creation of Binary Systems
        binaries=Particles()
        for j in xrange(0, num_binaries):
            binaries.add_particles(binary_system(stars_to_become_binaries[j]))
    # Adjust the ID for each Binary Component Particle
        binaries.id = np.arange(num_binaries*2) + num_stars + 1
    # Remove Old Place-Holder Particles
        stars_SI.remove_particles(stars_to_become_binaries)
    # Merge the Two Sets
        stars_SI.add_particles(binaries)
    # Fix the ID Problem
        stars_SI.id = np.arange(len(stars_SI.id)) + 1

# Final Radius Setting (Ensuring that the Interaction Distance is not Small)
    min_stellar_radius = 1000 | units.AU
    for star in stars_SI:
        if star.radius < min_stellar_radius:
            star.radius = min_stellar_radius

# Returns the Cluster & Converter
    return stars_SI, converter


def find_possible_binaries(stars_SI, num_binaries, BMasses_SI):
    stars_to_become_binaries = Particles()
    while len(stars_to_become_binaries) < num_binaries:
        for star in stars_SI:
        # From Raghavan et al. 2010 (Figure 12), by eye:
	    # Linear Fit: fb = 0.2 * log10(mass) + 0.5
	    # Therefore: 0.1 Msun (M7), fb = 0.3; 1 Msun (G8), fb = 0.5; 10 Msun (B2), fb = 0.7
            assigned_p = rp.uniform(0,1)
            if not star in stars_to_become_binaries:
                fb = 0.2 * np.log10(star.mass.value_in(units.MSun)) + 0.5
                if assigned_p <= fb:
                # Add the Selected Star to the Subset
                    stars_to_become_binaries.add_particle(star)
                # Creates a Counter to be Used in Selecting the Mass from a Pre-Defined List
                    B_Counter = len(stars_to_become_binaries)-1
                # Draws a Pre-Defined Companion Mass
                    stars_to_become_binaries[-1].mass = star.mass+BMasses_SI[B_Counter]
                # Redefines the Selected Star's Mass to Allow for Correct Scaling in Cluster Script
                    star.mass += BMasses_SI[B_Counter]
        # Check to Break the While Loop if the Number of Binaries is Reached
            if len(stars_to_become_binaries) >= num_binaries:
                break
    return stars_SI, stars_to_become_binaries


def binary_system(star_to_become_binary, **kwargs):
# Check Keyword Arguments
    doFlatEcc = kwargs.get("FlatEcc",True) # Apply Uniform Eccentricity Distribution
    doBasic = kwargs.get("Basic", False) # Apply a Basic Binary Distribution
    doFlatQ = kwargs.get("FlatQ",True) # Apply a Uniform Mass-Ratio Distribution
    doRag_P = kwargs.get("RagP",True) # Apply Raghavan et al. (2010) Period Distribution
    doSana_P = kwargs.get("SanaP", False) # Apply Sana et al. (2012) Period Distribution
    Pcirc = kwargs.get("Pcirc", 6 | units.day ) # Circularization Period
    Pmin = kwargs.get("Pmin", 3. | units.day ) # Min Orbital Period Allowed
    Pmax = kwargs.get("Pmax", 10.**5. | units.day ) # Max Orbital Period Allowed

# Define Original Star's Information
    rCM = star_to_become_binary.position
    vCM = star_to_become_binary.velocity
# Define Initial Binary Particle Set
    binary = Particles(2)
    star1 = binary[0]
    star2 = binary[1]
    star1.type = 'star'
    star2.type = 'star'

# If Desired, Apply a Basic Binary Distribution
    if (doBasic):
        semi_major_axis = 500. | units.AU
        e = 0.
        star1.mass = 0.5*star_to_become_binary.mass
        star2.mass = 0.5*star_to_become_binary.mass

# If Desired, Apply the Uniform Mass-Ratio Distribution (Goodwin, 2012)
    if (doFlatQ):
        min_stellar_mass = 100 | units.MJupiter # Greater Mass Than "AB Doradus C"
        while (star1.mass < min_stellar_mass) or (star2.mass < min_stellar_mass):
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
    #if (period < Pcirc):
    #    e = 0.0

# Create the New Binary
    newBinary = new_binary_from_orbital_elements(star1.mass, star2.mass, semi_major_axis,
                                                 eccentricity = e, G = constants.G)
# Rotate the System
    util.preform_EulerRotation(newBinary)
    star1.position = rCM + newBinary[0].position
    star1.velocity = vCM + newBinary[0].velocity
    star2.position = rCM + newBinary[1].position
    star2.velocity = vCM + newBinary[1].velocity

# Apply a Fitting Dynamical Radius
    binary.radius = 5*semi_major_axis

# Return the Particle Set of Stars in the Binary
    return binary


def planetary_systems(stars, num_systems, filename_planets, **kwargs):
    ''' Creates several mock planetary systems around random stars in the provided set.
        stars: The AMUSE Particle Set containing stellar information.
        num_systems: The number of planetary systems requested.
        filename_planets: Filename for the Initial Planetary System HDF5 Archive.
        Earth, Jupiter, Neptune: Booleans asking if they should be included.
    '''
    makeEarth = kwargs.get("Earth", False)
    makeJupiter = kwargs.get("Jupiter", True)
    makeNeptune = kwargs.get("Neptune", False)
# Sets Initial Parameters
    num_stars = len(stars)
    select_stars_indices = rp.sample(xrange(0, num_stars), num_systems)
    i = 0
    ID_Earth = 30000
    ID_Jupiter = 50000
    ID_Neptune = 80000
    systems = datamodel.Particles()
# Begins to Build Planetary Systems According to Provided Information
    for system in xrange(num_systems):
        planets = datamodel.Particles()
        j = select_stars_indices[system]
        if makeEarth:
            init_a = 1.000 | units.AU
            init_e = 0.016
            mass_E = 0.003 | units.MJupiter
            planets.add_particle(planet(ID_Earth+system, stars[j], mass_E, init_a, init_e))
        if makeJupiter:
            init_a = 5.454 | units.AU
            init_e = 0.048
            mass_J = 1 | units.MJupiter
            planets.add_particle(planet(ID_Jupiter+system, stars[j], mass_J, init_a, init_e))
        if makeNeptune:
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
        util.preform_EulerRotation(p)
# Moves the Planet to Orbit the Host Star
    p.position = p.position + host_star.position
    p.velocity = p.velocity + host_star.velocity
# Returns the Created AMUSE Particle
    return p

class GalacticCenterGravityCode(object):
    def __init__(self,R, M, alpha):
        self.radius=R
        self.mass=M
        self.alpha=alpha
    def get_gravity_at_point(self,eps,x,y,z):
        r2=x**2+y**2+z**2
        r=r2**0.5
        m=self.mass*(r/self.radius)**self.alpha
        fr=constants.G*m/r2
        ax=-fr*x/r
        ay=-fr*y/r
        az=-fr*z/r
        return ax,ay,az
    def circular_velocity(self,r):
        m=self.mass*(r/self.radius)**self.alpha
        vc=(constants.G*m/r)**0.5
    	return vc
    def move_particles_into_ellipictal_orbit(self, particles, Rinit):
        particles.x += Rinit
        particles.vy += 0.9*self.circular_velocity(Rinit)
