# Python Classes/Functions used to Creating Tycho's Elements

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import math
import numpy as np
import matplotlib as plt
import numpy.random as rp
import random

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

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from tycho import util

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #


def king_cluster_v2(num_stars, **kwargs):
    ''' Creates an open cluster according to the King Model & Kroupa IMF
        num_stars: The total number of stellar systems.
        w0: The King density parameter.
        vradius: The virial radius of the cluster.
        seed: The random seed used for cluster generation.
        do_binaries: Turn on/off binary creation.
        binary_recursions: The number of times a star is tested to be a binary.
        split_binaries: Turn on/off splitting Binary CoM into individual Companions.
    '''
# Check Keyword Arguments
    w0 = kwargs.get("w0", 2.5)
    virial_radius = kwargs.get("vradius", 2 | units.parsec)
    rand_seed = kwargs.get("seed", 7)
    do_binaries = kwargs.get("do_binaries", True)
    binary_recursions = kwargs.get("binary_recursions", 1)
    split_binaries = kwargs.get("split_binaries", True)

# Check if rand_seed is a integer or not. Convert it if it isn't.
    if not type(rand_seed) == type(1):
        rand_seed = util.new_seed_from_string(rand_seed)

# Apply the Seed for the Cluster
    rs = RandomState(MT19937(SeedSequence(rand_seed)))
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    min_stellar_mass = 100 | units.MJupiter
    max_stellar_mass = 10 | units.MSun
# Creates a List of Primary Masses (in SI Units) Drawn from the Kroupa IMF
    Masses_SI = util.new_truncated_kroupa(num_stars)

# If Primordial Binaries are Desired, Start the Conversion Process
    if do_binaries:
        # Find Sutable CoM Objects to Turn into Binaries, Update the System
        # Mass, and record the CoM's Index to Remove it Later.
        Masses_SI, ids_to_become_binaries = find_possible_binaries_v2(Masses_SI,
                                                binary_recursions = binary_recursions)

# Creates the SI-to-NBody Converter
    converter = nbody_system.nbody_to_si(Masses_SI.sum(), virial_radius)

# Creates a AMUS Particle Set Consisting of Positions (King) and Masses (Kroupa)
    stars_SI = new_king_model(num_stars, w0, convert_nbody=converter)
    stars_SI.mass = Masses_SI

# Assigning Type of System ('star' or 'primordial binary')
    stars_SI.type = "star"
    if do_binaries:
        for com_index in ids_to_become_binaries:
            stars_SI[com_index].type = "primordial binary"

# Shifts Cluster's CoM to the Origin Before Scaling to Virial Equilibrium
    stars_SI.move_to_center()
    if num_stars == 1:
        pass
    else:
        stars_SI.scale_to_standard(convert_nbody=converter)

# Assigning SOI Estimate for Interaction Radius
    if num_stars == 1:
        stars_SI.radius = 2000*stars_SI.mass/(1.0 | units.MSun) | units.AU
    else:
        # Temporary Solution
        stars_SI.radius = 2000*stars_SI.mass/(1.0 | units.MSun) | units.AU
        # Need to think of a better way to calculate the SOI
        # stars_SI.radius = 100*util.calc_SOI(stars_SI.mass, np.var(stars_SI.velocity), G=units.constants.G)

# If Requested, Split Binary Systems into Seperate Particles
    if do_binaries:
        # Define the Binary Set
        binaries = Particles()
        singles_in_binaries = Particles()
        com_to_remove = Particles()

        # Split the Binary into its Companions & Store in Seperate Sets
        for com_index in ids_to_become_binaries:
            stars_SI[com_index].id = com_index
            binary_particle, singles_in_binary = binary_system_v2(stars_SI[com_index])
            binaries.add_particle(binary_particle)
            singles_in_binaries.add_particle(singles_in_binary)
            com_to_remove.add_particle(stars_SI[com_index])

        # If Desired, Remove the CoM and Replace it with the Single Companions.
        # Note: Default is to do this until we get multiples.py and encounters.py
        #       to match functionality exactly plus Kira.py features.
        if split_binaries:
            stars_SI.remove_particles(com_to_remove)
            stars_SI.add_particles(singles_in_binaries)

# Set Particle Ids for Easy Referencing
    stars_SI.id = np.arange(len(stars_SI)) + 1

# Final Radius Setting (Ensuring that the Interaction Distance is not Small)
    min_stellar_radius = 1000 | units.AU
    for star in stars_SI:
        if star.radius < min_stellar_radius:
            star.radius = min_stellar_radius

# Return the Desired Particle Sets and Required Converter
    if do_binaries:
        return stars_SI, converter, binaries, singles_in_binaries
    else:
        return stars_SI, converter

def find_possible_binaries_v2(com_mass_array, **kwargs):
    binary_recursions = kwargs.get("binary_recursions", 1)
    min_stellar_mass = kwargs.get("min_mass", 100 | units.MJupiter)
    max_stellar_mass = kwargs.get("max_mass", 10 | units.MSun)

    ids_to_become_binaries = []
    recursion_counter = 0
    while recursion_counter < binary_recursions:
        recursion_counter += 1
        current_com_id = 0
        for com_mass in com_mass_array:
            assigned_probability = rp.uniform(0, 1)
            #print(assigned_probability)
            if not current_com_id in ids_to_become_binaries:
                fb = 0.2 * np.log10(com_mass.value_in(units.MSun)) + 0.5
                if assigned_probability <= fb:
                # If the Assigned Probability is LTE the Binary Likihood ...
                # Add the Index to the Array for Later CoM Removal
                    ids_to_become_binaries.append(current_com_id)
                # Draw a Distrubution of Kroupa Masses
                    possible_extra_mass = util.new_truncated_kroupa(100)
                # Randomly Select one of the Above Masses
                    selected_index = int(np.floor(100*rp.uniform(0, 1)))
                    selected_extra_mass = possible_extra_mass[selected_index]
                    #print(selected_extra_mass)
                # Add the Selected Mass to the Current CoM Mass
                    com_mass_array[current_com_id] += selected_extra_mass
            current_com_id += 1
    return com_mass_array, ids_to_become_binaries

def binary_system_v2(star_to_become_binary, **kwargs):
# Check Keyword Arguments
    doFlatEcc = kwargs.get("FlatEcc",True) # Apply Uniform Eccentricity Distribution
    doBasic = kwargs.get("Basic", False) # Apply a Basic Binary Distribution
    doFlatQ = kwargs.get("FlatQ",True) # Apply a Uniform Mass-Ratio Distribution
    doRag_P = kwargs.get("RagP",True) # Apply Raghavan et al. (2010) Period Distribution
    doSana_P = kwargs.get("SanaP", False) # Apply Sana et al. (2012) Period Distribution
    Pcirc = kwargs.get("Pcirc", 6 | units.day ) # Circularization Period
    Pmin = kwargs.get("Pmin", 10.**-1. | units.day ) # Min Orbital Period Allowed
    Pmax = kwargs.get("Pmax", 10.**10. | units.day ) # Max Orbital Period Allowed

# Define Original Star's Information
    rCM = star_to_become_binary.position
    vCM = star_to_become_binary.velocity
# Define Initial Binary Particle Set
    singles_in_binary = Particles(2)
    star1 = singles_in_binary[0]
    star2 = singles_in_binary[1]
    star1.type = 'star'
    star2.type = 'star'
    star1.mass = 0. | units.MSun
    star2.mass = 0. | units.MSun

# If Desired, Apply a Basic Binary Distribution
    if (doBasic):
        semi_major_axis = 500. | units.AU
        e = 0.
        star1.mass = 0.5*star_to_become_binary.mass
        star2.mass = 0.5*star_to_become_binary.mass

# If Desired, Apply the Uniform Mass-Ratio Distribution (Goodwin, 2012)
    if (doFlatQ):
        min_stellar_mass = 100. | units.MJupiter # Greater Mass Than "AB Doradus C"
        while star2.mass <= min_stellar_mass:
            q = np.random.random_sample()
            star1.mass = star_to_become_binary.mass / (1. + q)
            star2.mass =  q * star1.mass

# If Desired, Apply Raghavan et al. (2010) Period Distribution
    if (doRag_P):
        sigma = 2.28
        mu = 5.03
        period = 2.*Pmax
        while (period > Pmax or period < Pmin):
            #logP = sigma * np.random.randn() + mu
            logP = np.random.normal(loc=mu, scale=sigma)
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

# Get the Companion's Positions from Kepler Relative to the Origin
    newBinary = new_binary_from_orbital_elements(star1.mass, star2.mass, semi_major_axis,
                                                 eccentricity = e, G = constants.G)

# Rotate the Binary System & Move to the CoM's Position
    util.preform_EulerRotation(newBinary)
    star1.position = rCM + newBinary[0].position
    star1.velocity = vCM + newBinary[0].velocity
    star2.position = rCM + newBinary[1].position
    star2.velocity = vCM + newBinary[1].velocity

# Apply a Fitting Dynamical Radius
    singles_in_binary.radius = 2*semi_major_axis

# Create the Binary System Particle (For Stellar Evolution Code)
    star_to_become_binary.radius = 5*semi_major_axis
    binary_particle = star_to_become_binary.copy()
    binary_particle.child1 = star1
    binary_particle.child2 = star2
    binary_particle.semi_major_axis = semi_major_axis
    binary_particle.eccentricity = e
    binary_particle.id = star_to_become_binary.id

# Return the Binary System Particle & the Particle Set of Individual Companions
    return binary_particle, singles_in_binary

def planetary_systems_v2(stars, num_systems, **kwargs):
    ''' Creates several mock planetary systems around random stars in the provided set.
        stars: The AMUSE Particle Set containing stellar information.
        num_systems: The number of planetary systems requested.
        filename_planets: Filename for the Initial Planetary System HDF5 Archive.
        Earth, Jupiter, Neptune: Booleans asking if they should be included.
    '''
    makeEarth = kwargs.get("Earth", False)
    makeJupiter = kwargs.get("Jupiter", True)
    makeNeptune = kwargs.get("Neptune", False)
    makeTestPlanet = kwargs.get("TestP", False)

# Selects the Stars to Become Planetary Systems
    num_stars = len(stars)
    if num_systems > num_stars:
        num_systems = num_stars
    select_stars_indices = random.sample(range(0, num_stars), num_systems)

# Sets Important Parameters
    ID_Earth = 30000
    ID_Jupiter = 50000
    ID_Neptune = 80000
    systems = datamodel.Particles()
# Begins to Build Planetary Systems According to Provided Information
    for system in range(num_systems):
        planets = datamodel.Particles()
        j = select_stars_indices[system]
        host_star = stars[j]
        mu = constants.G*host_star.mass
        if makeEarth:
            period_ratio = np.sqrt((1.000 | units.AU)**3/(5.454 | units.AU)**3)
            mass_E = 0.003 | units.MJupiter
            init_a = util.calc_RelativePlanetPlacement(host_star, mass_E, period_ratio)
            init_e = 0.016
            Earth = planet_v2(ID_Earth+host_star.id, host_star, mass_E, init_a, init_e)
            Earth.stellar_type = 1
            planets.add_particle(Earth)
        if makeJupiter:
            init_a = util.calc_JovianPlacement(host_star)
            init_e = 0.048
            mass_J = 1 | units.MJupiter
            Jupiter = planet_v2(ID_Jupiter+host_star.id, host_star, mass_J, init_a, init_e)
            Jupiter.stellar_type = 1
            planets.add_particle(Jupiter)
        if makeTestPlanet:
            init_a = util.calc_JovianPlacement(host_star)
            init_e = 0.048
            mass_J = 20 | units.MJupiter
            TestP = planet_v2(ID_Jupiter+host_star.id, host_star, mass_J, init_a, init_e)
            TestP.stellar_type = 1
            planets.add_particle(TestP)
        if makeNeptune:
            period_ratio = np.sqrt((30.110 | units.AU)**3/(5.454 | units.AU)**3)
            mass_N = 0.054 | units.MJupiter
            init_a = util.calc_RelativePlanetPlacement(host_star, mass_N, period_ratio)
            init_e = 0.009
            Neptune = planet_v2(ID_Neptune+host_star.id, host_star, mass_N, init_a, init_e)
            Neptune.stellar_type = 1
            planets.add_particle(Neptune)
    # Moves Planetary System to the Origin and Applies a Random Euler Rotation
        for p in planets:
            p.position = p.position - host_star.position
            p.velocity = p.velocity - host_star.velocity
        util.preform_EulerRotation(planets)
        for p in planets:
            p.position = p.position + host_star.position
            p.velocity = p.velocity + host_star.velocity
    # Adds the System to the Provided AMUSE Particle Set
        systems.add_particles(planets)
    return systems

def planet_v2(ID, host_star, planet_mass, init_a, init_e, random_orientation=False):
    ''' Creates a planet as an AMUSE Particle with provided characteristics.
        ID: Identifying number unique to this planet.
        host_star: The AMUSE Particle that is the host star for the planet.
        planet_mass: The mass of the planet (in the nbody units).
        init_a: Initial semi-major axis (in nbody units).
        init_e: Initial eccentricity (in nbody units).
        random_orientation: Boolean to incline the planet in a random fashion.
    '''
# Define the Host Star's Original Location & Position
    rCM = host_star.position
    vCM = host_star.velocity
# Sets Planet Values to Provided Conditions
    p = datamodel.Particle()
    p.id = ID
    p.type = "planet"
    p.host_star = host_star.id
    p.mass = planet_mass
# Sets the Dynamical Radius to the Hill Sphere Approx.
    p.radius = util.calc_HillRadius(init_a, init_e, p.mass, host_star.mass)
# Generate a Random Position on the Orbit (True Anomaly)
# This ensures that all the planets don't start out along the same joining line.
    init_ta = 360*np.random.random() | units.deg
# Get the Host Star & Planets Positions from Kepler Relative to the Origin
    newPSystem = new_binary_from_orbital_elements(host_star.mass, p.mass, init_a,
                                                 eccentricity = init_e,
                                                 true_anomaly = init_ta,
                                                 G = constants.G)
# Rotate the Binary System & Move to the CoM's Position
    if random_orientation:
        util.preform_EulerRotation(newPSystem)
    host_star.position = rCM + newPSystem[0].position
    host_star.velocity = vCM + newPSystem[0].velocity
    p.position = rCM + newPSystem[1].position
    p.velocity = vCM + newPSystem[1].velocity
# Returns the Created AMUSE Particle
    return p
