# Python Classes/Functions used to Analyze Tycho's Output

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #
# Importing Necessary System Packages
import pickle
import numpy as np
import glob
import re, os
# Import the Amuse Base Packages
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.datamodel import particle_attributes
from amuse.io import *
from amuse.lab import *

from collections import defaultdict

# Create a Safety Mechanism for Not Having MatPlotLib Installed
try:
    %matplotlib inline
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import *
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
    matplotlib.rcParams['figure.figsize'] = (16, 6)
    matplotlib.rcParams['font.size'] = (14)
except ImportError:
    HAS_MATPLOTLIB = False

# ------------------------------------- #
#      Reading of Snapshots in Bulk     #
# ------------------------------------- #
def bulk_read_snapshots(snapshot_dir, step_size=None):
    file_list = glob.glob(snapshot_dir+'/*.hdf5')
    snapshots = []
    if step_size != None:
        for file in file_list[::step_size]:
            snapshots.append(read_set_from_file(file, format="hdf5"))
    else:
        for file in file_list:
            snapshots.append(read_set_from_file(file, format="hdf5"))
    return snapshots

def get_snapshots(base_dir, step_size=None):
    snapshots_s_dir = base_dir+"/Snapshots/Stars"
    snapshots_p_dir = base_dir+"/Snapshots/Planets"
    snapshots_stars = bulk_read_snapshots(snapshots_s_dir)
    snapshots_planets = bulk_read_snapshots(snapshots_p_dir)
    return snapshots_stars, snapshots_planets

# ------------------------------------- #
#           Keplerian Analysis          #
# ------------------------------------- #

def get_component_binary_elements(comp1, comp2, kep, peri = True):

    total_mass = comp1.mass + comp2.mass
    kep_pos = comp2.position - comp1.position
    kep_vel = comp2.velocity - comp1.velocity
    
    # Assigns Which Component is the Star/Planet
    if comp1.mass<comp2.mass:
        Mp = comp1.mass
        Ms = comp2.mass
        planet = comp1
        hoststar = comp2
    else:
        Mp = comp2.mass
        Ms = comp1.mass
        planet = comp2
        hoststar = comp1
        
    # For Dopplershift
    rel_pos = planet.position - hoststar.position
    rel_vel = planet.velocity - hoststar.velocity
    
    kep.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2],
                                  kep_vel[0], kep_vel[1], kep_vel[2])
    a,e = kep.get_elements()
    r = kep.get_separation()
    E,J = kep.get_integrals()           # per unit reduced mass, note
    if peri:
        M,th = kep.get_angles()
        if M < 0:
            kep.advance_to_periastron()
        else:
            kep.return_to_periastron()
    t = kep.get_time()
    P = kep.get_period()

    # Calculating Angular Momentum Vector, h = r x v
    h = np.cross(rel_pos, rel_vel)
    # Calculating the Inclination in Radians 
    # https://en.wikibooks.org/wiki/Astrodynamics/Classical_Orbit_Elements#Inclin$
    I = np.arccos(h[2]/np.sqrt(h.dot(h)))
    if e**2 >=1:
        K = 0 | units.kms
    else:
        K = (2*np.pi/P)*(Mp*np.sin(I)/total_mass)*(a/np.sqrt(1-e**2))
    return total_mass, a, e, r, E, t, P, I, K

def get_planet_orbit_snapshot(planet_set, star_set, kep):
    planetary_systems = Particles()
    for planet in planet_set:
        hoststar = star_set[star_set.id == planet.host_star][0]
        # Initialize Kepler
        converter = nbody_system.nbody_to_si(planet.mass+hoststar.mass, hoststar.radius)
        kep.unit_converter = converter
        # Initialize the System as an AMUSE Particle
        system = Particle()
        # Get the Kepler Parameters & Radial Velocity Amplitude
        system.mass, system.a, system.e, system.seperation, \
        system.E, system.time, system.period, system.incl, system.rad_vel = \
        get_component_binary_elements(hoststar, planet, kep)
        # Save the System into the Planetary Systems Particle Set
        system.id = planet.id
        planetary_systems.add_particle(system)
    return planetary_systems

def get_timeseries_orbital_elements(star_snapshots, planet_snapshots, kep):
    planetary_oe = defaultdict(list)
    for planet in planet_snapshots[0]:
        planetary_oe[str(planet.id)] = Particles()
    for index in xrange(len(planet_snapshots)):
        planet_set = planet_snapshots[index]
        star_set = star_snapshots[index]
        planetary_systems = get_planet_orbit_snapshot(planet_set, star_set, kep)
        for system in planetary_systems:
            planetary_oe[str(system.id)].add_particle(system)
        print "Progress: "+str((index+1)*1.0/len(planet_snapshots)*100)+"%"
    return planetary_oe

# ------------------------------------- #
#           Keplerian Analysis          #
# ------------------------------------- #
