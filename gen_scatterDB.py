import numpy as np
import random as rp
import os, sys
import scipy as sp
from scipy import optimize
from scipy import special
import pickle
import glob
import copy

# Import the Amuse Base Packages
import amuse
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.datamodel import particle_attributes
from amuse.io import *
from amuse.lab import *

from tycho import util, create, read, write, stellar_systems

set_printing_strategy("custom", preferred_units = [units.MSun, units.AU, units.day], precision = 6, prefix = "", separator = "[", suffix = "]")

def replace_planetary_system(bodies, base_planet_ID=50000, converter=None):
    enc_systems = stellar_systems.get_heirarchical_systems_from_set(bodies, converter=converter)
    sys_with_planets = []
    # Remove Any Tracer Planets in the Encounter and Adds the Key to Add in the New System
    for sys_key in enc_systems.keys():
        for particle in enc_systems[sys_key]:
            if particle.id >= base_planet_ID:
                enc_systems[sys_key].remove_particle(particle)
                sys_with_planets.append(sys_key)
    # Allows for Planets to be Added to Single Stars
    for sys_key in enc_systems.keys():
        if (len(enc_systems[sys_key]) == 1) and (sys_key not in sys_with_planets):
            sys_with_planets.append(sys_key)
    #print sys_with_planets
    # Add in a New Planetary System
    for sys_key in sys_with_planets:
        planets = create.planetary_systems_v2(enc_systems[sys_key], 1, Jupiter=True, Earth=True, Neptune=True)
        enc_systems[sys_key].add_particles(planets)
    new_bodies = Particles()
    for sys_key in enc_systems:
        new_bodies.add_particles(enc_systems[sys_key])
    return new_bodies


if __name__ == '__main__':
    cutEnc_filePath = glob.glob("/home/draco/jglaser/Public/Tycho_Runs/MarkG/*/*_encounters_cut.pkl")
    cluster_rootDir = glob.glob("/home/draco/jglaser/Public/Tycho_Runs/MarkG/*/")
    cluster_names = [x.split("/")[-2] for x in cluster_rootDir]

    cutEnc_db = {}
    for i, filepath in enumerate(cutEnc_filePath):
        opened_file = open(filepath, 'rb')
        cutEnc_db[cluster_names[i]] = pickle.load(opened_file)
        opened_file.close()
    print cutEnc_db.keys()

old_cutEnc_db = copy.deepcopy(cutEnc_db)

for clusterName in cutEnc_db.keys():
    for primaryStar in cutEnc_db[clusterName].keys():
        origionalEnc = cutEnc_db[clusterName][primaryStar][0].copy()
        newEncList = []
        for i in xrange(100):
            newEncList.append(replace_planetary_system(origionalEnc))
        cutEnc_db[clusterName][primaryStar] = newEncList

output_file = open(os.getcwd()+"full_scatterDB.pkl", 'rb')
pickle.dump(cutEnc_db, output_file)
print "Finished!"
