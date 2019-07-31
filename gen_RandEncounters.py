# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import sys, os, math
import numpy as np
import time as tp
from optparse import OptionParser
import glob

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
from tycho import util

# Import the Amuse Gravity & Close-Encounter Packages
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler

# Import the Tycho Packages
from tycho import create, util, read, write, stellar_systems

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def gen_scatteringIC(encounter_db):
    max_number_of_rotations = 100
    for star_ID in encounter_db.keys()
        output_KeyDirectory = os.getcwd()+"/Scatter_IC/"+str(star_id)
        encounter_id = 0
        for encounter in encounter_db[star_ID]:
            # Set Up Subdirectory for this Specific Encounter
            output_EncPrefix = output_KeyDirectory+"/Enc-"+str(encounter_id)
            # Set up Encounter Key for this Specific Encounter for this Specific Star
            rotation_id = 0
            while rotation_id <= max_number_of_rotations:
                # Set Up Output Directory for this Specific Iteration
                output_HDF5File = output_EncPrefix+"_Rot-"+str(rotation_id)+'.hdf5'
                next_outFile = output_EncPrefix+"_Rot-"+str(rotation_id+1)+'.hdf5'
                if os.path.exists(output_HDF5File):
                    if rotation_id == 99:
                        rotation_id += 1
                        continue
                    elif os.path.exists(next_outFile):
                        rotation_id += 1
                        continue
                # Remove Jupiter and Add Desired Planetary System
                enc_bodies = replace_planetary_system(encounter.copy())
                write_set_to_file(enc_bodies.savepoint(0 | units.Myr), output_HDF5File, 'hdf5', version='2.0')
                rotation_id += 1
            encounter_id += 1

def replace_planetary_system(bodies, base_planet_ID=50000, converter=None):
    # Set up the Converter if not Provided
    if converter == None:
        converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
    # Get the Hierarchical Systems from the Particle Set
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
    # Add in a New Planetary System
    for sys_key in sys_with_planets:
        planets = create.planetary_systems_v2(enc_systems[sys_key], 1, Jupiter=True, Earth=True, Neptune=True)
        enc_systems[sys_key].add_particles(planets)
    new_bodies = Particles()
    for sys_key in enc_systems:
        new_bodies.add_particles(enc_systems[sys_key])
    return new_bodies

# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--rootdirectory", dest="rootDir", default=None, type="str",
                      help="Enter the full directory of the Root Folder.")
    (options, args) = parser.parse_args()
    if options.rootDir != None:
        rootDir = options.rootDir
    else:
        rootDir = '/home/draco/jglaser/Public/Tycho_Runs/MarkG/'

    orig_stdout = sys.stdout
    log_file = open(os.getcwd()+"/rand_encounters.log","w")
    sys.stdout = log_file

    paths_of_enc_files = glob.glob(rootDir+'*_encounters_cut.pkl')
    cluster_names = [path.split("/")[-2] for path in paths_of_enc_files]

    for i, path in enumerate(paths_of_enc_files):
        # Read in Encounter Directory
        encounter_db = pickle.load(open(path, "rb"))
        # Report Start of Generating IC for Cut Encounter Directory
        sys.stdout.flush()
        print util.timestamp(), "Generating initial conditions for", cluster_names[i],"..."
        sys.stdout.flush()
        # Generate IC for Scattering Experiments
        gen_scatteringIC(encounter_db)

    sys.stdout = orig_stdout
    log_file.close()
