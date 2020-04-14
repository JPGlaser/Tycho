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
   import pickle as pickle
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

def gen_scatteringIC(encounter_db, doMultipleClusters=False):
    global rootDir
    global cluster_name
    max_number_of_rotations = 100
    if doMultipleClusters:
        output_ICDirectory = rootDir+'/'+cluster_name+'/Scatter_IC/'
    else:
        output_ICDirectory = rootDir+'/Scatter_IC/'
    if not os.path.exists(output_ICDirectory): os.mkdir(output_ICDirectory)
    # Set up the Kepler Workers for Subroutines Now
    converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    kepler_workers = [Kepler(unit_converter = converter, redirection = 'none'),
                      Kepler(unit_converter = converter, redirection = 'none')]
    for kw in kepler_workers:
        kw.initialize_code()
    # Loop Through the Star_IDs
    for star_ID in list(encounter_db.keys()):
        output_KeyDirectory = output_ICDirectory+str(star_ID)
        if not os.path.exists(output_KeyDirectory): os.mkdir(output_KeyDirectory)
        encounter_ID = 0
        for encounter in encounter_db[star_ID]:
            # Set Up Subdirectory for this Specific Encounter
            output_EncPrefix = output_KeyDirectory+"/Enc-"+str(encounter_ID)
            # Set up Encounter Key for this Specific Encounter for this Specific Star
            rotation_ID = 0
            while rotation_ID <= max_number_of_rotations:
                # Set Up Output Directory for this Specific Iteration
                output_HDF5File = output_EncPrefix+"_Rot-"+str(rotation_ID)+'.hdf5'
                next_outFile = output_EncPrefix+"_Rot-"+str(rotation_ID+1)+'.hdf5'
                if os.path.exists(output_HDF5File):
                    if rotation_ID == 99:
                        rotation_ID += 1
                        continue
                    elif os.path.exists(next_outFile):
                        rotation_ID += 1
                        continue
                # Remove Jupiter and Add Desired Planetary System
                enc_bodies = replace_planetary_system(encounter.copy(), kepler_workers=kepler_workers)
                write_set_to_file(enc_bodies.savepoint(0 | units.Myr), output_HDF5File, 'hdf5', version='2.0')
                printID = str(star_ID)+"-"+str(encounter_ID)+"-"+str(rotation_ID)
                print(util.timestamp(), "Finished Generating Random Encounter ID:", printID, "...")
                rotation_ID += 1
            encounter_ID += 1
    # Stop the Kepler Workers
    for kw in kepler_workers:
        kw.stop()

def replace_planetary_system(bodies, kepler_workers=None, base_planet_ID=50000, converter=None):
    # Set up the Converter if not ProvIDed
    if kepler_workers == None and converter == None:
        converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
    # Get the Hierarchical Systems from the Particle Set
    enc_systems = stellar_systems.get_heirarchical_systems_from_set(bodies, kepler_workers=kepler_workers, converter=converter)
    sys_with_planets = []
    # Remove Any Tracer Planets in the Encounter and Adds the Key to Add in the New System
    for sys_key in list(enc_systems.keys()):
        for particle in enc_systems[sys_key]:
            if particle.id >= base_planet_ID:
                enc_systems[sys_key].remove_particle(particle)
                sys_with_planets.append(sys_key)
    # Allows for Planets to be Added to Single Stars
    for sys_key in list(enc_systems.keys()):
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
                      help="Enter the full directory of the Root Folder. Defaults to your CWD unless -M is on.")
    parser.add_option("-M", "--doMultipleClusters", dest="doMultipleClusters", action="store_true",
                      help="Flag to turn on for running the script over a series of multiple clusters.")
    (options, args) = parser.parse_args()

    if options.doMultipleClusters:
        if options.rootDir != None:
            rootDir = options.rootDir+'/*'
        else:
            print(util.timestamp(), "Please provide the path to your root directory which contains all cluster folders!", cluster_name,"...")
    else:
        if options.rootDir != None:
            rootDir = options.rootDir
        else:
            rootDir = os.getcwd()
    # Bring Root Directory Path Inline with os.cwd()
    if rootDir.endswith("/"):
        rootDir = rootDir[:-1]

    orig_stdout = sys.stdout
    log_file = open(rootDir+"/rand_encounters.log","w")
    sys.stdout = log_file

    paths_of_enc_files = glob.glob(rootDir+'/*_encounters_cut.pkl')
    print(paths_of_enc_files)
    cluster_names = [path.split("/")[-2] for path in paths_of_enc_files]
    print(cluster_names)

    for i, path in enumerate(paths_of_enc_files):
        # Read in Encounter Directory
        encounter_db = pickle.load(open(path, "rb"))
        cluster_name = cluster_names[i]
        # Report Start of Generating IC for Cut Encounter Directory
        sys.stdout.flush()
        print(util.timestamp(), "Generating initial conditions for", cluster_name,"...")
        sys.stdout.flush()
        # Generate IC for Scattering Experiments
        gen_scatteringIC(encounter_db)

    sys.stdout = orig_stdout
    log_file.close()
