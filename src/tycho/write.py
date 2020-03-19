# Python Classes/Functions used to Export Tycho's Datasets

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# TO-DO: Add time back to bookkeeping becasue of Tyler's code


# Importing Necessary System Packages
import os, io
import numpy as np
import matplotlib as plt
import random
import numpy

# Import the Amuse Base Packages
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.io import *
from amuse.lab import *
from amuse.couple import multiples
from amuse import io

# Import the Amuse Stellar Packages
from amuse.ic.kingmodel import new_king_model
from amuse.ic.kroupa import new_kroupa_mass_distribution

# Import cPickle/Pickle
try:
   import pickle as pickle
except:
   import pickle

from tycho import util
from tycho import multiples2 as multiples

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def write_initial_state(master_set, ic_array, file_prefix):
    ''' Writes out an initial state for the Tycho Module.
        master_set: The Master Amuse Particle Set used in Tycho
        ic_array: Predefined Numpy Array that Stores Initial Conditions in SI Units
        file_prefix: String Value for a Prefix to the Saved File
    '''    
# First, Define/Make the Directory for the Initial State to be Stored
    file_dir = os.getcwd()+"/InitialState"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_base = file_dir+"/"+file_prefix
# Second, Write the AMUSE Particle Set to a HDF5 File
    file_format = "hdf5"
    write_set_to_file(master_set, file_base+"_particles.hdf5", format=file_format, close_file=True)
# Third, Pickle the Initial Conditions Array
    ic_file = open(file_base+"_ic.pkl", "wb")
    pickle.dump(ic_array, ic_file)
    ic_file.close()

def write_time_step(gravity_set, current_time, file_prefix):
    ''' Writes out necessary information for a time step.
        master_set: The Master AMUSE Particle Set used in Tycho
        multiples_code: The Multiples Instance for Tycho
        current_time: The Simulations Current Time
        file_prefix: String Value for a Prefix to the Saved File
    '''
# First, Define/Make the Directory for the Time Step to be Stored
    file_dir_MS = os.getcwd()+"/Snapshots"
    if not os.path.exists(file_dir_MS):
        os.makedirs(file_dir_MS)
    file_base_MS = file_dir_MS+"/"+file_prefix
# Second, Write the AMUSE Particle Set to a HDF5 File
    file_format = "hdf5"
    write_set_to_file(gravity_set, file_base_MS+"_MS_t%.3f.hdf5" %(current_time.number), \
                      format=file_format, close_file=True)

# ------------------------------------ #
#        WRITING  RESTART FILE         #
# ------------------------------------ #

def write_state_to_file(time, stars_python,gravity_code, multiples_code, write_file, cp_hist=False, backup = 0 ):
    res_dir = os.getcwd()+"/Restart"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print("Writing state to write file: ", write_file,"\n\n")
    if write_file is not None:
        particles = gravity_code.particles.copy()
        write_channel = gravity_code.particles.new_channel_to(particles)
        write_channel.copy_attribute("index_in_code", "id")
        bookkeeping = {'neighbor_veto': multiples_code.neighbor_veto,
            'neighbor_distance_factor': multiples_code.neighbor_distance_factor,
            'multiples_external_tidal_correction': multiples_code.multiples_external_tidal_correction,
            'multiples_integration_energy_error': multiples_code.multiples_integration_energy_error,
            'multiples_internal_tidal_correction': multiples_code.multiples_internal_tidal_correction,
            'model_time': multiples_code.model_time,
            'root_index': multiples.root_index
        }
        
        for root, tree in multiples_code.root_to_tree.items():
            root_in_particles = root.as_particle_in_set(particles)
            subset = tree.get_tree_subset().copy()
            if root_in_particles is not None:
                root_in_particles.components = subset
        io.write_set_to_file(particles,write_file+".stars.hdf5",'hdf5',version='2.0', 
                             append_to_file=False, copy_history=cp_hist)
        io.write_set_to_file(stars_python,write_file+".stars_python.hdf5",'hdf5',version='2.0', 
                             append_to_file=False, copy_history=cp_hist)
        config = {'time' : time,
                  'py_seed': pickle.dumps(random.getstate()),
                  'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                 'options': pickle.dumps(options)
        }
        with open(write_file + ".conf", "wb") as f:
            pickle.dump(config, f)
        with open(write_file + ".bookkeeping", "wb") as f:
            pickle.dump(bookkeeping, f)
        print("\nState successfully written to:  ", write_file)
        print(time)
        if backup > 0:
            io.write_set_to_file(particles,write_file+".backup.stars.hdf5",'hdf5', version='2.0', 
                                 append_to_file=False, copy_history=cp_hist, close_file=True)
            io.write_set_to_file(stars_python,write_file+".backup.stars_python.hdf5",'hdf5', 
                                 version='2.0', append_to_file=False, copy_history=cp_hist, 
                                 close_file=True)
            config2 = {'time' : time,
                       'py_seed': pickle.dumps(random.getstate()),
                       'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                      'options': pickle.dumps(options)
            }
            with open(write_file + ".backup.conf", "wb") as f:
                pickle.dump(config2, f)
                f.close()
            with open(write_file + ".backup.bookkeeping", "wb") as f:
                pickle.dump(bookkeeping, f)
                f.close()
            print("\nBackup write completed.\n")
        
        if backup > 2:
            io.write_set_to_file(particles, write_file+"."+str(int(time.number))
                                 +".stars.hdf5",'hdf5',version='2.0', append_to_file=False, 
                                 copy_history=cp_hist, close_file=True)
            io.write_set_to_file(stars_python, write_file+"."+str(int(time.number))
                                 +".stars_python.hdf5",'hdf5',version='2.0', append_to_file=False, 
                                 copy_history=cp_hist, close_file=True)
            config2 = {'time' : time,
                       'py_seed': pickle.dumps(random.getstate()),
                       'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                      'options': pickle.dumps(options)
            }
            with open(write_file + "." +str(int(time.number))+".conf", "wb") as f:
                pickle.dump(config2, f)
                f.close()
            with open(write_file + "."+str(int(time.number))+".bookkeeping", "wb") as f:
                pickle.dump(bookkeeping, f)
                f.close()
            print("\nBackup write completed.\n")


# ----------------------------------------- #
#        WRITING CRASH RESTART FILE         #
# ----------------------------------------- #

def write_crash_save(time, stars_python,gravity_code, multiples_code, write_file, cp_hist=False, backup = 0 ):
    crash_dir = os.getcwd()+"/CrashSave"
    if not os.path.exists(crash_dir):
        os.makedirs(crash_dir)

    print("Writing state to write file: ", write_file,"\n\n")
    if write_file is not None:
        particles = gravity_code.particles.copy()
        write_channel = gravity_code.particles.new_channel_to(particles)
        write_channel.copy_attribute("index_in_code", "id")
        bookkeeping = {'neighbor_veto': multiples_code.neighbor_veto,
            'neighbor_distance_factor': multiples_code.neighbor_distance_factor,
                'multiples_external_tidal_correction': multiples_code.multiples_external_tidal_correction,
                    'multiples_integration_energy_error': multiples_code.multiples_integration_energy_error,
                        'multiples_internal_tidal_correction': multiples_code.multiples_internal_tidal_correction,
                        'model_time': multiples_code.model_time,
                        'root_index': multiples.root_index
        }
       
        '''
            bookkeeping.neighbor_veto =
            bookkeeping.multiples_external_tidal_correction = multiples_code.multiples_external_tidal_correction
            bookkeeping.multiples_integration_energy_error = multiples_code.multiples_integration_energy_error
            bookkeeping.multiples_internal_tidal_correction = multiples_code.multiples_internal_tidal_correction
            bookkeeping.model_time = multiples_code.model_time
        '''
        for root, tree in multiples_code.root_to_tree.items():
            #multiples.print_multiple_simple(tree,kep)
            root_in_particles = root.as_particle_in_set(particles)
            subset = tree.get_tree_subset().copy()
            if root_in_particles is not None:
                root_in_particles.components = subset
        io.write_set_to_file(particles,write_file+".stars.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist)
        io.write_set_to_file(stars_python,write_file+".stars_python.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist)
        config = {'time' : time,
            'py_seed': pickle.dumps(random.getstate()),
                'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                    'options': pickle.dumps(options)
        }

        with open(write_file + ".conf", "wb") as f:
            pickle.dump(config, f)
        with open(write_file + ".bookkeeping", "wb") as f:
            pickle.dump(bookkeeping, f)
        print("\nState successfully written to:  ", write_file)
        print(time)

        if backup > 0:
            io.write_set_to_file(particles,write_file+".backup.stars.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            io.write_set_to_file(stars_python,write_file+".backup.stars_python.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            config2 = {'time' : time,
                'py_seed': pickle.dumps(random.getstate()),
                    'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                        'options': pickle.dumps(options)
            }

            with open(write_file + ".backup.conf", "wb") as f:
                pickle.dump(config2, f)
                f.close()
            with open(write_file + ".backup.bookkeeping", "wb") as f:
                pickle.dump(bookkeeping, f)
                f.close()
            print("\nBackup write completed.\n")
        
        if backup > 2:
            io.write_set_to_file(particles,write_file+"."+str(int(time.number))+".stars.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            io.write_set_to_file(stars_python,write_file+"."+str(int(time.number))+".stars_python.hdf5",'hdf5',version='2.0', append_to_file=False, copy_history=cp_hist, close_file=True)
            config2 = {'time' : time,
                'py_seed': pickle.dumps(random.getstate()),
                    'numpy_seed': pickle.dumps(numpy.random.get_state()),
#                        'options': pickle.dumps(options)
            }

            with open(write_file + "." +str(int(time.number))+".conf", "wb") as f:
                pickle.dump(config2, f)
                f.close()
            with open(write_file + "."+str(int(time.number))+".bookkeeping", "wb") as f:
                pickle.dump(bookkeeping, f)
                f.close()
            print("\nBackup write completed.\n")    
    



