import os
import glob
from amuse.units import nbody_system
from amuse.units import units
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.datamodel import particle_attributes
from amuse.io import *
from amuse.lab import *
from amuse.couple import multiples
from amuse.ext.galactic_potentials import MWpotentialBovy2015

# Import cPickle/Pickle
try:
   import cPickle as pickle
except:
   import pickle

def read_initial_state(initial_dir, file_prefix):
    ''' Reads in an initial state for the Tycho Module.
        file_prefix: String Value for a Prefix to the Saved File
    '''
# TODO: Convert the saved datasets from SI to NBody. Also everything else in this function.

# First, Define the Directory where Initial State is Stored
#    file_dir = os.getcwd()+"/InitialState"
    file_dir = initial_dir
    file_base = file_dir+"/"+file_prefix
# Second, Read the Master AMUSE Particle Set from a HDF5 File
    file_format = "hdf5"
    master_set = read_set_from_file(file_base+"_particles.hdf5", format=file_format, close_file=True)
# Third, unPickle the Initial Conditions Array
    ic_file = open(file_base+"_ic.pkl", "rb")
    ic_array = pickle.load(ic_file)
    ic_file.close()
# Fourth, convert ic_array.total_smass and viral_radius from strings to floats
    total_smass = float(ic_array.total_smass) | units.kg
    viral_radius = float(ic_array.viral_radius) | units.m
# Fifth, Define the Master Set's Converter
    converter = nbody_system.nbody_to_si(total_smass, viral_radius)
    return master_set, ic_array, converter


cluster_names = ["Emlen_N1000_W6"]

for cluster_name in cluster_names:
    pic_dir = "/home/draco/jthornton/FrescoPictures/"+cluster_name+"2/"
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    base_dir =    "/home/draco/jglaser/Public/Tycho_Runs/"
    initial_dir = base_dir+"Methods/"+cluster_name+"/InitialState"
    star_files_dir = base_dir+"Methods/"+cluster_name+"/Snapshots/Stars/"
    track_file = "/home/draco/jthornton/TrackSnaps/"+cluster_name+"_track.pkl"

    star_files = glob.glob(star_files_dir+"*.hdf5")
    print len(star_files)
    print cluster_name

    #star_files = open(pkl_file, "rb")
    #snapshot_stars = pickle.load(star_files)
    #star_files.close()

    track_file = open(track_file, "rb")
    snapshot_track = pickle.load(track_file)
    track_file.close()

    track2_x = snapshot_track[0]
    track2_y = snapshot_track[1]

#    age = "8" # In units of Myr
    # Gather the time steps from the file names and save them in an array

    #ms, ic, converter = read_initial_state(initial_dir, cluster_name)
    '''
    k=0
    for name in files:
        time = name.split("_")
        age.append(str(((converter.to_si(float(time[2][1:-5]) | nbody_system.time)).value_in(units.Myr) + 1 | units.Myr).number))
        current_time.append(time[2][1:-5])
        if k>0:
            elapsed_time.append( str( float(current_time[k])-float(current_time[0]) ) )
        k+=1
    '''
    i=0
    for file in star_files:
        ms = read_set_from_file(file, version=1, close_file="True", format="hdf5")
        comx = track2_x(ms.age[0].value_in(units.Myr))
        comy = track2_y(ms.age[0].value_in(units.Myr))
        print "creating image number "+ str(i)
#    filename = "/home/draco/jthornton/testFresco/"+cluster_name+"_%08d"%(i)
        filename = pic_dir+"/"+cluster_name+"%08d" %(i)
#        os.system("python /home/draco/jthornton/Fresco-master/fresco.py -s "+file+" -o "+filename+" -w 50")
        # I cant just feed it an already read star file, 
        os.system("python /home/draco/jthornton/Fresco/fresco.py -s "+file+" -o "+filename+" --comx "+str(comx)+" --comy "+str(comy)+" --new-ev")
        print "image created"
        i+=1

#for file in files:
#    print "Creating image for time: "+current_time[i]
#    filename = "/home/draco/jthornton/FrescoPictures/"+cluster_name+"/"+cluster_name+"_"+current_time[i]

