import os
import glob
from amuse.units import nbody_system
from amuse.units import units
from tycho import read


initial_dir = "/home/draco/jthornton/Tycho/InitialState"
runs_dir = "/home/draco/jthornton/Tycho/Run/MasterParticleSet/"
cluster_name = "Josh8"

print runs_dir
print cluster_name

files = glob.glob(runs_dir+cluster_name+"*.hdf5")
age = "8" # In units of Myr
# Gather the time steps from the file names and save them in an array

ms, ic, converter = read.read_initial_state(cluster_name)

current_time = []
age = []

for name in files:
    time = name.split("_")
    age.append(str(((converter.to_si(float(time[2][1:-5]) | nbody_system.time)).value_in(units.Myr) + 1 | units.Myr).number))
    current_time.append(time[2][1:-5])

print age[0]

print current_time[0]
# Loop through the hdf5 files to make the Fresco Pictures

i=0

#filename = "/home/draco/jthornton/FrescoPictures/"+cluster_name+"_"+current_time[0]
#os.system("python /home/draco/jthornton/Fresco-master/fresco.py -s"+" "+files[0]+ " -o"+" "+filename+" -a "+age[0])

for file in files:
    print "creating image for time:"+current_time[i]
    filename = "/home/draco/jthornton/FrescoPictures/Josh2/"+cluster_name+"_"+current_time[i]
    os.system("python /home/draco/jthornton/Fresco-master/fresco.py -s "+file+" -o "+filename+" -a "+age[i])
    print "image created"
    i+=1

