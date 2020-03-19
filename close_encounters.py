#------------------------------------------------------------------------------#
#------------------------------close_encounters.py-----------------------------#
#------------------------------------------------------------------------------#
#--------------------------Created by Mark Giovinazzi--------------------------#
#------------------------------------------------------------------------------#
#-This program was created for the purpose of recording parameters for systems-#
#-containing both stars + planets every few iterations during close encounters-#
#------------------------------------------------------------------------------#
#---------------------------Date Created: 11/17/2017---------------------------#
#------------------------Date Last Modified: 01/03/2018------------------------#
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#-------------------------------Import Libraries-------------------------------#
#------------------------------------------------------------------------------#

import matplotlib; matplotlib.use('agg') # first set the proper backend
import sys; sys.path.insert(0, 'GitHub/Tycho/src') # specify path to create.py
#from create import planetary_systems # load planetary_systems in from create.py
from amuse.lab import *
from amuse.plot import *
import numpy as np, matplotlib.pyplot as plt, pickle
import os

# Push everything off so that we can run this job first!
sys.stdout.flush() # not sure if I need to have this?i

#------------------------------------------------------------------------------#
#------The following function supervises a collision system in generality------#
#------------------------------------------------------------------------------#
def run_collision(bodies, t_max, dt, identifier):

    # The follwoing two lines compute how many stars/planets are in each encounter    
    stars = np.argpartition(np.array(bodies.mass.number), -num_stars)[-num_stars:]
    planets = np.argpartition(np.array(bodies.mass.number), -num_planets)[:num_planets]
  
    # Adjust the bodies' positions to center around the origin
    bodies.position = bodies.position - bodies.center_of_mass()

    # First, we'll need a converter for our SmallN gravity code; define it now
    converter = nbody_system.nbody_to_si(bodies[stars].mass.sum(), bodies[stars].dynamical_timescale())

    # The following block of code sets up the gravity for our system
    gravity = SmallN(redirection = 'none', convert_nbody = converter)
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.timestep_parameter = 0.1 # this is a standard value
    gravity.particles.add_particles(bodies) # adds bodies to gravity calculations
    gravity.commit_particles()
    channel = gravity.particles.new_channel_to(bodies)
    channel.copy() 

    # It will be easier to set-up a list of all time steps to call during the simulation
    time_steps = np.arange(0. | units.yr, t_max, dt)

    # This loop will iterate over all the time steps we wish to take
    for time_step in time_steps:

        # Define variable for how many iterations have happened for a given encounter
        iteration = str("%06d" % int(1. / dt * time_step))

        # This statement will allow the user to know how much time has passed
        if int(iteration) % 1000 == 0: print(iteration, '/', len(time_steps))

        # Now, we will evolve the system by the next time step
        gravity.evolve_model(time_step)
        
        # Below we update and combine the stars and planets into one particle set
        try:

            gravity.update_particle_set()
            gravity.particles.synchronize_to(bodies)

        except: pass
   
        # Copy over stars and planets 
        channel.copy()
        
        # Delete the child1 and child2 attributes in order to write set to files
        del bodies.child1, bodies.child2       
    
        # The following lines of code save our stars and planets to a file
        file_path = '/home/draco/jglaser/Public/Tycho_Runs/231AAS/SebaTest2/CloseEncounters' # this should ideally be defined globally?
        encounter_path = file_path+'/'+'EncounterID_'+identifier
        if not os.path.exists(encounter_path):
            os.makedirs(encounter_path)
        file_name = 'ce_' + identifier + '_' + str("%09.2f" % time_step.number)
        write_set_to_file(bodies, encounter_path +'/'+ file_name+'.hdf5', 'hdf5')

        # Check to see if the evolution is over 
        over = gravity.is_over() # 1 => it's over, 0 => it's not over

        if over:

            print('it\'s over!') # let the user know that we're wrapping up 

            gravity.update_particle_tree()
            gravity.update_particle_set()
            gravity.particles.synchronize_to(bodies)
            channel.copy()
     
            break # since over was called, 'break' frees us from finishing the loop
        
    # Call for the stoppage of gravity -- hopefully this never happens in real life!
    gravity.stop()

# Define parameters here which will later go at the top
#encounter_database = 'Bullhead_encounters.pkl' # the filename of the database
#encounter_database = 'Omega_encounters.pkl'    # another filename for database
encounter_database = '/home/draco/jglaser/Public/Tycho_Runs/231AAS/SebaTest2/Bullhead_encounters.pkl'

# Load the enounter database into the variable "encounters"
with open(encounter_database, 'rb') as f: encounters = pickle.load(f)

# The following list comprehension will pull all keys which have two stars and one+ planet(s)
# If you would like to be much less specific, simply comment out all of the conditionals
keys = [str(i) for i in range(len(encounters)) 
                   if encounters[str(i)] != []             # make sure encounter happens 
                       and len(encounters[str(i)]) == 1    # only worry about single encounters
                       and len(encounters[str(i)][0]) > 2] # make sure there are > 2 bodies

# Install a list of encounters which meet all of the same criteria as the keys above 
interesting_encounters = [encounters[key][0] for key in keys]

# Or if you'd rather specify one specific key...
#interesting_encounters = encounters['1000020'][0]

# Set up input paramters for our run_collisions function
t_max = 1000000      | units.yr # the time at which to stop the simulation (if not already 'over')
dt    = 100 * 2.**-3 | units.yr # time step to begin iterating by (SmallN may vary this)

# Simulate all of the interesting encounters from t=0 until 'over'
for key in range(len(keys)):

        bodies = interesting_encounters[key]
        num_stars = len([i for i in range(len(bodies)) if bodies[i].radius > 10**10 | units.m])
        num_planets = len(bodies) - num_stars
        print(num_stars, ' stars | ', num_planets, ' planets')       
        run_collision(bodies, t_max, dt, identifier = str("%02d" % key))
