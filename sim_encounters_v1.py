#------------------------------------------------------------------------------#
#-------------------------------sim_encounters.py------------------------------#
#------------------------------------------------------------------------------#
#--------------------------Created by Mark Giovinazzi--------------------------#
#------------------------------------------------------------------------------#
#-This program was created for the purpose of recording parameters for systems-#
#-containing both stars + planets every few iterations during close encounters-#
#------------------------------------------------------------------------------#
#---------------------------Date Created: 11/17/2017---------------------------#
#------------------------Date Last Modified: 05/17/2018------------------------#
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#-------------------------------Import Libraries-------------------------------#
#------------------------------------------------------------------------------#
import matplotlib; matplotlib.use('agg') # first set the proper backend
import sys; sys.path.insert(0, 'GitHub/Tycho/src') # specify path to create.py
from tycho.create import planetary_systems_v2 as planetary_systems # load planetary_systems in from create.py
from amuse.lab import *
from amuse.plot import *
from amuse import datamodel
import numpy as np, matplotlib.pyplot as plt, pickle, os, itertools
from amuse.community.kepler.interface import kepler
import time
# Push everything off so that we can run this job first!
sys.stdout.flush() # not sure if I need to have this?

# if you apply this function to a particle set, it will return the stars
def get_stars(bodies):

    # anything with a Mass >= 13 Jupiter masses is a star
    limiting_mass_for_planets = 13 | units.MJupiter
    stars = bodies[bodies.mass > limiting_mass_for_planets]

    return stars

# if you apply this function to a particle set, it will return the planets
def get_planets(bodies):

    # anything with a Mass <= 13 Jupiter masses is a planet
    limiting_mass_for_planets = 13 | units.MJupiter
    planets = bodies[bodies.mass <= limiting_mass_for_planets]

    return planets

#------------------------------------------------------------------------------#
#-The following function returns a list to match planets with their host stars-#
#------------------------------------------------------------------------------#
def get_solar_systems(bodies, converter = None, rel_pos = True):

    # initialize Kepler before doing any orbital calculations!
    kep = Kepler(unit_converter = converter, redirection = 'none')
    kep.initialize_code()

    # let's separate our stars and planets for searching
    stars, planets = get_stars(bodies), get_planets(bodies)
    num_stars, num_planets = len(stars), len(planets)

    systems = []

    for star in stars:

        planetary_system = []
        planetary_system.append(star.id)  # may need .id[0]

        for planet in planets:

            planetary_system.append(planet)
            total_mass = star.mass + planet.mass
            kep_pos = star.position - planet.position
            kep_vel = star.velocity - planet.velocity
            kep.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])

            a, e = kep.get_elements()

            if e >= 1: del planetary_system[-1]

            else:

                # there is a very slim chance that this could be a planet being "called bound" to a member of a two-star system
                # to account for this, let's quickly chedk to make sure that this star isn't bounded to other stars
                # yikes; if this can be true, couldn't other cases? i.e. massive lone star or massive star with planetary syst?
                for other_star in stars - star:

                    kep.initialize_from_dyn(star.mass + other_star.mass, star.x - other_star.x, star.y - other_star.y, star.z - other_star.z,
                                                                         star.vx - other_star.vx, star.vy - other_star.vy, star.vz - other_star.vz)

                    a, e = kep.get_elements()

                    if e < 1:

                        safe_to_add_planets = False
                        del planetary_system[-1]
                        break

                    else: safe_to_add_planets = True

                if safe_to_add_planets and rel_pos:

                    planetary_system[-1].position -= star.position

        if planetary_system: systems.append(planetary_system)

    #To stop Kepler (killing the worker instance), just run:
    kep.stop()

    return systems

#------------------------------------------------------------------------------#
#-The following function checks to see if the encounter is finished simulating-#
#------------------------------------------------------------------------------#
def is_over(solar_systems, bodies):

    # let's separate our stars and planets
    stars, planets = get_stars(bodies), get_planets(bodies)
    num_stars, num_planets = len(stars), len(planets)

    stellar_ids_with_planets = [solar_system[0] for solar_system in solar_systems if len(solar_system) > 1]

    stars_with_planets = [star for star in stars if star.id in stellar_ids_with_planets]
    # now we have the actual stars with planets...let's only worry about seeing if these stars are far enough away

    # must compare these (plural to be general) stars to ALL others in system now.
    stars_are_done = False # so we can keep track of the stars that are not done interacting yet

    for star1 in stars_with_planets:

        for star2 in stars:

            # if these are the same stars, skip!
            if star1 == star2: continue

            rel_position = [star1.x.number - star2.x.number, star1.y.number - star2.y.number, star1.z.number - star2.z.number]
            stellar_distance = np.linalg.norm(rel_position)

            if stellar_distance > star1.radius.number + star2.radius.number: stars_far_apart = True
            else: continue

            rel_velocity = [star1.vx.number - star2.vx.number, star1.vy.number - star2.vy.number, star1.vz.number - star2.vz.number]
            rel_position = [star1.x.number - star2.x.number, star1.y.number - star2.y.number, star1.z.number - star2.z.number]

            if np.dot(rel_velocity, rel_position) > 1: stars_moving_apart = True
            else: continue

            # If we get to here, then the stars must be far apart and moving apart from another

            # Let's actually track which stars this thing is still interacting with
            stars_are_done = True

    if stars_are_done:

        # if this is the case, the stars must be done; now let's check if the planets are!
        planets_are_done = True #False
        for planet1 in planets:
            continue
            for planet2 in planets:

                # if these are the same planets, skip!
                if planet1 == planet2: continue

                # now how can I check the paths of these planets to ensure they are not intersecting?

                # basically, what we want to do is check for planets' stability and that they are not crossing
                # but how?

                #planets_still_interacting.append(planet2)

                planets_are_done = True

    if stars_are_done and planets_are_done: is_over = True
    else: is_over = False

    return is_over

#------------------------------------------------------------------------------#
#------The following function supervises a collision system in generality------#
#------------------------------------------------------------------------------#
def run_collision(bodies, t_max, dt, identifier, path, planet_patching = False, converter = None):

    # Adjust the bodies' positions to center around the origin
    bodies.position -= bodies.center_of_mass()

    # don't know why I'm doing these two lines but they make it work..
    bodies = Particles(particles = bodies)
    if 'child1' in bodies.get_attribute_names_defined_in_store(): del bodies.child1, bodies.child2

    # Set up the gravity; please, please work efficiently...
    gravity = SmallN(redirection = 'none', convert_nbody = converter)
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.allow_full_unperturbed = 0
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

        # Now, we will evolve the system by the next time step
        gravity.evolve_model(time_step) # this line can take a while :///

        # Copy over stars and planets
        channel.copy() # original
        channel.copy_attribute("index_in_code", "id")

        # Delete the child1 and child2 attributes in order to write set to files
        if 'child1' in bodies.get_attribute_names_defined_in_store(): del bodies.child1, bodies.child2

        # The following lines of code save our stars and planets to a file
        dir_name = os.path.join(path, 'encounter' + identifier)
        if not os.path.isdir(dir_name): os.mkdir(dir_name) # make encounter directory if it doens't exist
        file_name = os.path.join(dir_name, str("%09.2f" % time_step.number) + '.hdf5')
        write_set_to_file(bodies.savepoint(time_step), file_name, 'hdf5')

        # Just check if the system `is_over` after every 50 iterations -- worst case it runs a tad long but this is too expensive to run every time
        if int(iteration) % 50. == 0. and time_step.number > 0:

            # Check to see if the evolution is over (verbose = 2 allows for recursive printing of details)
            over = gravity.is_over() # 1 => it's over, 0 => it's not over

            if over:

                print time_step, 'it\'s over!' # let the user know that we're wrapping up

                gravity.update_particle_tree()
                gravity.update_particle_set()
                gravity.particles.synchronize_to(bodies)
                channel.copy()

                break # since over was called, 'break' frees us from finishing the loop

            '''
            # Check to see if the evolution is over
            over = is_over(get_solar_systems(bodies, converter = converter), bodies)

            if over:

                print time_step, ' -- it\'s over! ' # let the user know that we're wrapping up i

                gravity.update_particle_set()
                gravity.particles.synchronize_to(bodies)
                channel.copy()

                break # since over was called, 'break' frees us from finishing the loop
             '''

    # Call for the stoppage of gravity -- hopefully this never happens in real life!
    gravity.stop()

    if planet_patching: return get_solar_systems(bodies, converter = converter)
    else: return None

#------------------------------------------------------------------------------#
#--The following function will install several cuts needed to run the program--#
#------------------------------------------------------------------------------#
def cut_function(bodies, converter = None):

    cut_bodies = bodies.copy()

    clumps = get_solar_systems(cut_bodies, converter = converter, rel_pos = False)

    # if this is ever the case, SKIP; something must have been incorrectly triggered in the database
    if len(cut_bodies) > sum([len(clump) for clump in clumps]):

        cut_instruction = 'skip'
        return cut_instruction

    # First, collect clump 1; this will necessarily have a planetary system
    clump_index = 0

    for clump in clumps:

        # this means we have come across a planetary system; there is always at least one of these
        if len(clump) > 1:

            for i in range(len(clump)):

                if i == 0:

                    clump_1 = cut_bodies[cut_bodies.id == clump[i]]

                else:

                    clump_1.add_particle(cut_bodies[cut_bodies.id == clump[i].id])

            break

        clump_index += 1

    # we've found our first clump, so remove it from the solar systems list
    del clumps[clump_index]

    # Now, collect clump_2; this will effectively be the original bodies minus clump_1
    clump_index = 0

    for clump in clumps:

        # this means we have come across a planetary system; there is always at least one of these
        if len(clump) > 1:

            for i in range(len(clump)):

                if i == 0:

                    clump_2 = cut_bodies[cut_bodies.id == clump[i]]

                else:

                    clump_2.add_particle(cut_bodies[cut_bodies.id == clump[i].id])

        else:

            if clump_index == 0:

                clump_2 = cut_bodies[cut_bodies.id == clump[0]]

            else:

                clump_2.add_particle(cut_bodies[cut_bodies.id == clump[0]])

        clump_index += 1

    # now that we have our clumps, we can begin to compute some things
    mass_1, mass_2 = clump_1.mass.sum(), clump_2.mass.sum()
    total_mass = mass_1 + mass_2 # get total_mass

    # compute some center of mass properties, and well as relative vectors
    cm_pos_1, cm_pos_2 = clump_1.center_of_mass(), clump_2.center_of_mass()
    cm_vel_1, cm_vel_2 = clump_1.center_of_mass_velocity(), clump_2.center_of_mass_velocity()

    total_mass = mass_1 + mass_2 # get total_mass
    rel_pos = cm_pos_1 - cm_pos_2 # get rel_pos
    rel_vel = cm_vel_1 - cm_vel_2 # get rel_vel

    # initialize kepler and compute the periastron; we'll need it
    kep = Kepler(unit_converter = converter, redirection = 'none')
    kep.initialize_code()
    kep.initialize_from_dyn(total_mass, rel_pos[0], rel_pos[1], rel_pos[2], rel_vel[0], rel_vel[1], rel_vel[2])
    p = kep.get_periastron()

    # we will want the ratio between masses of our center of masses
    mass_ratio = mass_2 / mass_1

    # if the periastron is at least this far away, it's ignorable
    ignore_distance = mass_ratio**(1. / 3) * 600 | units.AU

    # if it's this far away, don't worry about anything else; just toss it
    if p > ignore_distance:

        # we're getting out of this function, so stop kepler
        kep.stop()

        cut_instruction = 'skip'
        return cut_instruction

    # make particles in clumps relative to their centers of masses
    for particle in clump_1:

        particle.position -= cm_pos_1
        particle.velocity -= cm_vel_1

    for particle in clump_2:

        particle.position -= cm_pos_2
        particle.velocity -= cm_vel_2

    # if you ever care to make an intermediate distance, these lines will make for a good start
    # if is_bin is True, then the perturbing object is a binary; otherwise, it's a lone star (w/ or w/out planets doesn't matter)
    #if len(clump_2)
    #neptune_distance = np.linalg.norm(clump_1[clump_1.id >= 80000].position - clump_1[clump_1.id < 30000].position).in_(units.AU)
    #clump_distance = np.linalg.norm(rel_pos).in_(units.AU)
    #replace_bin_distance = (p - neptune_distance).in_(units.AU) # if the periastron is at least this far away, we can replace binaries with a stellar point mass

    # some code (largely from dr. mcmillan) written to advance two clumps to a closer distance
    kep.advance_to_radius(ignore_distance)
    x, y, z = kep.get_separation_vector()
    rel_pos_2 = rel_pos.copy()
    rel_pos_2[0], rel_pos_2[1], rel_pos_2[2] = x, y, z
    vx, vy, vz = kep.get_velocity_vector()
    rel_vel_2 = rel_vel.copy()
    rel_vel_2[0], rel_vel_2[1], rel_vel_2[2] = vx, vy, vz

    # now restore the absolute coordinates
    cm_pos_1, cm_pos_2 = -mass_2 * rel_pos_2 / total_mass, mass_1 * rel_pos_2 / total_mass
    cm_vel_1, cm_vel_2 = -mass_2 * rel_vel_2 / total_mass, mass_1 * rel_vel_2 / total_mass

    # make particles in clumps relative to their centers of masses
    for particle in clump_1:

        particle.position += cm_pos_1
        particle.velocity += cm_vel_1

    for particle in clump_2:

        particle.position += cm_pos_2
        particle.velocity += cm_vel_2

    # combine the clumps to reform our bodies; they have now been advanced to a more reasonable distance for simiulating
    combined_clumps = ParticlesSuperset([clump_1, clump_2])

    kep.stop() # before leaving the function, we had best stop kepler

    return combined_clumps

#------------------------------------------------------------------------------#
#----------------------------Begin the main program----------------------------#
#------------------------------------------------------------------------------#

# Set up general input paramters for our run_collisions function
t_max = 1e5  | units.yr # the time at which to stop the simulation (if not already 'over')
dt    = 10. | units.yr # * 2.**-3 | units.yr # time step to begin iterating by (SmallN may vary this)

# Define parameters for locating the encounter databases to load in
par_str = sys.argv[1] # this string is simply for parallelizing, in order to simplify the process for making directories
par_seed = 's' + sys.argv[2] # this is so we can loop over each 100 times
data_files_path = os.path.join('ce_databases',  par_str) # this is the directory in which to find all of our databases
out_files_path  = os.path.join('ce_directory',  par_str) # this is the directory in which to put all of the newly generated data
if not os.path.exists(out_files_path): os.mkdir(out_files_path)
files = os.listdir(data_files_path) # this creates a list of all files in the aforementioned directory

# We will now loop through every data file in our directory
for file in files:

    # We create an individual `file` variable to automatically name each data file we'll be using
    file = os.path.join(data_files_path, file)

    # The data path will serve as the location for us to dump the data we generate with each file
    data_path = os.path.join(out_files_path, file[len(data_files_path) + 1:-4])
    if not os.path.exists(data_path): os.mkdir(data_path)
    data_path = os.path.join(out_files_path, file[len(data_files_path) + 1:-4], par_seed)
    if not os.path.exists(data_path): os.mkdir(data_path)

    # If you're searching through multiple files, this print statement will help keep things organized
    print '--------------------------------------------'
    print file, data_path

    # Load the enounter database into the variable `encounters`
    with open(file, 'rb') as f: encounters = pickle.load(f); f.close()

    # The following lines will pull all keys which have two stars and one+ planet(s)
    keys = []

    for key in encounters:

        if len(encounters[key]) > 1:

            for particle in encounters[key][0]:

                if particle.id >= 50000:

                    for particle in encounters[key][1]:

                        if particle.id >= 50000:

                            keys.append(key)
                            break

    # Make directories which are titled after each stellar id so that one day we can go back and look at encounters by stellar id
    [os.mkdir(os.path.join(data_path, key)) for key in keys if not os.path.exists(os.path.join(data_path, key))]

    # the `multiples` code doesn't include the first encounter as an encounter, so slice all the first encounters off
    encounters = [encounters[key][1:] for key in keys]

    # Determine the total number of close encounters which will occur during the simulation for this file
    num_encounters = len(keys) # but is this true in general, or could some keys have multiple encounters?

    # This list gets populated with star ids which start with planets so that we know never to double count
    stellar_id_box = []

    # we want to keep track of every solar system we find
    solar_systems = []

    # Simulate all of the encounters from t=0 until 'over'
    for key in range(num_encounters):

        # Initialize a numerical encounter label, to be used as, i.e. encounter_${identifier}
        ID = 0

        # for this key, update the data path to allow for the creation of directories named after stella ids
        data_path = os.path.join(out_files_path, file[len(data_files_path) + 1:-4], par_seed, keys[key])

        # the number of times we need to patch planets; 1 if no patch needed
        num_patches = len(encounters[key])

        for patch in range(num_patches):

            # Immediately determine if we will need to patch planets or not
            if num_patches - patch > 1: to_patch = True
            else: to_patch = False

            ID += 1

            identifier = str(("%0" + str(len(str(num_encounters))) + "d") % ID)
            encounter_path = os.path.join(data_path, 'encounter' + str(("%0" + str(len(str(num_encounters))) + "d") % (int(identifier))))

            # get the bodies and create oarticle sets for their stars and planets
            bodies = encounters[key][patch]
            stars, planets = get_stars(bodies), get_planets(bodies)
            num_stars, num_planets = len(stars), len(planets)

            # If there are no planets in this encounter, why would you simulate it??? ...move on
            if num_planets == 0: continue

            # the following lines couple an Earth-Neptune to a system which initially only has a Jupiter
            converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
            kep = Kepler(unit_converter = converter, redirection = 'none')
            kep.initialize_code()

            # Initially, let's assume we will not need to add anymore planets (Earths, Neptunes) to our system
            adding_planets = False

            # however, we do still need to check this
            for star in stars:

                # If this star is in our `stellar_id_box`, then we should skip it
                if star.id in stellar_id_box: continue

                for planet in planets:

                    total_mass = star.mass + planet.mass
                    kep_pos = star.position - planet.position
                    kep_vel = star.velocity - planet.velocity
                    kep.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])
                    temp_planet_string = planet.id - 50000

                    # compute the eccentricity for this star-planet pair to check for stability
                    a, e = kep.get_elements()

                    # if the eccentricity between this star-planet pair is < 1, then they are gravitationally bound
                    if e < 1:

                        # We have found a host star; let's remove its Jupiter and add a fresh planetary system
                        planetary_system = planetary_systems(stars = [star], num_systems = 1, Earth = True, Jupiter = True, Neptune = True)

                        # At this point, we have added planets, so let's flag this as True
                        adding_planets = True

                        # Adjust the ids of the planets in our `temp_bodies` for future bookkeeping
                        planetary_system.id += temp_planet_string

                        # a note about this: temp_planet string is just going to range from 1 - `the number or encounters` so as to label every planet uniquely
                        # it may be smart to switch `temp_planet_string` with `star.id` that way an Earth orbiting Star 96 would be labeled as 30096
                        # and a Jupiter orbiting Star 235 would be labeled as 50235.

                        # Now that we have updated the new planets' ids, add them to our bodies
                        bodies.add_particles(planetary_system)

                        # If we've made it here, this host star should be documented so that we know about its past encounters
                        stellar_id_box.append(star.id) # possibly add [0] to this

            #To stop Kepler (killing the worker instance), just run:
            kep.stop()

            # If we have added planets, then we will want to append our temp_bodies
            if adding_planets:

                bodies.remove_particles(planets)

                # if we're here, then our converter has also changed; update it now.
                converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)

            # We have now possibly turned stars with Jupiters to stars with three planets, so let's update our number of planets, as well as star/planet indices
            stars, planets = get_stars(bodies), get_planets(bodies)
            num_stars, num_planets = len(stars), len(planets)

            # This is where the planet patching actually happens
            if patch > 0: # in an ideal world, there would be a second condition since we may not need to patch i.e. in case of cutting (function) thus eliminating encounter 01 and skipping

                for star in stars:

                    for stellar_id in stellar_id_box:

                        if star.id == stellar_id:

                            for systems in solar_systems:

                                for system in systems:

                                    if system[0] == star.id and len(system) > 1:

                                        # at this point it is safe to remove the planet(s) attached to this star before we patch in the new ones(s)
                                        bodies.remove_particles(planets)

                                        for planet in system[1:]:

                                            planet.position += star.position#.x += star.x
                                            bodies.add_particle(planet)

                                        # if we make it here, we've added a particle; update the converter
                                        converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)

            # we have updated the bodies, so let's make sure we're accounting for them all
            stars, planets = get_stars(bodies), get_planets(bodies)
            num_stars, num_planets = len(stars), len(planets)

            # here, "cutting" means removing close encounters that aren't actually very close; we redefine what ie means to be "close" in the `cut_function`
            cut_instruction = cut_function(bodies, converter = converter)

            if cut_instruction == 'skip':

                # we need to be careful when skipping; we want to still include the solar system; do that now
                if to_patch: solar_systems.append(get_solar_systems(bodies, converter = converter))

                # lower the ID by one; since we're skipping encounter N, we want to label the next one as encounter N, rather than encounter N+1
                ID -= 1

                continue

            else: bodies = cut_instruction

            # after cutting data and advancing the distance, it's possible we may need to update our converter?
            converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)

            # run the close encounter function!
            solar_systems.append(run_collision(bodies, t_max, dt, identifier = identifier, path = data_path, planet_patching = to_patch, converter = converter))

            # the `run_collision` function above may return None...if it did, let's remove it
            del solar_systems[solar_systems is None]

            # This is where the secular evolution function should be added
            ''' i.e.

            secular_evolution(bodies, t_max)

            `bodies` will need to be the most recent planetary system simulated
            `t_max` will be the time between Star X's next encounter and the time that Star X completed its last encounter

            ^^^ it will require some thought as to what the best way to get those two variables will be...

            `secular_evolution` will basically simulate until either
                a) the star's next encounter is going to begin
                b) the conditions of planetary stability (decide these! or use AMUSE's function) have been met

            '''
