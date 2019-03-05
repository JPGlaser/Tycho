#------------------------------------------------------------------------------#
#------------------------------plot_encounters.py------------------------------#
#------------------------------------------------------------------------------#
#--------------------------Created by Mark Giovinazzi--------------------------#
#------------------------------------------------------------------------------#
#-This program was created for the purpose of loading in saved encounter files-#
#-to plot our encounters in a more general sense and will also generate movies-#
#------------------------------------------------------------------------------#
#---------------------------Date Created: 12/??/2017---------------------------#
#------------------------Date Last Modified: 05/02/2018------------------------#
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#-------------------------------Import Libraries-------------------------------#
#------------------------------------------------------------------------------#

import matplotlib; matplotlib.use('agg')
import numpy as np, matplotlib.pyplot as plt
import os, sys, glob, time, pickle
from amuse.lab import *
from amuse import datamodel
from amuse.community.kepler.interface import kepler
#from sim_encounters import get_stars, get_planets
import matplotlib.patches as mpatches

# get the stars
def get_stars(bodies):

    limiting_mass_for_planets = 13 | units.MJupiter

    stars = bodies[bodies.mass > limiting_mass_for_planets]

    return stars

def get_planets(bodies):

    limiting_mass_for_planets = 13 | units.MJupiter

    planets = bodies[bodies.mass <= limiting_mass_for_planets]

    return planets

#------------------------------------------------------------------------------#
#-The following function will handle all of this project's visualization tools-#
#------------------------------------------------------------------------------#
def generate_visuals(bodies, time_stamps, path, name, make_movie = False, make_map = True):
    ms = 2
    # For labeling purposes, we should determine if there are multiple planets
    if num_planets == 1: plural = ''
    else: plural = 's'
    num_bodies = len(bodies[0][0])
    #print num_bodies
    # xs -> x-values; ys -> y-values; zs -> z-values; cs -> color-values; ss -> size-values
    xs = [];          ys = [];        zs = [];        cs = [];            ss = []
    # loop through and populate wach list with the proper values
    for i in range(len(bodies)):
        #print bodies[i]
        xs.append(bodies[i][0])#.flatten())#.ravel()
        ys.append(bodies[i][1])#.ravel())#.ravel()
        zs.append(bodies[i][2])#.flatten())#.ravel()
        cs.append(bodies[i][3])#.ravel())#.ravel()
        ss.append(bodies[i][4])#.ravel())#.ravel()
    #print xs#.ravel()
    # Now we can determine the min and max values for our x-y plots (for consistency)
    xs_min = np.min([[float(xs[i][j]) for j in range(len(xs[i]))]  for i in range(len(xs))])
    xs_max = np.max([[float(xs[i][j]) for j in range(len(xs[i]))]  for i in range(len(xs))])
    ys_min = np.min([[float(ys[i][j]) for j in range(len(ys[i]))]  for i in range(len(ys))])
    ys_max = np.max([[float(ys[i][j]) for j in range(len(ys[i]))]  for i in range(len(ys))])
    zs_min = np.min([[float(zs[i][j]) for j in range(len(zs[i]))]  for i in range(len(zs))])
    zs_max = np.max([[float(zs[i][j]) for j in range(len(zs[i]))]  for i in range(len(zs))])
    #print xs_min
    # We want all axes to share the same horizontal and vertival max and min; find them now
    max, min = np.max([xs_max, ys_max, zs_max]), np.min([xs_min, ys_min, zs_min])

    # make some quick visualization adjustments to provide some blackspace on the edges
    min -= 0.1 * np.abs(min)
    max += 0.1 * np.abs(max)
   # print min, max
    #print bodies#[0][0], bodies[0][2]
    #print '----------'
    #xs, ys, zs, cs, ss = np.array(xs[0]), np.array(ys[0]), np.array(zs[0]), np.array(cs[0]), np.array(ss[0])
    xs, ys, zs, cs, ss = np.array(np.array(xs).flatten()), np.array(np.array(ys).flatten()), np.array(np.array(zs).flatten()), np.array(np.array(cs).flatten()), np.array(np.array(ss).flatten()).astype(float)
    xs, ys, zs = map(float, xs), map(float, ys), map(float, zs)
    # If the user wants to make trail maps in their run, the following code will be executed
    if make_map == True:

        plt.figure(figsize = (5.5, 5))
        #print xs[0], zs[0], cs[0], ss[0]
        # x by z plot (top left corner)
        plt.subplot(221, aspect = 'equal')
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.set_xticklabels([])
        #print range(len(xs))
        #print [(xs[i], zs[i], cs[i], ss[i]) for i in range(len(xs))]
        #[plt.scatter(float(xs[i]), float(zs[i]), c = cs[i], s = int(ss[i])) for i in range(len(xs))]
        #print '---------------', xs, zs
        #print xs[:12]
        plt.scatter(xs, zs, c = cs, s = ss)
        plt.scatter(xs[:num_bodies], zs[:num_bodies], marker = 'x',  s = ms, c = ['black'] * num_bodies)
        plt.ylabel('Z [AU]')
        plt.xlim(min, max)
        plt.ylim(min, max)
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        # x by y plot (lower left corner)
        plt.subplot(223, aspect = 'equal')
        ax = plt.gca()
        ax.set_facecolor('black')
        plt.scatter(xs, ys, c = cs, s = ss)
        #[plt.scatter(float(xs[i]), float(ys[i]), c = cs[i], s = int(ss[i])) for i in range(len(xs))]
        plt.scatter(xs[:num_bodies], ys[:num_bodies], marker = 'x', s = ms,  c = ['black'] * num_bodies)
        plt.xlabel('X [AU]')
        plt.ylabel('Y [AU]')
        plt.xlim(min, max)
        plt.ylim(min, max)
        #ax.set_xticklabels([min, max])
        #ax.set_yticklabels([min, max])
        #ax.set_xicks([min, max])
        #ax.set_yicks([min, max])
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        # z by y plot (lower right corner)
        plt.subplot(224, aspect = 'equal')
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.set_yticklabels([])
        #[plt.scatter(float(zs[i]), float(ys[i]), c = cs[i], s = int(ss[i])) for i in range(len(xs))]
        plt.scatter(zs, ys, c = cs, s = ss)
        plt.scatter(zs[:num_bodies], ys[:num_bodies], marker = 'x', s = ms, c = ['black'] * num_bodies)
        plt.xlabel('Z [AU]')
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1475, right=0.95, hspace=0.25, wspace=0.01)
        #plt.subplots_adjust(wspace = 0.001) # shorten the width between left and right since there aren't tick marks
        plt.xlim(min, max)
        plt.ylim(min, max)
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        star_color = mpatches.Patch(color='yellow', label='Stars')
        earth_color = mpatches.Patch(color='blue', label='Earth(s)')
        jupiter_color = mpatches.Patch(color='orange', label='Jupiter(s)')
        neptune_color = mpatches.Patch(color='green', label='Neptune(s)')
        leg = plt.legend(handles=[star_color, earth_color, jupiter_color, neptune_color], loc='upper right', bbox_to_anchor=(0.9, 2))#loc = (1.5, 1))#, loc = 'upper right')
        leg.get_frame().set_edgecolor('white')

        # make some plot_radius function to determine value for s for stars and planets...
        # stars and planets should look noticeably different in size, but stars/stars and planets/planets should be subtle
        plt.suptitle(str(time_stamps[-1]) + ' yr Simulation (' + str(num_stars) + ' Stars, ' + str(num_planets) + ' Planet' + plural + ')')
        print 'saving///' # just for the user's sake..
        plt.savefig('trail_maps/' + str(name) + '.pdf', dpi = 1000)
        plt.savefig('movies/' + name + '_00000.png', dpi = 300)
        plt.clf()



    # If the user wants to make movies in their run, the following code will be executed
    if make_movie == True:
        #print xs, zs, range(len(bodies))
        # Now, let's make this encounter's movie
        for snapshot in range(len(bodies)):

            s_min, s_max = num_bodies*snapshot, num_bodies*(snapshot+1)
            print s_max
            plt.figure(figsize = (5.5, 5))

            # x by z plot (top left corner)
            plt.subplot(221, aspect = 'equal')
            ax = plt.gca()
            ax.set_facecolor('black')
            ax.set_xticklabels([])
            plt.scatter(xs[s_min:s_max], zs[s_min:s_max], c = cs[s_min:s_max], s = ss[s_min:s_max])
            plt.ylabel('Z (AU)')

            #plt.ylim(-200, 200)

            plt.xlim(min, max)
            plt.ylim(min, max)
            plt.xlim(-200, 200)
            plt.ylim(-200, 200)


#            star_color = mpatches.Patch(color='yellow', label='Stars')
 #           earth_color = mpatches.Patch(color='blue', label='Earth(s)')
  #          jupiter_color = mpatches.Patch(color='orange', label='Jupiter(s)')
   #         neptune_color = mpatches.Patch(color='green', label='Neptune(s)')
                  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=[star_color, earth_color, jupiter_color, neptune_color])
    #        leg = plt.legend(handles=[star_color, earth_color, jupiter_color, neptune_color], loc='upper right', bbox_to_anchor=(0.9, 2))#loc = (1.5, 1))#, loc = 'upper right')
     #       leg.get_frame().set_edgecolor('white')

            # x by y plot (lower left corner)
            plt.subplot(223, aspect = 'equal')
            ax = plt.gca()
            ax.set_facecolor('black')
            plt.scatter(xs[s_min:s_max], ys[s_min:s_max], c = cs[s_min:s_max], s = ss[s_min:s_max])
            plt.xlabel('X (AU)')
            plt.ylabel('Y (AU)')

            plt.xlim(min, max)
            plt.ylim(min, max)
            plt.xlim(-200, 200)
            plt.ylim(-200, 200)

            # z by y plot (lower right corner)
            plt.subplot(224, aspect = 'equal')
            ax = plt.gca()
            ax.set_facecolor('black')
            ax.set_yticklabels([])
            plt.scatter(zs[s_min:s_max], ys[s_min:s_max], c = cs[s_min:s_max], s = ss[s_min:s_max])
            plt.xlabel('Z (AU)')
            plt.xlim(min, max)
            plt.ylim(min, max)

            plt.xlim(-200, 200)
            plt.ylim(-200, 200)
#            plt.figure().legend(handles=[star_color, earth_color, jupiter_color, neptune_color], 'upper right')#(l3, l4), ('Line 3', 'Line 4'), 'upper right')
            #star_color = mpatches.Patch(color='yellow', label='Stars')
            #earth_color = mpatches.Patch(color='blue', label='Earth(s)')
            #jupiter_color = mpatches.Patch(color='orange', label='Jupiter(s)')
            #neptune_color = mpatches.Patch(color='green', label='Neptune(s)')
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=[star_color, earth_color, jupiter_color, neptune_color])#, loc = 'upper right')

            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1475, right=0.95, hspace=0.25, wspace=0.01)
            #plt.subplots_adjust(wspace = 0.001) # shorten the width between left and right since there aren't tick marks

            star_color = mpatches.Patch(color='yellow', label='Stars')
            earth_color = mpatches.Patch(color='blue', label='Earth(s)')
            jupiter_color = mpatches.Patch(color='orange', label='Jupiter(s)')
            neptune_color = mpatches.Patch(color='green', label='Neptune(s)')
                  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=[star_color, earth_color, jupiter_color, neptune_color])
            leg = plt.legend(handles=[star_color, earth_color, jupiter_color, neptune_color], loc='upper right', bbox_to_anchor=(0.9, 2))#loc = (1.5, 1))#, loc = 'upper right')
            leg.get_frame().set_edgecolor('white')

            # title the whole figure and save!
            plt.suptitle(str("%.2f" % time_stamps[snapshot]) + ' yr Simulation (' + str(num_stars) + ' Stars, ' + str(num_planets) + ' Planet' + plural + ')')
            plt.savefig('movies/' + name + '_' + str("%05d" % (time_stamps[snapshot] / (time_stamps[1] - time_stamps[0]) + 1)) + '.png', dpi = 300)
            plt.clf()
mage2 -i movies/${name}_${i}_%5d.png -vcodec mpeg4 -b 800k movies/${name}_${i}.mp4
        #subprocess.call("ffmpeg -f image2 -i movies/${name}_${i}_%5d.png -vcodec mpeg4 -b 800k movies/${name}_${i}.mp4")

#------------------------------------------------------------------------------#
#----The following function derives the orbital parameters for the analysis----#
#------------------------------------------------------------------------------#
def get_orbital_parameters(bodies, orbital_dictionary, t):

    # define the stars and planets
    stars, planets = get_stars(bodies), get_planets(bodies)
    num_stars, num_planets = len(stars), len(planets)

    # initialize both kepler and a converter
    converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
    kep = Kepler(unit_converter = converter, redirection = 'none')
    kep.initialize_code()

    for planet in planets:

        # let's define a temporary boolean to identify whether `planet` is bound
        planet_is_bound = False
        p_id = str(planet.id)

        for star in stars:

            s_id = star.id

            total_mass = star.mass + planet.mass
            kep_pos = star.position - planet.position
            kep_vel = star.velocity - planet.velocity
            kep.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])

            # compute the eccentricity for this star-planet pair to check for stability
            a, e = kep.get_elements()

            if t < 10. and e >= 0.4: continue

            # if this is the first iteration, planet MUST be bound to something; don't let it pass without being such
            # we're looking for the details of this planet relative to its host star; if it's not bound, continue
            if e >= 1.: continue

            # if we've made it here, then `planet` must be bound to `star`; let's adjust our boolean
            planet_is_bound = True

            # since we're here, let's also compute some extra values of interest (p, i)
            p = a * (1 - e) # the periapsis
            rel_pos = planet.position - star.position
            rel_vel = planet.velocity - star.velocity
            i_cross = np.cross(rel_pos, rel_vel) / np.linalg.norm(np.cross(rel_pos, rel_vel))# the inclination
            i_cross = [i_cross.number[j].number for j in range(3)]

            break

        if not planet_is_bound:

            s_id = 0 # s_id = 0 indicates that the planet is not bound to any stars, and is thus rogue

            # these are just temporary fixes to help make us aware of when planets drift off
            a = 0 | units.m
            e = 9999
            p = 0 | units.m
            i_cross = [9999, 9999, 9999]
            # now we still need to compute the would-be `a` and `e` values

        if p_id not in orbital_dictionary:

            orbital_dictionary[p_id] = {'s_id': [s_id], 't': [t], 'a': [a], 'e': [e], 'p': [p], 'i': [i_cross]}

        else:

            (orbital_dictionary[p_id]['s_id']).append(s_id)
            (orbital_dictionary[p_id]['t']).append(t)
            (orbital_dictionary[p_id]['a']).append(a)
            (orbital_dictionary[p_id]['e']).append(e)
            (orbital_dictionary[p_id]['p']).append(p)
            (orbital_dictionary[p_id]['i']).append(i_cross)

    kep.stop()

#--------------------------------------
# Here is where we begin the program...
#--------------------------------------

# set up some conditionals to determine what we're going to use this porgram for
gen_vis = True   # if `gen_vis` is True, you will get a trail map for each close encounter
gen_dict = False # if `gen_dict` is True, you will get a dictionary of orbital elements for each close encounter
gen_movie = True # if `gen_movie` is True, you will get snapshots of each close encounter (which you can later stitch together to make a movie)

# Define the path for all of our hdf5 files
par_str = sys.argv[1]
data_path = os.path.join('ce_directory', par_str) #+ par_str # where the data is going to be pulled from
out_path  = 'analysis_dictionaries' # where the dictionaries will be placed
data_dirs = os.listdir(data_path)

nan_counter = 0

# Each data file has its own subdirectory; let's chalk through them in this loop
for data_dir in data_dirs:

    seed_dirs = os.listdir(os.path.join(data_path, data_dir))

    for seed_dir in seed_dirs:

        key_dirs = os.listdir(os.path.join(data_path, data_dir, seed_dir))

        for key_dir in key_dirs:
            print os.path.join(data_path, data_dir, seed_dir, key_dir)
            encounter_dirs = os.listdir(os.path.join(data_path, data_dir, seed_dir, key_dir))

            if len(encounter_dirs) == 0: continue

            # define an empty dictionary here, since all encounters in this key_dir will share one
            orbital_dictionary = {}
            data_name = data_dir + '_' + seed_dir + '_' + key_dir

            if os.path.exists(os.path.join(out_path, data_name + '.pickle')): continue

            # Each encounter has its own subdirectory; let's chalk through them in this loop
            for encounter_dir in encounter_dirs:

                fin_data_path = os.path.join(data_path, data_dir, seed_dir, key_dir, encounter_dir)
                #fin_data_path = 'ce_directory/16/CaptainMarvel_N1000_W6_encounters/s04/238/encounter001'#'ce_directory/13/CaptainMarvel_N1000_W3_encounters/s00/295/encounter001/'#'ce_directory/36/Drax_N1000_W6_encounters/s00/385/encounter001/'#'ce_directory/36/Drax_N1000_W6_encounters/s00/385/encounter001/'#'ce_directory/66/Hulk_N1000_W6_encounters/' + seed_dir + '/' + key_dir + 'encounter001/'#93/Thor_N1000_W3_encounters/s11/102/encounter001/' #workaround for just plotting one file for debugging
                # `snapshots` will be coined as the list of bodies at all time steps within an encounter
                #fin_data_path = 'ce_directory/13/CaptainMarvel_N1000_W3_encounters/s00/295/encounter001/'

                snapshots = os.listdir(fin_data_path)
                print 'There are', len(snapshots), 'snapshots within `' + fin_data_path + '`.'

                # If we compute how many stars and planets there are up top, we can ignore doing so in the next loop
                bodies = read_set_from_file(os.path.join(fin_data_path, snapshots[0]), 'hdf5', close_file = True)

                stars, planets = get_stars(bodies), get_planets(bodies)
                num_stars, num_planets = len(stars), len(planets)

                iteration = 0
                bodies_plot_data = []
                time_stamps = []

                # Each encounter has many individual snapshots saved
                for snapshot in snapshots[:651]:

                    # this will help the user keep track of where the program is at
                    #if iteration % 100. == 0: print iteration, '/', len(snapshots)
                    #iteration += 1
                    print os.path.join(fin_data_path, snapshot)
                    # At each new time step, we will need to load in the new data file
                    bodies = read_set_from_file(os.path.join(fin_data_path, snapshot), 'hdf5', close_file = True)
                    if np.isnan(bodies[0].vx.number): nan_counter += 1; print 'qqq'; break#or np.nan in bodies.vy.number or np.nan in bodies.vz.number or np.nan in bodies.x.number or np.nan in bodies.y.number or np.nan in bodies.z.number: break

                    time_stamps.append(bodies.get_timestamp().in_(units.yr).number) # get the list of time steps
                    # if we're making plots, let's set some things up
                    if gen_vis:

                        # The following two lines of code separate bodies into stars and planets
                        stars, planets = get_stars(bodies), get_planets(bodies)
                        stars.color = 'y' # Let's make the stars yellow
                        stars.ms = 5
                        #print stars, planets, stars.id, planets.host_star
                        swp = stars[stars.id ==238]# planets.host_star]
                        swp.color = 'white'
                        #planets.color = 'r' # and the planets red
                        #planets.ms = 5
                        #print stars
                        for planet in planets:

                            if planet.id < 40000.: planet.ms = 1; planet.color = 'blue'
                            elif planet.id < 60000.: planet.ms = 3; planet.color = 'orange'
                            else: planet.ms = 2; planet.color = 'green'

                        # This could be where your code slows down!!!
                        bodies_plot_data.append(np.array([bodies.x.in_(units.AU).number, bodies.y.in_(units.AU).number, bodies.z.in_(units.AU).number, bodies.color, bodies.ms]))
                    if gen_dict:
                        # need to return some particle set each iteration
                        get_orbital_parameters(bodies, orbital_dictionary = orbital_dictionary, t = float(snapshot[-14:-5]))

                # Call our function to generate the trail map and make the movie for this encounter
                if gen_vis:
                    generate_visuals(bodies_plot_data, time_stamps = time_stamps, path = None, name = str(fin_data_path).replace('/', '_').replace('__', '_').replace('_.', '.'), make_movie = gen_movie, make_map = True)

            if gen_dict:
                data_name = fin_data_path.replace('/', '_')#data_dir + '_' + seed_dir + '_' + key_dir
                with open(os.path.join(out_path, data_name + '.pickle'), 'wb') as f:
                    pickle.dump(orbital_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL); f.close()

print 'There are currently', nan_counter, 'files that have turned into nans.'
