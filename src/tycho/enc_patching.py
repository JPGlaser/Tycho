from tycho import *
from tycho import stellar_systems
from amuse import *
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units

import hashlib

from amuse.community.secularmultiple.interface import SecularMultiple
from amuse.datamodel.trees import BinaryTreesOnAParticleSet
from amuse.ext.orbital_elements import get_orbital_elements_from_binary

set_printing_strategy("custom", preferred_units = [units.MSun, units.AU, units.day, units.deg], precision = 6, prefix = "", separator = "[", suffix = "]")

def get_node_of_leaf(particle_set, chosen_id):
    '''Retrieves the node of the desired leaf particle given its ID.'''
    tree = BinaryTreesOnAParticleSet(particle_set, 'child1', 'child2')
    for root in tree.iter_roots():
        print("Current Root ID:",root.particle.id)
        print("List of all Leaves of the Tree:",root.get_leafs_subset().id)
        if chosen_id in root.get_leafs_subset().id:
            for node in root.iter_inner_nodes():
                if node.child1.id == chosen_id or node.child2.id == chosen_id:
                    print('Chosen Node and its Children:', node.id, node.child1.id, node.child2.id)
                    return node
        else:
            print('!!! Error: Could not find desired leaf in the tree.')
            return None

def get_root_of_leaf(particle_set, chosen_id):
    '''Retrieves the root of the desired leaf particle given its ID.'''
    tree = BinaryTreesOnAParticleSet(particle_set, 'child1', 'child2')
    for root in tree.iter_roots():
        leaves_id = root.get_leafs_subset().id
        if chosen_id in leaves_id:
            return root.particle



def get_physical_radius(particle):
    try:
        radius = particle.physical_radius
        return radius
    except:
        print('No physical Radius Given, Applying Estimate!')
        if particle.mass <= 13 | units.MJupiter:
            if particle.mass <= 0.01 | units.MJupiter:
                return 1 | units.REarth
            elif particle.mass <= 0.1 | units.MJupiter:
                return 4 | units.REarth
            elif particle.mass <= 1 | units.MJupiter:
                return 1 | units.RJupiter
        else:
            return util.get_stellar_radius(particle)



def get_full_hierarchical_structure(bodies, RelativePosition=False):
    hierarchical_set = Particles()
    for body in bodies:
        print(body.id)
        body.radius = get_physical_radius(body)
        print(body.radius)
    # Calculate Distances to All Bodies for Each Body
    for index, body in enumerate(bodies):
        try:
            if body.id in hierarchical_set.id:
                print("Body Already Added:", body.id)
                continue
        except:
            print('First Pass')
            pass
        list_of_sqd = bodies.distances_squared(body).number
        for nonzero_sqd in list_of_sqd[np.nonzero(list_of_sqd)]:
            print("Finding closet Neighbor for ID:", body.id)
            # Assign the Closest Partner
            closest_partner = bodies[[i for i, j in enumerate(list_of_sqd) if j == nonzero_sqd]]
            print("Closest Neighbor is:", closest_partner.id)

            # Check if closest_partner is in a binary already
            #print(index)
            if index != 0:
                node = get_root_of_leaf(hierarchical_set, closest_partner.id)
                if node != None:
                    print('Heirarchical Set when finding nodes:',[x.id for x in hierarchical_set if x.is_binary==True])
                    closest_partner = node
            print("Closest Neighbor is:", closest_partner.id)

            # Calculate the Orbital Elements
            k_set = Particles()
            k_set.add_particle(body.copy())
            k_set.add_particle(closest_partner.copy())
            k_set_sorted = k_set.sorted_by_attribute('mass')[::-1] # Ensures Heaviest is First
            print(k_set.id)
            (mass1, mass2, semimajor_axis, eccentricity, \
             true_anomaly, inclination, long_asc_node, arg_per) \
                = get_orbital_elements_from_binary(k_set_sorted, G=constants.G)
            # Determine if 'body' orbits 'closest_partner'
            if eccentricity < 1.0:
                bigbro = k_set.sorted_by_attribute('mass')[1].copy() #Child1 is Always Heaviest
                lilsis = k_set.sorted_by_attribute('mass')[0].copy() #Child2 is Always Lightest
                # Add in Leaves
                if bigbro not in hierarchical_set:
                    temp = Particle()
                    temp.id = bigbro.id
                    temp.child1 = None
                    temp.child2 = None
                    temp.is_binary = False
                    temp.mass = bigbro.mass
                    temp.radius = bigbro.radius
                    temp.position = bigbro.position
                    temp.velocity = bigbro.velocity
                    print(bigbro.velocity)
                    hierarchical_set.add_particle(temp) # Child1 is at -2
                #elif bigbro in hierarchical_set:
                #    hierarchical_set[]
                if lilsis not in hierarchical_set:
                    temp = Particle()
                    temp.id = lilsis.id
                    temp.child1 = None
                    temp.child2 = None
                    temp.is_binary = False
                    temp.mass = lilsis.mass
                    temp.radius = lilsis.radius
                    temp.position = lilsis.position
                    temp.velocity = lilsis.velocity
                    print(lilsis.velocity)
                    hierarchical_set.add_particle(temp) # Child2 is at -1
                # Reset bigbro and lilsis to the copies in the set
                i1 = np.where(hierarchical_set.id==bigbro.id)[0][0]
                i2 = np.where(hierarchical_set.id==lilsis.id)[0][0]
                bigbro = hierarchical_set[i1]
                lilsis = hierarchical_set[i2]
                # Add in Root
                root_particle = Particle()
                root_particle.mass = bigbro.mass+lilsis.mass
                root_particle.is_binary = True
                root_particle.semimajor_axis = semimajor_axis
                root_particle.radius = 0 | units.RSun
                root_particle.eccentricity = eccentricity
                root_particle.inclination = inclination
                root_particle.argument_of_pericenter = arg_per
                root_particle.longitude_of_ascending_node = long_asc_node
                root_particle.period = 2.0*np.pi/np.sqrt(constants.G*(bigbro.mass))*semimajor_axis**(3./2.)
                root_particle.child1 = bigbro
                root_particle.child2 = lilsis
                root_particle.position = (hierarchical_set.select(lambda x : x == False, ["is_binary"])).center_of_mass()
                root_particle.velocity = (hierarchical_set.select(lambda x : x == False, ["is_binary"])).center_of_mass_velocity()
                root_particle.id = root_particle.child1.id+root_particle.child2.id
                hierarchical_set.add_particle(root_particle)
                break
            continue
    #for particle in hierarchical_set:
    #    if particle.is_binary == True:
    #        print(particle.child1.id, particle.child2.id)
    #    else:
    #        print(particle.id)
    return hierarchical_set.copy()

def map_node_oe_to_lilsis(hierarchical_set):
    for node in hierarchical_set.select(lambda x : x == True, ["is_binary"]):
        lilsis = node.child2
        lilsis.semimajor_axis = node.semimajor_axis
        lilsis.eccentricity = node.eccentricity
        lilsis.inclination = node.inclination
        lilsis.argument_of_pericenter = node.argument_of_pericenter
        lilsis.longitude_of_ascending_node = node.longitude_of_ascending_node
        lilsis.period = node.period

def run_secularmultiple(particle_set, end_time, start_time=(0 |units.Myr), N_output=100, debug_mode=False, genT4System=False, exportData=False):
    try:
        hierarchical_test = [x for x in particle_set if x.is_binary == True]
        print("The supplied set has", len(hierarchical_test), "node particles and is a tree.")
        py_particles = particle_set
    except:
        print("The supplied set is NOT a tree set! Building tree ...")
        py_particles = get_full_hierarchical_structure(particle_set)
        hierarchical_test =  [x for x in py_particles if x.is_binary == True]
        print("Tree has been built with", len(hierarchical_test), "node particles.")
    nodes = py_particles.select(lambda x : x == True, ["is_binary"])
    Num_nodes = len(nodes)

    if exportData:
        plot_a_AU = [[] for x in range(Num_nodes)]
        plot_e = [[] for x in range(Num_nodes)]
        plot_peri_AU = [[] for x in range(Num_nodes)]
        plot_stellar_inc_deg = [[] for x in range(Num_nodes)]
        plot_times_Myr = []

    if genT4System:
        nodes = py_particles.select(lambda x : x == True, ["is_binary"])
        print(nodes.inclination)
        nodes.inclination = [-18.137, 0.0, 23.570] | units.deg
        print(nodes.inclination)

    if debug_mode:
        nodes = py_particles.select(lambda x : x == True, ["is_binary"])
        print('='*50)
        print('t/kyr',0.00)
        print('a/AU', nodes.semimajor_axis)
        print('p/day', nodes.period)
        print('e',nodes.eccentricity)
        print('i/deg', nodes.inclination)
        print('AP/deg', \
            nodes.argument_of_pericenter)
        print('LAN/deg', \
            nodes.longitude_of_ascending_node)

    code = SecularMultiple()
    code.particles.add_particles(py_particles)
    code.model_time(start_time)

    channel_from_particles_to_code = py_particles.new_channel_to(code.particles)
    channel_from_code_to_particles = code.particles.new_channel_to(py_particles)
    channel_from_particles_to_code.copy()

    time = start_time
    output_time_step = end_time/float(N_output)

    if exportData:
        plot_times_Myr.append(time.value_in(units.Myr))
        for i, node in enumerate(nodes):
            plot_a_AU[i].append(node.semimajor_axis.value_in(units.AU))
            plot_e[i].append(node.eccentricity)
            plot_peri_AU[i].append(node.semimajor_axis.value_in(units.AU)*(1.0-node.eccentricity))
            plot_stellar_inc_deg[i].append(node.inclination.value_in(units.deg))

    while time <= end_time:
        time += output_time_step
        code.evolve_model(time)
        print('Evolved model to:', time.value_in(units.Myr), "Myr")
        channel_from_code_to_particles.copy()

        if exportData:
            plot_times_Myr.append(time.value_in(units.Myr))
            for i, node in enumerate(nodes):
                plot_a_AU[i].append(node.semimajor_axis.value_in(units.AU))
                plot_e[i].append(node.eccentricity)
                plot_peri_AU[i].append(node.semimajor_axis.value_in(units.AU)*(1.0-node.eccentricity))
                plot_stellar_inc_deg[i].append(node.inclination.value_in(units.deg))

        if time == end_time+output_time_step:
            map_node_oe_to_lilsis(py_particles)

        if debug_mode:
            if time == end_time or time==output_time_step:
                print('='*50)
                print('t/kyr',time.value_in(units.kyr))
                print('a/AU', nodes.semimajor_axis)
                print('p/day', nodes.period)
                print('e',nodes.eccentricity)
                print('i/deg', nodes.inclination)
                print('AP/deg', \
                    nodes.argument_of_pericenter)
                print('LAN/deg', \
                    nodes.longitude_of_ascending_node)
    code.stop()
    if exportData:
        data = plot_times_Myr,plot_a_AU, plot_e, plot_peri_AU, plot_stellar_inc_deg
        return py_particles, data
    else:
        return py_particles
