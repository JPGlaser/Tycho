from tycho import *
from tycho import stellar_systems
from amuse import *
import numpy as np
import matplotlib.pyplot as plt

import hashlib

from collections import defaultdict

# Import the Amuse Base Packages
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.datamodel import particle_attributes
from amuse.io import *

from amuse.community.secularmultiple.interface import SecularMultiple
from amuse.datamodel.trees import BinaryTreesOnAParticleSet
from amuse.ext.orbital_elements import get_orbital_elements_from_binary
from amuse.community.sse.interface import SSE

from amuse.support.console import (
    set_printing_strategy, get_current_printing_strategy,
)

from amuse.datamodel import (
    particle_attributes, Particle, Particles, ParticlesSuperset, Grid,
)

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



def get_physical_radius(particle, SEVCode=None):
    '''
    This is a very basic function which pulls an estimate for the radius of
    a planet for use in Secular integrators.
    '''
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
            else:
                #print(particle.id, particle.mass.value_in(units.MJupiter), "MJupiter")
                return 1 | units.RJupiter
        else:
            return util.get_stellar_radius(particle, SEVCode)



def get_full_hierarchical_structure(bodies, RelativePosition=False, \
                                    KeySystemID=None, SEVCode = None):
    '''
    This function creates a tree-based particle set for use in hierarchical
    particle integrators (like SecularMultiple).
    '''
    hierarchical_set = Particles()
    for body in bodies:
        #print(body.id)
        body.radius = get_physical_radius(body, SEVCode=SEVCode)
        #print(body.radius)
    # If Desired, Center the Structure on a Specific Key Body's ID
    if KeySystemID!= None:
        # We do this by sorting the next loop to advance in ascending order
        # according to distance from the designated KeyBody.
        KeyBody = bodies[bodies.id == KeySystemID]
        for i, dist in enumerate(bodies.distances_squared(KeyBody)):
            bodies[i].distsqd_KeyCenter = dist[0].number | dist[0].unit
        bodies = bodies.sorted_by_attribute("distsqd_KeyCenter")

    for index, body in enumerate(bodies):
        try:
            if body.id in hierarchical_set.id:
                print("Body Already Added:", body.id)
                continue
        except:
            print('First Pass')
            pass
        list_of_sqd = bodies.distances_squared(body).number
        for nonzero_sqd in sorted(list_of_sqd[np.nonzero(list_of_sqd)]):
            print("Finding closet Neighbor for ID:", body.id)
            # Assign the Closest Partner
            closest_partner = bodies[[i for i, j in enumerate(list_of_sqd) if j == nonzero_sqd]]
            print("Closest Neighbor is:", closest_partner.id)

            # Check if closest_partner is in a binary already
            if index != 0:
                node = get_root_of_leaf(hierarchical_set, closest_partner.id)
                if node != None:
                    print('Heirarchical Set when finding nodes:',[x.id for x in hierarchical_set if x.is_binary==True])
                    closest_partner = node
            print("New Closest Neighbor is:", closest_partner.id)

            # Calculate the Orbital Elements
            k_set = Particles()
            k_set.add_particle(body.copy())
            k_set.add_particle(closest_partner.copy())
            k_set_sorted = k_set.sorted_by_attribute('mass')[::-1] # Ensures Heaviest is First
            #print(k_set.id)
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
                    #print(bigbro.velocity)
                    hierarchical_set.add_particle(temp) # Child1 is at -2
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
                    #print(lilsis.velocity)
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
                root_particle.position = (hierarchical_set.select(lambda x : x == bigbro.id or x == lilsis.id, ["id"])).center_of_mass()
                root_particle.velocity = (hierarchical_set.select(lambda x : x == bigbro.id or x == lilsis.id, ["id"])).center_of_mass_velocity()
                root_particle.id = root_particle.child1.id+root_particle.child2.id
                hierarchical_set.add_particle(root_particle)
                break
            continue
    return hierarchical_set.copy()

def check_for_stellar_collision(hierarchical_set, KeySystemID=None, SEVCode=None):
    '''
    This function checks for a planet entering into the Roche Limit
    of its parent star. This is meant to be used as a check for Secular
    orbit integrators.
    '''
    map_node_oe_to_lilsis(hierarchical_set)
    children_particles = hierarchical_set.select(lambda x : x == False, ["is_binary"])
    host_star = util.get_stars(children_particles)[0]
    #print(host_star)
    planets = util.get_planets(children_particles)
    for planet in planets:
        perihelion = planet.semimajor_axis*(1.0-planet.eccentricity)
        roche_limit = 2.46*planet.radius*(host_star.mass/planet.mass)**(1/3.0)
        #print("Perihelion:", perihelion, "| Roche Limit:", roche_limit)
        if perihelion <= roche_limit:
            # Host Star Gains Mass of Planet
            host_star.mass += planet.mass
            # Planet is removed from the set
            planets.remove_particle(planet)
            # Temporary Particle Set is Created with just Star and Planets
            temp = Particles()
            temp.add_particle(host_star)
            temp.add_particles(planets)
            # Update Position and Velocity Vectors
            temp = update_posvel_from_oe(temp)
            # Hierarchy is Rebuilt and Returned
            return get_full_hierarchical_structure(temp, KeySystemID=KeySystemID, SEVCode=SEVCode)
    return None

def update_posvel_from_oe(particle_set):
    '''
    This function generates position and velocity Vector
    quantities for particles given their orbital elements.
    This is meant to be used with SecularMultiple.
    '''
    host_star = util.get_stars(particle_set)
    planets = util.get_planets(particle_set)
    host_star.position = [0.0, 0.0, 0.0] | units.AU
    host_star.velocity = [0.0, 0.0, 0.0] | units.kms
    temp = Particles()
    temp.add_particle(host_star)
    for planet in planets:
        nbody_PlanetStarPair = \
        new_binary_from_orbital_elements(host_star.mass, planet.mass, planet.semimajor_axis, G=units.constants.G, \
                                         eccentricity = planet.eccentricity, inclination=planet.inclination, \
                                         longitude_of_the_ascending_node=planet.longitude_of_ascending_node, \
                                         argument_of_periapsis=planet.argument_of_pericenter, \
                                         true_anomaly = 360*rp.uniform(0.0,1.0) | units.deg) # random point in the orbit
        planet.position = nbody_PlanetStarPair[1].position
        planet.velocity = nbody_PlanetStarPair[1].velocity
        temp.add_particle(planet)
    temp[0].position += particle_set.center_of_mass()
    temp[0].velocity += particle_set.center_of_mass_velocity()
    return temp

def map_node_oe_to_lilsis(hierarchical_set):
    ''' Maps Nodes' orbital elements to their child2 (lilsis) particle. '''
    for node in hierarchical_set.select(lambda x : x == True, ["is_binary"]):
        lilsis = node.child2
        lilsis.semimajor_axis = node.semimajor_axis
        lilsis.eccentricity = node.eccentricity
        lilsis.inclination = node.inclination
        lilsis.argument_of_pericenter = node.argument_of_pericenter
        lilsis.longitude_of_ascending_node = node.longitude_of_ascending_node
        lilsis.period = node.period

def reset_secularmultiples(code):
    '''Useful function to reset the SecularMultiple integrator.
       Saves on CPU overhead.'''
    unit_l = units.AU
    unit_m = units.MSun
    unit_t = 1.0e6*units.yr
    unit_e = unit_m*unit_l**2/(unit_t**2) ### energy
    code.particles = Particles()
    code.model_time = 0.0 | units.Myr
    code.particles_committed = False
    code.initial_hamiltonian = 0.0 | unit_e
    code.hamiltonian = 0.0 | unit_e
    code.flag = 0
    code.error_code = 0
    return code

def get_jovian_parent(hierarchical_set):
    '''Gets the Node which contains the Jovian'''
    nodes = hierarchical_set.select(lambda x : x == True, ["is_binary"])
    return [x for x in nodes if np.floor(x.child2.id/10000) == 5][0]

def initialize_PlanetarySystem_from_HierarchicalSet(hierarchical_set):
    '''Initializes a PlanetarySystem class for AMD calculations.'''
    map_node_oe_to_lilsis(hierarchical_set)
    children_particles = hierarchical_set.select(lambda x : x == False, ["is_binary"])
    host_star = util.get_stars(children_particles)[0]
    #print(host_star)
    planets = util.get_planets(children_particles)
    PS = stellar_systems.PlanetarySystem(host_star, planets, system_name=str(host_star.id))
    PS.get_SystemBetaValues()
    return PS

def update_oe_for_PlanetarySystem(PS, hierarchical_set):
    '''Updates the orbital elements from a hierarchical set to
       an existant PlanetarySystem class.'''
    map_node_oe_to_lilsis(hierarchical_set)
    for planet in PS.planets:
        matched_planet = hierarchical_set.select(lambda x : x == planet.id, ["id"])
        #print(planet.id)
        #print(matched_planet.mass)
        planet.semimajor_axis = matched_planet.semimajor_axis
        planet.eccentricity = matched_planet.eccentricity
        planet.period = matched_planet.period
        planet.z_inc = matched_planet.inclination
    PS.planets = PS.planets.sorted_by_attribute('semimajor_axis')
    stellar_systems.get_rel_inclination(PS.planets)
    return PS

def run_secularmultiple(particle_set, end_time, start_time=(0 |units.Myr), \
                        N_output=100, debug_mode=False, genT4System=False, \
                        exportData=True, useAMD=True, GCode = None, \
                        KeySystemID=None, SEVCode=None):
    '''Does what it says on the tin.'''
    try:
        hierarchical_test = [x for x in particle_set if x.is_binary == True]
        print("The supplied set has", len(hierarchical_test), "node particles and is a tree.")
        py_particles = particle_set
    except:
        print("The supplied set is NOT a tree set! Building tree ...")
        py_particles = get_full_hierarchical_structure(particle_set, KeySystemID=KeySystemID, SEVCode=SEVCode)
        hierarchical_test =  [x for x in py_particles if x.is_binary == True]
        print("Tree has been built with", len(hierarchical_test), "node particles.")
    nodes = py_particles.select(lambda x : x == True, ["is_binary"])
    Num_nodes = len(nodes)
    stellarCollisionOccured = False

    if GCode == None:
        code = SecularMultiple()
    else:
        code = GCode

    if exportData:
        plot_a_AU = defaultdict(list)
        plot_e = defaultdict(list)
        plot_peri_AU = defaultdict(list)
        plot_stellar_inc_deg = defaultdict(list)
        if useAMD:
            plot_AMDBeta = defaultdict(list)
        plot_times_Myr = []

    if genT4System:
        print(nodes.inclination)
        nodes.inclination = [-18.137, 0.0, 23.570] | units.deg
        print(nodes.inclination)

    if debug_mode:
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
    #print(py_particles)
    code.particles.add_particles(py_particles)
    #code.commit_particles()
    #print(code.particles.semimajor_axis)

    code.model_time = start_time
    #print(py_particles.id, py_particles.semimajor_axis)
    channel_from_particles_to_code = py_particles.new_channel_to(code.particles)
    channel_from_code_to_particles = code.particles.new_channel_to(py_particles)
    #print(py_particles.id, py_particles.semimajor_axis)

    channel_from_particles_to_code.copy() #copy_attributes(['semimajor_axis', 'eccentricity', \
            #'longitude_of_ascending_node', 'argument_of_pericenter', 'inclination'])
    #print('This is After the First Channel Copy:', code.particles.semimajor_axis)
    time = start_time
    if useAMD:
        #print(py_particles.id, py_particles.mass)
        jovianParent = get_jovian_parent(py_particles)
        output_time_step = 1000*jovianParent.period.value_in(units.Myr) | units.Myr
        PS = initialize_PlanetarySystem_from_HierarchicalSet(py_particles)
        PS.get_SystemBetaValues()
        if exportData:
            for planet in PS.planets:
                plot_AMDBeta[planet.id].append(planet.AMDBeta)
    else:
        output_time_step = end_time/float(N_output)

    if exportData:
        plot_times_Myr.append(time.value_in(units.Myr))
        for i, node in enumerate(nodes):
            plot_a_AU[node.child2.id].append(node.semimajor_axis.value_in(units.AU))
            plot_e[node.child2.id].append(node.eccentricity)
            plot_peri_AU[node.child2.id].append(node.semimajor_axis.value_in(units.AU)*(1.0-node.eccentricity))
            plot_stellar_inc_deg[node.child2.id].append(node.inclination.value_in(units.deg))
    counter = 0
    while time <= end_time:
        #print('Start of Time Loop')
        #print(output_time_step)
        time += output_time_step
        counter += 1
        #print(time)
        #print(code.model_time)
        #print(code.particles.semimajor_axis)
        code.evolve_model(time)
        #print('Evolved model to:', time.value_in(units.Myr), "Myr")
        #print(code.particles.semimajor_axis)
        channel_from_code_to_particles.copy()
        #channel_from_code_to_particles.copy_attributes(['semimajor_axis', 'eccentricity', \
            #'longitude_of_ascending_node', 'argument_of_pericenter', 'inclination'])
        #print('Hello')
        py_particles.time = time
        if exportData:
            plot_times_Myr.append(time.value_in(units.Myr))
            nodes = py_particles.select(lambda x : x == True, ["is_binary"])
            for i, node in enumerate(nodes):
                plot_a_AU[node.child2.id].append(node.semimajor_axis.value_in(units.AU))
                plot_e[node.child2.id].append(node.eccentricity)
                plot_peri_AU[node.child2.id].append(node.semimajor_axis.value_in(units.AU)*(1.0-node.eccentricity))
                plot_stellar_inc_deg[node.child2.id].append(node.inclination.value_in(units.deg))

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
        # Check for Planet Destruction from Star
        #print(py_particles.id, py_particles.semimajor_axis)
        temp = check_for_stellar_collision(py_particles)
        #print(temp)
        # Returns 'None' if No Destruction!
        if temp != None:
            code.stop()
            code = SecularMultiple()
            code.model_time = time
            py_particles = Particles()
            py_particles.add_particles(temp)
            code.particles.add_particles(py_particles)
            py_particles.time = time
            #code.commit_particles()
            channel_from_particles_to_code = py_particles.new_channel_to(code.particles)
            channel_from_code_to_particles = code.particles.new_channel_to(py_particles)
            channel_from_particles_to_code.copy()
            nodes = py_particles.select(lambda x : x == True, ["is_binary"])
            stellarCollisionOccured = True
            if useAMD:
                PS = initialize_PlanetarySystem_from_HierarchicalSet(py_particles)
            #channel_from_code_to_particles.copy_attributes(['semimajor_axis', 'eccentricity', \
            #'longitude_of_ascending_node', 'argument_of_pericenter', 'inclination'])
        #print(code.particles.semimajor_axis)
        # AMD Checking
        if useAMD:
            PS = update_oe_for_PlanetarySystem(PS, py_particles)
            PS.get_SystemBetaValues()
            if exportData:
                for planet in PS.planets:
                    plot_AMDBeta[planet.id].append(planet.AMDBeta)
            if counter%100==0 and len(PS.planets.select(lambda x : x < 1.0, ["AMDBeta"])) > 1:
                break

    if GCode == None:
        code.stop()
    else:
        code = reset_secularmultiples(code)
    if exportData:
        if useAMD:
            data = plot_times_Myr,plot_a_AU, plot_e, plot_peri_AU, plot_stellar_inc_deg, plot_AMDBeta
        else:
            data = plot_times_Myr,plot_a_AU, plot_e, plot_peri_AU, plot_stellar_inc_deg
    else:
        data = None
    # Set the Output Code to be the New Code if a Stellar Collision Occured.
    if stellarCollisionOccured:
        newcode = code
    else:
        newcode = None
    return py_particles, data, newcode
