# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Add a as a read in from the restart and crash function for Tyler's code!

# Importing Necessary System Packages
import sys, os, math
import numpy as np
import matplotlib as plt
import time as tp
import random as rp
from optparse import OptionParser
import glob

from guppy import hpy

# Tyler's imports
import hashlib
import copy
import traceback
import signal
from time import gmtime

from collections import defaultdict

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
#from amuse.lab import *

# Import the Amuse Stellar Packages
from amuse.ic.kingmodel import new_king_model
from amuse.ic.kroupa import new_kroupa_mass_distribution

# Import the Amuse Gravity & Close-Encounter Packages
from amuse.community.ph4.interface import ph4
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.community.seba.interface import SeBa
from amuse.couple.bridge import Bridge
from amuse.ext.galactic_potentials import MWpotentialBovy2015

# Import the Tycho Packages
from tycho import create, util, read, write, encounter_db
from tycho import JPGmultiples as TychoMultiples

import tycho

from amuse.support.console import (
    set_printing_strategy, get_current_printing_strategy,
)

from amuse.datamodel import (
    particle_attributes, Particle, Particles, ParticlesSuperset, Grid,
)

set_printing_strategy("custom", preferred_units = [units.MSun, units.AU, units.Myr, units.deg], \
                       precision = 6, prefix = "", separator = "[", suffix = "]")


# ------------------------------------- #
#   Required Non-Seperable Functions    #
# ------------------------------------- #

def print_diagnostics(grav, E0=None):
    # Simple diagnostics.
    ke = grav.kinetic_energy
    pe = grav.potential_energy
    Nmul, Nbin, Emul = grav.get_total_multiple_energy()
    print('')
    print('Time =', grav.get_time().in_(units.Myr))
    print('    top-level kinetic energy =', ke)
    print('    top-level potential energy =', pe)
    print('    total top-level energy =', ke + pe)
    print('   ', Nmul, 'multiples,', 'total energy =', Emul)
    E = ke + pe + Emul
    print('    uncorrected total energy =', E)
    # Apply known corrections.
    Etid = grav.multiples_external_tidal_correction \
            + grav.multiples_internal_tidal_correction  # tidal error
    Eerr = grav.multiples_integration_energy_error	# integration error
    E -= Etid + Eerr
    print('    corrected total energy =', E)
    if E0 is not None: print('    relative energy error=', (E-E0)/E0)
    return E


class EncounterHandler(object):
    def __init__(self):
        self.encounterDict = defaultdict(list)
        self.debug_mode = 0
        self.limiting_mass_for_planets = 13 | units.MJupiter
        self.original_stdout = sys.stdout

    def log_encounter_classbased_v1(self, time, particles_in_encounter):
        # Initialize the Temporary Particle Set to Ensure Nothing
        #    Changes inside "particles_in_encounter"
        _temp = Particles()
        _temp.add_particles(particles_in_encounter)

        # Set the Origin to be the Center of Mass for the Encounter Particle Set.
        _temp.position -= particles_in_encounter.center_of_mass()
        _temp.velocity -= particles_in_encounter.center_of_mass_velocity()
        _temp.time = time

        # Seperate out Stars to Nab Keys for EncounterDictionary Logging
        enc_stars = _temp[_temp.mass > self.limiting_mass_for_planets]
        IDs_of_StarsInEncounter = [star.id for star in enc_stars if star.id < 1000000]
        for star_ID in IDs_of_StarsInEncounter:
            self.encounterDict[star_ID].append(_temp.copy())
        if self.debug_mode > 0:
                enc_planets = _temp[_temp.mass <= self.limiting_mass_for_planets]
                print("Stars:", enc_stars.id)
                print("Planets:", enc_planets.id)
                print("Keys Set for EncounterDictionary:", IDs_of_StarsInEncounter)
                print(self.encounterDict[star_ID][-1])
        # Delete the Temporary Particle Set
        #del _temp
        return True

    def log_encounter_v5(self, time, star1, star2):
        # Create the Scattering CoM Particle Set
        print("Entering Encounter Logger ...")
        if self.debug_mode > 0:
            f = open(os.getcwd()+"/EH-Debug.log", "a")
            sys.stdout = f

        if self.debug_mode > 0:
            print("!-----------------------------")
            print(star1)
            print(star2)
            print("!-----------------------------")
            sys.stdout.flush()
        scattering_com = Particles()
        scattering_com.add_particle(star1)
        scattering_com.add_particle(star2)
        com_pos = scattering_com.center_of_mass()
        com_vel = scattering_com.center_of_mass_velocity()

        # Expand enconter returns a particle set with all of the children
        # when given a particle set of two objects involved in an encounter
        enc_particles = multiples_code.expand_encounter(scattering_com, delete=False)[0]
        if self.debug_mode > 0:
            try:
                print("Scattering:", scattering_com.id, scattering_com.child1.id, scattering_com.child2.id)
            except:
                print("Scattering:", scattering_com.id, scattering_com.child1, scattering_com.child2)
            try:
                print("EncPart:", enc_particles.id, enc_particles.child1.id, enc_particles.child2.id)
            except:
                print("EncPart:", enc_particles.id, enc_particles.child1, enc_particles.child2)

        # Assign the time of the encounter to the Encounter Particle Set.
        enc_particles.time = time

        # Set the Origin to be the Center of Mass for the Encounter Particle Set.
        enc_particles.position -= com_pos
        enc_particles.velocity -= com_vel
        enc_stars = enc_particles[enc_particles.mass > self.limiting_mass_for_planets]
        enc_planets = enc_particles[enc_particles.mass <= self.limiting_mass_for_planets]
        if self.debug_mode > 0:
            print("Stars:", enc_stars.id)
            print("Planets:", enc_planets.id)
            sys.stdout.flush()

        IDs_of_StarsInEncounter = [star.id for star in enc_stars if star.id < 1000000]
        if self.debug_mode > 0:
            print(IDs_of_StarsInEncounter)
            sys.stdout.flush()
        for star_ID in IDs_of_StarsInEncounter:
            self.encounterDict[star_ID].append(enc_particles.copy())

        if self.debug_mode > 0:
            print(self.encounterDict[star_ID])
            sys.stdout.flush()
            f.close()
            sys.stdout = self.original_stdout
       # Return True is Necessary for the Multiples Code
        return True

class ChildUpdater(object):
    def update_children_bodies(self, multiples_code, Individual_Stars, Planets):
        inmemory = (multiples_code._inmemory_particles).copy()
        parents = inmemory[inmemory.id >= 1000000]
        children = self.retrieve_all_children(multiples_code, parents)
        #print "Parents: ", parents.id
        self.sync_particle_subsets_with_children(children, parents, Individual_Stars, Planets)

    def retrieve_all_children(self, multiples_code, parents):
        children = Particles(0)
        for parent in parents:
            if parent in multiples_code.root_to_tree:
                tree = multiples_code.root_to_tree[parent].copy()
                leaves = tree.get_leafs_subset().copy()
                #print "Root ID: ", parent.id
                #print "Leaves ID: ", leaves.id
                self.update_children_position(parent, tree, leaves)
                children.add_particles(leaves)
            else:
                children.add_particle(parent)
        return children

    def update_children_position(self, parent, tree, leaves):
        #print leaves.position.lengths().in_(units.parsec)
        leaves.position -= tree.particle.position
        leaves.velocity -= tree.particle.velocity
        #print leaves.position.lengths().in_(units.parsec)
        leaves.position += parent.position
        leaves.velocity += parent.velocity
        tree.particle.position = parent.position
        tree.particle.velocity = parent.velocity
        #print leaves.position.lengths().in_(units.parsec)

    def sync_particle_subsets_with_children(self, children, parents, Individual_Stars, Planets):
        limiting_mass_for_planets = 13 | units.MJupiter
                # ^ See Definition of an Exoplanet: http://home.dtm.ciw.edu/users/boss/definition.html
        # Seperate Chilren into Planets & Stars
        s_children = children[children.mass > limiting_mass_for_planets]
        p_children = children[children.mass <= limiting_mass_for_planets]
        # TODO: For some reason nearest_neighbour isn't working well with reshapes?
        p_children.host_star = p_children.nearest_neighbour(s_children).id
        print("Stellar Children: ", s_children.id)
        print("Plenatary Children: ", p_children.id)
        print("IStars Position Before Update: ", Individual_Stars[Individual_Stars.id == s_children.id[0]].x.in_(units.parsec))
        # Update Positions of Children in Subsets
        (s_children.new_channel_to(Individual_Stars)).copy_attributes(['x', 'y', 'z', 'vx', 'vy', 'vz'])
        print("IStars Position After Update: ", Individual_Stars[Individual_Stars.id == s_children.id[0]].x.in_(units.parsec))
        (p_children.new_channel_to(Planets)).copy_attributes(['x', 'y', 'z', 'vx', 'vy', 'vz', 'host_star'])


    def move_particle(set_from, set_to, particle_id):
        set_to.add_particle(particle=set_from[particle_id])
        set_from.remove_particle(particle=set_to[particle_id])
        return set_from, set_to


# ------------------------------------- #
#         Main Production Script        #
# ------------------------------------- #

if __name__=="__main__":

# ------------------------------------- #
#      Setting up Required Variables    #
# ------------------------------------- #

    # Read in User Inputs & Set Others to Default Values
    parser = OptionParser()
    parser.add_option("-g", "--no-gpu", dest="no_gpu", action="store_true",
                      help="Disables GPU for computation.")
    parser.add_option("-i", "--gpu-id", dest="gpu_ID", default= -1, type="int",
                      help="Select which GPU to use by device ID.")
    parser.add_option("-p", "--num-psys", dest="num_psys", default=32, type="int",
                      help="Enter the number of planetary systems desired.")
    parser.add_option("-s", "--num-stars", dest="num_stars", default=64, type="int",
                      help="Enter the number of stars desired.")
    parser.add_option("-t", "--timestep", dest="dt", default=2**-3, type="float",
                      help="Enter the Top-Level Timestep in Myr.")
    parser.add_option("-c", "--cluster-name", dest="cluster_name", default=None, type="str",
                      help="Enter the name of the cluster (Defaults to Numerical Naming Scheme).")
    parser.add_option("-w", "--w0", dest="w0", default=4.5, type="float",
                      help="Enter the w0 parameter for the King's Model.")
    parser.add_option("-T", "--end-time", dest="t_end", default=1., type="float",
                      help="Enter the desired end time in Myr.")
    parser.add_option("-b", "--doBinaries", dest="doBinaries", action="store_true",
       		          help = "Enables formation of Primordial Binaries.")
    parser.add_option("-S", "--seed", dest="seed", default = "42", type="str",
                      help = "Enter a random seed for the simulation.")
    parser.add_option("-R", "--doRestart", dest="doRestart", action="store_true",
                      help = "Enables restarting every 100 top-level timesteps.")
    parser.add_option("-P", "--pregen-flag", dest="pregen", action="store_true",
		              help = "Enables loading a pregenerated HDF5 file in the Execution Directory.")
    parser.add_option("-N", "--grav_workers", dest="grav_workers", default=1, type="float",
                          help="Enter the desired number of PH4 workers.")
    parser.add_option("-r", "--virial_radius", dest="vrad", default=2, type="float")
    parser.add_option("-D", "--galactic_dist", dest="galactic_dist", default = 9, type="float")
    parser.add_option("-I", "--interactive_debug", dest="debug", action="store_true",
       		          help = "Enables interactive debugging.")
    (options, args) = parser.parse_args()

    # Set Commonly Used Python Variables from Options
    num_stars = options.num_stars
    num_psys = options.num_psys
    doDebug = options.debug
    w0 = options.w0
    vrad = options.vrad | units.pc
    t_start = 0.0 | units.Myr
    t_end = options.t_end | units.Myr
    delta_t = options.dt | units.Myr
    cluster_name = options.cluster_name
    if cluster_name == None:
        cluster_name = str(options.seed)
    pregen = options.pregen
    doRestart = options.doRestart
    doBinaries = options.doBinaries


# ------------------------------------- #
#        Checking for Crash Files       #
# ------------------------------------- #
    # TODO: Create a System that Handles Crashes & Works w/ Bridge!
    crash = False
    if crash:
        pass
    else:
        pass

# ------------------------------------- #
#         Setting up the Cluster        #
# ------------------------------------- #
    if crash:
        # Should a Crash is Found, Do Not Generate a Cluster.
        pass
    else:
        # Check if a Pregenerated Cluster is Desired
        if pregen:
            # Load the Pregenerated Cluster
            #pregen_file = "/home/draco/jthornton/Tycho/PregenClusters/DracoM3V02AnewSFplt0975.amuse"
            pregen_file = os.getcwd()+"/"+cluster_name+"_IC.amuse"
            Imported_Stars = read_set_from_file(pregen_file, format='amuse', close_file=True)
            # Remove Stellar Type to Prevent Errors from Kepler Worker
            try:
                Imported_Stars.remove_attribute_from_store('stellar_type')
            except:
                pass
            # Define Necessary Varriables & Particle Traits
            num_stars = len(Imported_Stars)
            Starting_Stars = Imported_Stars.sorted_by_attribute('id')
            Starting_Stars.type = "star"
            Starting_Stars.se_radius = Imported_Stars.radius
            Starting_Stars.radius = 2000*Starting_Stars.mass/(1.0 | units.MSun) | units.AU
            # Create the Large Scale Converter & Store the Converter & Initial Conditions
            LargeScaleConverter = nbody_system.nbody_to_si(Starting_Stars.total_mass(),
                                                           Starting_Stars.virial_radius())
            initial_conditions = util.store_ic(LargeScaleConverter, options)
            # Add Planets to the Pregenerated Cluster
            # Select Possible Stars to Become Planetary Systems (NO BINARIES ATM)
            HostStar_MinMass = 0.09 | units.MSun # TRAPPIST-1
            HostStar_MaxMass = 4.00 | units.MSun # NGC 4349-127 or HD 13189
            PossibleHostStars = Starting_Stars.copy()
            for star in PossibleHostStars:
                if star.mass < HostStar_MinMass or star.mass > HostStar_MaxMass:
                    PossibleHostStars.remove_particle(star)
            if num_psys == 32:
                num_psys = int(len(PossibleHostStars)*0.6)
            # Create the Planetary Systems in SU Units
            PSysConverter = nbody_system.nbody_to_si(2*np.mean(PossibleHostStars.mass),
                                                           2*np.mean(PossibleHostStars.radius))
            kep_p = Kepler(unit_converter = PSysConverter, redirection = 'none')
            Starting_Planets = create.planetary_systems_v2(PossibleHostStars, num_psys, Earth=False, Jupiter=True, TestP=False, kepler_worker = kep_p)
            kep_p.stop()
        else:
            # Attempt to Load Matching Initial Conditions
            if os.path.exists(os.getcwd()+"/InitialState"):
                # Reload the Matching Cluster Here
                Starting_Stars, ic_array, LargeScaleConverter = read.read_initial_state(cluster_name)

            # Should Matching I.C. Not Exist, Generate a New Cluster ...
            else:
                # Generate a New Cluster Matching Desired Initial Conditions & the Large-Scale Converter
                if doBinaries:
                    Starting_Stars, LargeScaleConverter, Binary_CoMs, Binary_Singles = \
                        create.king_cluster_v2(num_stars, vradius = vrad, w0 = w0, do_binaries = True,
                                                split_binaries = True, seed = options.seed)
                else:
                    Starting_Stars, LargeScaleConverter = \
                        create.king_cluster_v2(num_stars, vradius = vrad, w0 = w0, do_binaries = False, seed = options.seed)

                # Create Initial Conditions Array
                initial_conditions = util.store_ic(LargeScaleConverter, options)

                # Select Possible Stars to Become Planetary Systems (NO BINARIES ATM)
                HostStar_MinMass = 0.09 | units.MSun # TRAPPIST-1
                HostStar_MaxMass = 4.00 | units.MSun # NGC 4349-127 or HD 13189
                PossibleHostStars = Starting_Stars.copy()
                for star in PossibleHostStars:
                    if doBinaries:
                        if (star.mass in Binary_Singles.mass):
                            PossibleHostStars.remove_particle(star)
                            continue
                    if star.mass < HostStar_MinMass or star.mass > HostStar_MaxMass:
                        PossibleHostStars.remove_particle(star)

                # Create the Planetary Systems in SU Units
                PSysConverter = nbody_system.nbody_to_si(2*np.mean(PossibleHostStars.mass),
                                                               2*np.mean(PossibleHostStars.radius))
                kep_p = Kepler(unit_converter = PSysConverter, redirection = 'none')
                Starting_Planets = create.planetary_systems_v2(PossibleHostStars, num_psys, Earth=False, Jupiter=True, TestP=False, kepler_worker = kep_p)
                kep_p.stop()
    # Set up the Small Scale Converter for Kepler/SmallN
    SmallScaleConverter = nbody_system.nbody_to_si(2*np.mean(Starting_Stars.mass),
                                                   2*np.mean(Starting_Stars.radius))
    # Ensuring the Minimum Interaction Radius for Stars is Held
    for star in Starting_Stars:
        min_star_radius = 1000 | units.AU
        max_star_radius = 10000 | units.AU
        if star.radius <= min_star_radius:
            star.radius = min_star_radius
        elif star.radius >= max_star_radius:
            star.radius = max_star_radius

# ------------------------------------- #
#    Setting up Req Particle Subsets    #
# ------------------------------------- #

    # Setting up "Individual_Stars" (Tracking Individual Stellar Bodies)
    Individual_Stars = Starting_Stars.copy()
    if not pregen:
        # Set Initial Mass to the Initial Stellar Mass
        Individual_Stars.initial_mass = Individual_Stars.mass

    # Setting up "Planets" (Tracking Planets)
    Planets = Starting_Planets.copy()

    # Setting up "Multi_Systems" (Tracking Individual Bodies in Heirarchical Systems)
    # TODO: Implent this Book-Keeping Particle Set
    Multi_Systems = Particles()


# ------------------------------------- #
#   Setting up Req Particle Supersets   #
# ------------------------------------- #

# NOTE: All Changes to These Sets Effect the Subsets/Connected-Supersets & Visa-Versa

    # Setting up "Gravitating_Bodies" (Tracking Bodies Involved w/ Gravity Codes)
    Gravitating_Bodies = ParticlesSuperset([Individual_Stars, Planets])

    # Setting up "Stellar_Bodies" (Tracking Bodies Involved w/ Stellar Evolution Codes)
    Stellar_Bodies = ParticlesSuperset([Individual_Stars])


# ------------------------------------- #
#       Setting up the Integrators      #
# ------------------------------------- #
    # Setting up Galactic Potential Code (MGalaxy)
    galactic_code = MWpotentialBovy2015()
    # Moving Gravitating_Bodies into a Circular Orbit Around Galactic Core
    rinit_from_galactic_core = options.galactic_dist | units.kpc
    vcircular = galactic_code.circular_velocity(rinit_from_galactic_core)
    Gravitating_Bodies.x += rinit_from_galactic_core
    Gravitating_Bodies.vy += vcircular

    #Rgal=1. | units.kpc
    #Mgal=1.6e10 | units.MSun
    #alpha=1.2
    #galactic_code = create.GalacticCenterGravityCode(Rgal, Mgal, alpha)
    #rinit_from_galaxy_core = 5.0 | units.kpc
    #galactic_code.move_particles_into_ellipictal_orbit(Gravitating_Bodies, rinit_from_galaxy_core)


    # ----------------------------------------------------------------------------------------------------

    # Setting up Top-Level Gravity Code (PH4)
    eps2 = (0.0|units.parsec)**2 #1 | units.AU**2
    try:
        no_gpu = options.no_gpu
        gpu_ID = options.gpu_ID
    except:
        no_gpu = False
    if no_gpu:
        num_workers = options.grav_workers
        gravity_code = ph4(number_of_workers = num_workers, redirection = "none",
                           convert_nbody = LargeScaleConverter)
    else:
        num_workers = options.grav_workers
        gravity_code = ph4(number_of_workers = num_workers, redirection = "none", mode = "gpu",
                           convert_nbody = LargeScaleConverter)
    gravity_code.initialize_code()
    gravity_code.parameters.set_defaults()
    #gravity_code.parameters.begin_time = t_start
    gravity_code.parameters.epsilon_squared = eps2
    gravity_code.parameters.timestep_parameter = 2**(-4)
    if no_gpu:
        pass
    else:
        gravity_code.parameters.use_gpu = 1
        if num_workers == 1:
            gravity_code.parameters.gpu_id = gpu_ID
    stopping_condition = gravity_code.stopping_conditions.collision_detection
    stopping_condition.enable()
    gravity_code.particles.add_particles(Gravitating_Bodies)
    gravity_code.commit_particles()
    sys.stdout.flush()
    # ----------------------------------------------------------------------------------------------------

    # Setting up Two-Body Gradvity Code (Kepler)
    kep = Kepler(unit_converter=SmallScaleConverter, redirection = "none")
    kep.initialize_code()
    # ----------------------------------------------------------------------------------------------------

    # Setting up Close-Encounter Gravity Code (SmallN)
    util.init_smalln(unit_converter=SmallScaleConverter)
    # ----------------------------------------------------------------------------------------------------

    # Initializing the Encounters Dictionary
    # Each Key (Star's ID) will Associate with a List of Encounter Particle
    # Sets as Encounters are Detected
    encounter_file = None
    EH = EncounterHandler()
    EH.debug_mode = 1

    # Setting up Encounter Handler (Multiples)
    multiples_code = TychoMultiples.Multiples(gravity_code, util.new_smalln, kep,
                                         gravity_constant=units.constants.G,
                                         encounter_callback = EH.log_encounter_v5)
    multiples_code.neighbor_perturbation_limit = 0.05
    multiples_code.neighbor_veto = True
    multiples_code.global_debug = 1

    # ----------------------------------------------------------------------------------------------------

    # Setting up Stellar Evolution Code (SeBa)
    sev_code = SeBa()
    sev_code.particles.add_particles(Stellar_Bodies)
    supernova_detection = sev_code.stopping_conditions.supernova_detection
    supernova_detection.enable()

    # ----------------------------------------------------------------------------------------------------


    # Setting up Gravity Coupling Code (Bridge)
    bridge_code = Bridge()
    bridge_code.add_system(multiples_code, (galactic_code,))
    bridge_code.timestep = delta_t
    # ----------------------------------------------------------------------------------------------------


# ------------------------------------- #
#      Setting up Required Channels     #
# ------------------------------------- #

    # Setting up Channels to/from "Individual_Stars"

    # Setting up Channels to/from "Planets"

    # Setting up Channels to/from "Multi_Systems"

    # Setting up Channels to/from "Gravitating_Bodies"
    channel_from_multi_to_gravitating = multiples_code.particles.new_channel_to(Gravitating_Bodies)
    channel_from_gravitating_to_multi = Gravitating_Bodies.new_channel_to(multiples_code.particles)

    # Setting up Channels to/from "Stellar_Bodies"
    channel_from_sev_to_stellar = sev_code.particles.new_channel_to(Stellar_Bodies)


# ------------------------------------- #
#        Setting up Final Tweeks        #
# ------------------------------------- #

    # Piping all Terminal Output to the Log File
    orig_stdout = sys.stdout
    if not doDebug:
        f = open("%s_%s.log" %(cluster_name, tp.strftime("%y%m%d", tp.gmtime())), 'w')
        sys.stdout = f
    sys.stdout.flush()

    # Writing the Initial Conditions & Particle Sets
    if not crash:
        write.write_initial_state(Gravitating_Bodies, initial_conditions, cluster_name)

    snapshots_dir = os.getcwd()+"/Snapshots"
    snapshots_s_dir = os.getcwd()+"/Snapshots/Stars"
    snapshots_p_dir = os.getcwd()+"/Snapshots/Planets"
    if not os.path.exists(snapshots_dir):
        os.makedirs(snapshots_dir)
    if not os.path.exists(snapshots_s_dir):
        os.makedirs(snapshots_s_dir)
    if not os.path.exists(snapshots_p_dir):
        os.makedirs(snapshots_p_dir)

    snapshots_gb_dir = os.getcwd()+"/Snapshots/GravitatingBodies"
    snapshots_sb_dir = os.getcwd()+"/Snapshots/StellarBodies"
    if not os.path.exists(snapshots_gb_dir):
        os.makedirs(snapshots_gb_dir)
    if not os.path.exists(snapshots_sb_dir):
        os.makedirs(snapshots_sb_dir)

    # Artificially Age the Stars
    # TODO: Work on Non-Syncronus Stellar Evolution
    if pregen:
        t_start = (Starting_Stars.ct.max()-Starting_Stars.ct.min()).value_in(units.Myr) | units.Myr # Time Since Stars Begun Forming To Last Formed
        sev_code.evolve_model(t_start)
        util.resolve_supernova(supernova_detection, Stellar_Bodies, t_start)
    elif not crash:
        t_start = 10 | units.Myr # Average for Age of After Gas Ejection
        sev_code.evolve_model(t_start)
        util.resolve_supernova(supernova_detection, Stellar_Bodies, t_start)

    channel_from_sev_to_stellar.copy_attributes(["mass", "luminosity", "stellar_type",
                                                 "temperature", "age"])
    channel_from_gravitating_to_multi.copy_attributes(["mass", "vx", "vy", "vz"])

    # Ensuring that Multiples Picks up All Desired Systems
    gravity_code.parameters.begin_time = t_start
    gravity_code.parameters.zero_step_mode = 1
    multiples_code.evolve_model(gravity_code.model_time)
    gravity_code.parameters.zero_step_mode = 0

    # Ensuring the Gravity Code Starts at the Right Time
    t_current = t_start
    bridge_code.evolve_model(t_current, timestep = t_start/4.)

    min_r = LargeScaleConverter.to_nbody(1000 | units.AU)

    print([r for r in gravity_code.particles.radius if r.number <= min_r.number])


# ------------------------------------- #
#          Evolving the Cluster         #
# ------------------------------------- #

    # TODO: Implement Leap-Frog Coupling of Stellar Evolution & Gravity
    step_index = 0
    #t_catch = t_start + (2000 | units.yr)
    E0 = print_diagnostics(multiples_code)

    #dt_small = 10 | units.yr
    #increase_index = False
    #timestep_reset = False

    write_out_step = np.floor((5 | units.Myr)/delta_t)

    #subset_sync = ChildUpdater()

    if doDebug:
        hp = hpy()
        before = hp.heap()
    while t_current <= t_end:
        t_current += delta_t
        # Evolve the Gravitational Codes ( via Bridge Code)
        bridge_code.evolve_model(t_current)
        #multiples_code.evolve_model(t_current)

        # Update the Leaves of the Multiples Code Particles
        multiples_code.update_leaves_pos_vel()

        # Sync the Gravitational Codes w/ the "Gravitating_Bodies" Superset
        channel_from_multi_to_gravitating.copy_attributes(['x', 'y', 'z', 'vx', 'vy', 'vz'])

        # Evolve the Stellar Codes (via SEV Code with Channels)
        # TODO: Ensure Tight Binaries are Evolved Correctly (See Section 3.2.8)
        sev_code.evolve_model(t_current)
        util.resolve_supernova(supernova_detection, Stellar_Bodies, t_current)

        # Sync the Stellar Code w/ the "Stellar_Bodies" Superset
        channel_from_sev_to_stellar.copy_attributes(["mass", "luminosity", "stellar_type",
                                                    "temperature", "age"])

        # Sync the Multiples Particle Set's Masses to the Stellar_Bodies' Masses
        channel_from_gravitating_to_multi.copy_attributes(["mass", "vx", "vy", "vz"])
        # Note: The "mass" Attribute in "Gravitating_Bodies" is synced when "Stellar_Bodies" is.

        if step_index == 5:
            E0_1 = print_diagnostics(multiples_code)

        # Write out the "Gravitating_Bodies" Superset Every 5 Time-Steps
        if step_index%write_out_step == 0:
            # (On a Copy) Recursively Expand All Top-Level Parent Particles & Update Subsets
            # Note: This Updates the Children's Positions Relative to their Top-Level Parent's Position
            #subset_sync.update_children_bodies(multiples_code, Individual_Stars, Planets)
            snapshot_s_filename = snapshots_s_dir+"/"+cluster_name+"_stars_t%.3f.hdf5" %(t_current.number)
            write_set_to_file(Individual_Stars, snapshot_s_filename, format="hdf5", close_file=True, version='2.0')
            snapshot_p_filename = snapshots_p_dir+"/"+cluster_name+"_planets_t%.3f.hdf5" %(t_current.number)
            write_set_to_file(Planets, snapshot_p_filename, format="hdf5", close_file=True, version='2.0')
            snapshot_gb_filename = snapshots_gb_dir+"/"+cluster_name+"_gravitating_t%.3f.hdf5" %(t_current.number)
            write_set_to_file(Gravitating_Bodies, snapshot_gb_filename, format="hdf5", close_file=True, version='2.0')
            snapshot_sb_filename = snapshots_sb_dir+"/"+cluster_name+"_sev_t%.3f.hdf5" %(t_current.number)
            write_set_to_file(Stellar_Bodies, snapshot_sb_filename, format="hdf5", close_file=True, version='2.0')

        # TODO: Write out a Crash File Every 50 Time-Steps
        #crash_base = "CrashSave/"+cluster_name+"_time_"+t_current.in_(units.Myr)

        # TODO: Restart the Integrators Every 100 Time-Steps
        #restart_file = "Restart/"+cluster_name+"_time_"+t_current.in_(units.Myr)

        # Save the Encounters Dictionary Thus Far (If Not the First Reset)
        if step_index != 0:
            if encounter_file != None:
                # First Try to Delete the Previous Backup File
                # Its in a try because there will not be a backup the first time this runs
                # Could replace try loop with a counter if desired
                try:
                    os.remove(cluster_name+"_encounters_backup.pkl")
                except:
                    pass
                # Then Rename the Previous Encounter Dictionary
                os.rename(cluster_name+"_encounters.pkl",
                          cluster_name+"_encounters_backup.pkl")
            # Finally, Save the Encounter Dictionary!
            encounter_file = open(cluster_name+"_encounters.pkl", "wb")
            if EH.debug_mode > 0:
                stars_in_EHDB = EH.encounterDict.keys()
                print(stars_in_EHDB)
                print(EH.encounterDict)
            pickle.dump(EH.encounterDict, encounter_file)
            encounter_file.close()
            # Log that a the Encounters have been Saved!
            print('\n-------------')
            print('[UPDATE] Encounters Saved at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime())))
            print('-------------\n')
            sys.stdout.flush()

            if step_index%500 == 0 and step_index != 0 and doDebug:
                after = hp.heap()
                leftover = after - before
                #sys.stdout = orig_stdout
                sys.stdout.flush()
                import pdb; pdb.set_trace()

        # Increase the Step Index
        #if increase_index:
        step_index += 1

        # Log that a Step was Taken
        print('\n-------------')
        print('[UPDATE] Step Taken at %s!' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime())))
        print('-------------\n')

        # Flush Terminal Output to Log File
        sys.stdout.flush()

# ------------------------------------- #
#        Stopping the Integrators       #
# ------------------------------------- #

    print_diagnostics(multiples_code, E0)
    print_diagnostics(multiples_code, E0_1)
    sys.stdout.flush()

    if not doDebug:
        sys.stdout = orig_stdout
    print('\n[UPDATE] Run Finished at %s! \n' %(tp.strftime("%Y/%m/%d-%H:%M:%S", tp.gmtime())))
    print_diagnostics(multiples_code, E0)
    print_diagnostics(multiples_code, E0_1)
    sev_code.stop()
    gravity_code.stop()
    kep.stop()
    util.stop_smalln()
    try:
        bridge_code.stop()
    except:
        pass

    sys.stdout.flush()
