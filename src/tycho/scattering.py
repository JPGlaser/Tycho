from tycho import *
from tycho import stellar_systems
from tycho import enc_patching
from amuse import *
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units

import glob

import hashlib

import pickle

from collections import defaultdict

from amuse.community.secularmultiple.interface import SecularMultiple

from amuse.datamodel.trees import BinaryTreesOnAParticleSet
from amuse.ext.orbital_elements import new_binary_from_orbital_elements

from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.community.ph4.interface import ph4
from amuse.community.sse.interface import SSE

set_printing_strategy("custom", preferred_units = [units.MSun, units.AU, units.day, units.deg], precision = 6, prefix = "", separator = "[", suffix = "]")

def build_ClusterEncounterHistory(rootExecDir):
    # Strip off Extra '/' if added by user to bring inline with os.cwd()
    if rootExecDir.endswith("/"):
        rootExecDir = rootExecDir[:-1]
    # Define the Cluster's Name
    #cluster_name = rootExecDir.split("/")[-1]
    # Generate List of Scattering IC HDF5 Paths
    paths_of_IC_files = glob.glob(rootExecDir+'/Scatter_IC/*/*.hdf5')
    # Find all Primary Star IDs
    star_IDs = np.unique([int(path.split("/")[-2]) for path in paths_of_IC_files]) # '1221'
    EncounterHistory = defaultdict(dict)
    for i, star_ID in enumerate(star_IDs[:10]):
        RotationKeys = np.unique([path.split("/")[-1].split(".")[-2].split('_')[-1] for path in paths_of_IC_files if int(path.split("/")[-2]) == star_ID])
        EncounterKeys = np.unique([path.split("/")[-1].split(".")[-2].split('_')[0] for path in paths_of_IC_files if int(path.split("/")[-2]) == star_ID])
        EncounterHistory[star_ID] = defaultdict(dict)
        for RotationKey in RotationKeys:
            EncounterHistory[star_ID][RotationKey] = []
            for EncounterKey in EncounterKeys:
                EncounterHistory[star_ID][RotationKey].append(str(rootExecDir+'/Scatter_IC/'+str(star_ID)+'/'+EncounterKey+'_'+RotationKey+'.hdf5'))
    return EncounterHistory

from amuse.community.secularmultiple.interface import SecularMultiple

def initialize_GravCode(desiredCode, **kwargs):
    converter = kwargs.get("converter", None)
    n_workers = kwargs.get("number_of_workers", 1)
    if converter == None:
        converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    GCode = desiredCode(number_of_workers = n_workers, redirection = "none", convert_nbody = converter)
    GCode.initialize_code()
    GCode.parameters.set_defaults()
    if desiredCode == ph4:
        GCode.parameters.timestep_parameter = 2.0**(-4.0)
    if desiredCode == SmallN:
        GCode.parameters.timestep_parameter = 0.05
    return GCode

def initialize_isOverCode(**kwargs):
    converter = kwargs.get("converter", None)
    if converter == None:
        converter = nbody_system.nbody_to_si(1 | units.MSun, 100 |units.AU)
    isOverCode = SmallN(redirection = "none", convert_nbody = converter)
    isOverCode.initialize_code()
    isOverCode.parameters.set_defaults()
    isOverCode.parameters.allow_full_unperturbed = 0
    return isOverCode

class CloseEncounters():
    def __init__(self, Star_EncounterHistory, KeplerWorkerList = None, \
                 NBodyWorkerList = None, SecularWorker = None, SEVWorker = None):
        '''EncounterHistory should be a List of the Format {RotationKey: [Encounter0_FilePath, ...]}'''
        # Find the Main System's Host Star's ID and Assign it to 'KeySystemID'
        self.doEncounterPatching = True
        self.KeySystemID = int(Star_EncounterHistory[list(Star_EncounterHistory)[0]][0].split("/")[-2])
        self.ICs = defaultdict(list)
        self.StartTimes = defaultdict(list)
        self.desired_endtime = 1.0 | units.Gyr
        self.max_end_time =  0.1 | units.Myr
        self.kep = KeplerWorkerList
        self.NBodyCodes = NBodyWorkerList
        self.SecularCode = SecularWorker
        self.SEVCode = SEVWorker
        self.getOEData = True
        self.OEData = defaultdict(list)
        # Create a List of StartingTimes and Encounter Initial Conditions (ICs) for all Orientations
        for RotationKey in Star_EncounterHistory.keys():
            for i, Encounter in enumerate(Star_EncounterHistory[RotationKey]):
                self.ICs[RotationKey].append(read_set_from_file(Encounter, format="hdf5", version='2.0', close_file=True))
                self.StartTimes[RotationKey].append(np.max(np.unique(self.ICs[RotationKey][i].time)))
        #print(self.KeySystemID)
        self.FinalStates = defaultdict(list)

    def SimAllEncounters(self):
        ''' Call this function to run all Encounters for a System.'''
        # Start up Kepler Functions if Needed
        if self.kep == None:
            self.kep = []
            bodies = self.ICs[next(iter(self.ICs))][0]
            converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
            self.kep.append(Kepler(unit_converter = converter, redirection = 'none'))
            self.kep.append(Kepler(unit_converter = converter, redirection = 'none'))
            self.kep[0].initialize_code()
            self.kep[1].initialize_code()
        # Start up NBodyCodes if Needed
        if self.NBodyCodes == None:
            self.NBodyCodes = [initialize_GravCode(ph4), initialize_isOverCode()]
        # Start up SecularCode if Needed
        if self.SecularCode == None:
            self.SecularCode = SecularMultiple()
        if self.SEVCode == None:
            self.SEVCode = SSE()

        # Begin Looping over Rotation Keys ...
        for RotationKey in self.ICs.keys():
            for i in range(len(self.ICs[RotationKey])):
                try:
                    print(util.timestamp(), "!!! UPDATE: Starting the following encounter", \
                          "Star", self.KeySystemID, RotationKey, "-", i)

                    # Identify the Current Encounter in the List for This Rotation
                    CurrentEncounter = self.ICs[RotationKey][i]
                    #print(CurrentEncounter[0].position)

                    # Create the Encounter Instance with the Current Encounter
                    Encounter_Inst = self.SingleEncounter(CurrentEncounter)

                    # Simulate the Encounter till the Encounter is Over via N-Body Integrator
                    # -OR- the time to the Next Encounter is Reached
                    if len(self.StartTimes[RotationKey]) == 1 or i+1 == len(self.StartTimes[RotationKey]):
                        current_max_endtime = self.max_end_time
                    else:
                        current_max_endtime = self.StartTimes[RotationKey][i+1]
                    EndingState = Encounter_Inst.SimSingleEncounter(current_max_endtime, \
                                                                    start_time = self.StartTimes[RotationKey][i], \
                                                                    GCodes = self.NBodyCodes)
                    EndingStateTime = np.max(np.unique(EndingState.time))

                    #print(Encounter_Inst.particles[0].position)
                    print("The Encounter was over after:", (EndingStateTime- self.StartTimes[RotationKey][i]).value_in(units.Myr))


                    #print(EndingState.id, EndingState.x)
                    print('----------')
                    #print(Encounter_Inst.particles.id, Encounter_Inst.particles.x)

                    # Strip off Anything Not Associated with the Key System
                    systems_in_current_encounter = stellar_systems.get_heirarchical_systems_from_set(EndingState, kepler_workers=self.kep)

                    # Reassign the EndingState to include the Primary System ONLY
                    EndingState = systems_in_current_encounter[self.KeySystemID]
                    print("Before Secular:", EndingState.id,  EndingState.x)
                    #print(EndingState[0].position)

                    #print(len(self.ICs[RotationKey])-1)
                    # If Encounter Patching is Desired -AND- it isn't the last Encounter
                    if i + 1 < len(self.ICs[RotationKey]) and self.doEncounterPatching:

                        # Identify the Next Encounter in the List
                        NextEncounter = self.ICs[RotationKey][i+1]

                        # Simulate System till the Next Encounter's Start Time
                        Encounter_Inst = self.SingleEncounter(EndingState)
                        FinalState, data, newcode = Encounter_Inst.SimSecularSystem(self.StartTimes[RotationKey][i+1], \
                                                                     start_time = EndingStateTime, \
                                                                     GCode = self.SecularCode, getOEData=self.getOEData, \
                                                                     KeySystemID = self.KeySystemID, SCode=self.SEVCode)
                        if newcode != None:
                            self.SecularCode = newcode
                        print("After Secular:", FinalState.id)
                        # Begin Patching of the End State to the Next Encounter
                        self.ICs[RotationKey][i+1] = self.PatchedEncounter(FinalState, NextEncounter)
                    else:
                        # Simulate System till Desired Global Endtime
                        #print(CurrentEncounter[0].time.value_in(units.Myr))
                        #print(EndingState[0].time.value_in(units.Myr))
                        Encounter_Inst = self.SingleEncounter(EndingState)
                        FinalState, data, newcode = Encounter_Inst.SimSecularSystem(self.desired_endtime, \
                                                                     start_time = EndingStateTime, \
                                                                     GCode = self.SecularCode, getOEData=self.getOEData, \
                                                                     KeySystemID = self.KeySystemID, SCode=self.SEVCode)
                        if newcode != None:
                            self.SecularCode = newcode
                        print("After Secular:", FinalState.id)

                    # Append the FinalState of Each Encounter to its Dictionary
                    self.FinalStates[RotationKey].append(FinalState)
                    if self.getOEData and data != None:
                        self.OEData[RotationKey].append(data)
                except:
                    print("!!!! Alert: Skipping", RotationKey,"-", i, "for Star", self.KeySystemID, "due to unforseen issues!")
                    print("!!!!        The Particle Set's IDs are as follows:", self.ICs[RotationKey][i].id)

        # Stop the NBody Codes if not Provided
        if self.kep == None:
            self.kep[0].stop()
            self.kep[1].stop()
        # Start up NBodyCodes if Needed
        if self.NBodyCodes == None:
            self.NBodyCodes[0].stop()
            self.NBodyCodes[1].stop()
        # Start up SecularCode if Needed
        if self.SecularCode == None:
            self.SecularCode.stop()
        return None

    def PatchedEncounter(self, EndingState, NextEncounter):
        ''' Call this function to Patch Encounter Endstates to the Next Encounter'''
        # Determine Time to Next Encounter
        current_time = max(EndingState.time)
        final_time = max(NextEncounter.time)

        # Map the Orbital Elements to the Child2 Particles (LilSis) [MUST BE A TREE SET]
        enc_patching.map_node_oe_to_lilsis(EndingState)

        # Seperate Next Encounter Systems to Locate the Primary System
        systems_at_next_encounter = stellar_systems.get_heirarchical_systems_from_set(NextEncounter)
        sys_1 = systems_at_next_encounter[self.KeySystemID]
        # Note: This was changed to handle encounters of which result in one
        #       bound object of multiple subsystems. ~ Joe G. | 8/24/20
        BoundObjOnly = False
        if len(systems_at_next_encounter.keys()) == 1:
            BoundObjOnly = True
        else:
            secondary_sysID = [key for key in list(systems_at_next_encounter.keys()) if key!=int(self.KeySystemID)][0]
            sys_2 = systems_at_next_encounter[secondary_sysID]

        # Get Planet and Star Subsets for the Current and Next Encounter
        children_at_EndingState = EndingState.select(lambda x : x == False, ["is_binary"])
        planets_at_current_encounter = util.get_planets(children_at_EndingState)
        hoststar_at_current_encounter = util.get_stars(children_at_EndingState).select(lambda x : x == self.KeySystemID, ["id"])[0]
        planets_at_next_encounter = util.get_planets(sys_1)
        print("Planets at Next Encount:", planets_at_next_encounter.id)
        hoststar_at_next_encounter = util.get_stars(sys_1).select(lambda x : x == self.KeySystemID, ["id"])[0]
        #print(hoststar_at_next_encounter)

        # Update Current Positions & Velocitys to Relative Coordinates from Orbital Parameters!!
        # TO-DO: Does not handle Binary Star Systems
        for planet in planets_at_current_encounter:
            #print(planet.id, planet.position)
            nbody_PlanetStarPair = \
            new_binary_from_orbital_elements(hoststar_at_current_encounter.mass, planet.mass, planet.semimajor_axis, \
                                             eccentricity = planet.eccentricity, inclination=planet.inclination, \
                                             longitude_of_the_ascending_node=planet.longitude_of_ascending_node, \
                                             argument_of_periapsis=planet.argument_of_pericenter, G=units.constants.G, \
                                             true_anomaly = 360*rp.uniform(0.0,1.0) | units.deg) # random point in the orbit
            planet.position = nbody_PlanetStarPair[1].position
            planet.velocity = nbody_PlanetStarPair[1].velocity
            #print(planet.id, planet.position)

        for planet in planets_at_current_encounter:
            print(planet.id, planet.position)

        # Release a Warning when Odd Planet Number Combinations Occur (Very Unlikely, More of a Safe Guard)
        if len(planets_at_current_encounter) != len(planets_at_next_encounter):
            print("!!!! Expected", len(planets_at_next_encounter), "planets but recieved only", len(planets_at_current_encounter))

        # Move Planets to Host Star in the Next Encounter
        for next_planet in planets_at_next_encounter:
            for current_planet in planets_at_current_encounter:
                if next_planet.id == current_planet.id:
                    next_planet.position = current_planet.position + hoststar_at_next_encounter.position
                    next_planet.velocity = current_planet.velocity + hoststar_at_next_encounter.velocity
                    break

        #for planet in planets_at_next_encounter:
        #    print(planet.id, planet.position)
        #for particle in sys_1:
        #    print(particle.id, particle.position)


        # Recombine Seperated Systems to Feed into SimSingleEncounter
        UpdatedNextEncounter = Particles()
        print("IDs in System 1", sys_1.id)
        UpdatedNextEncounter.add_particles(sys_1)
        if not BoundObjOnly:
            UpdatedNextEncounter.add_particles(sys_2)
            print("IDs in System 2", sys_2.id)

        # Return the Updated and Patched Encounter as a Partcile Set for the N-Body Simulation
        return UpdatedNextEncounter

    class SingleEncounter():
        def __init__(self, EncounterBodies):
            self.particles = EncounterBodies

        def SimSecularSystem(self, desired_end_time, **kwargs):
            start_time = kwargs.get("start_time", 0 | units.Myr)
            getOEData = kwargs.get("getOEData", False)
            KeySystemID = kwargs.get("KeySystemID", None)
            GCode = kwargs.get("GCode", None)
            SEVCode = kwargs.get("SCode", None)

            self.particles, data, newcode = enc_patching.run_secularmultiple(self.particles, desired_end_time, \
                                                                  start_time = start_time, N_output=1, \
                                                                  GCode=GCode, exportData=getOEData, \
                                                                  KeySystemID=KeySystemID, SEVCode=SEVCode)
            return self.particles, data, newcode

        def SimSingleEncounter(self, max_end_time, **kwargs):
            delta_time = kwargs.get("delta_time", 100 | units.yr)
            converter = kwargs.get("converter", None)
            start_time = kwargs.get("start_time", 0 | units.yr)
            doStateSaves = kwargs.get("doStateSaves", False)
            doVerbosSaves = kwargs.get("doEncPatching", False)
            GCodes = kwargs.get("GCodes", None)

            GravitatingBodies = self.particles

            # Set Up the Integrators
            if GCodes == None:
                if converter == None:
                    converter = nbody_system.nbody_to_si(GravitatingBodies.mass.sum(), \
                                                         2 * np.max(GravitatingBodies.radius.number) | GravitatingBodies.radius.unit)
                gravity = initialize_GravCode(ph4, converter=converter)
                over_grav = initialize_isOverCode(converter=converter)
            else:
                gravity = GCodes[0]
                over_grav = GCodes[1]

            # Set up SaveState if Requested
            if doStateSaves:
                pass    # Currently Massive Memory Leak Present

            # Store Initial Center of Mass Information
            rCM_i = GravitatingBodies.center_of_mass()
            vCM_i = GravitatingBodies.center_of_mass_velocity()

            # Remove Attributes that Cause Issues with SmallN
            if 'child1' in GravitatingBodies.get_attribute_names_defined_in_store():
                del GravitatingBodies.child1, GravitatingBodies.child2

             # Moving the Encounter's Center of Mass to the Origin and Setting it at Rest
            GravitatingBodies.position -= rCM_i
            GravitatingBodies.velocity -= vCM_i

            # Add and Commit the Scattering Particles
            gravity.particles.add_particles(GravitatingBodies) # adds bodies to gravity calculations
            gravity.commit_particles()
            #gravity.begin_time = start_time

            # Create the Channel to Python Set & Copy it Over
            channel_from_grav_to_python = gravity.particles.new_channel_to(GravitatingBodies)
            channel_from_grav_to_python.copy()

            # Get Free-Fall Time for the Collision
            s = util.get_stars(GravitatingBodies)
            t_freefall = s.dynamical_timescale()
            #print(gravity.begin_time)
            #print(t_freefall)

            # Setting Coarse Timesteps
            list_of_times = np.arange(0.0, start_time.value_in(units.yr)+max_end_time.value_in(units.yr), \
                                      delta_time.value_in(units.yr)) | units.yr
            stepNumber = 0

            # Loop through the List of Coarse Timesteps
            for current_time in list_of_times:
                #print(current_time)
                #print(GravitatingBodies.time)
                #print(gravity.sync_time)
                # Evolve the Model to the Desired Current Time
                gravity.evolve_model(current_time)

                # Update Python Set in In-Code Set
                channel_from_grav_to_python.copy() # original
                channel_from_grav_to_python.copy_attribute("index_in_code", "id")


                # Check to See if the Encounter is Over After the Freefall Time and Every 25 Steps After That
                if current_time > 1.25*t_freefall and stepNumber%25 == 0:
                    over = util.check_isOver(gravity.particles, over_grav)
                    print("Is it Over?", over)
                    if over:
                        #print(gravity.particles[0].position)
                        #print(GravitatingBodies[0].position)
                        current_time += 100 | units.yr
                        # Get to a Final State After Several Planet Orbits
                        gravity.evolve_model(current_time)
                        gravity.update_particle_set()
                        gravity.particles.synchronize_to(GravitatingBodies)
                        channel_from_grav_to_python.copy()
                        GravitatingBodies.time = start_time + current_time
                        # Create a Save State
                        break
                if current_time == list_of_times[-1]:
                    # Create a Save State
                    pass
                stepNumber +=1
            if GCodes == None:
                # Stop the Gravity Code Once the Encounter Finishes
                gravity.stop()
                over_grav.stop()
            else:
                # Reset the Gravity Codes Once Encounter Finishes
                gravity.reset()
                over_grav.reset()

            # Return the GravitatingBodies as they are at the End of the Simulation
            return GravitatingBodies
