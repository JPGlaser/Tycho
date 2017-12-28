import numpy
from amuse.units import nbody_system, units, constants
from amuse.ic.plummer import new_plummer_model
from amuse.community.ph4.interface import ph4
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
#from amuse.couple import multiples
from tycho import multiples2, create
from amuse import datamodel

class EncounterHandler(object):
    def handle_encounter_v3(self, time, star1, star2):
        fh = open("TestingEncounters.txt", "a")
        scattering_com = datamodel.Particles(particles = (star1, star2))
        com_pos = scattering_com.center_of_mass()
        com_vel = scattering_com.center_of_mass_velocity()
        enc_list = multiples_code.expand_encounter(scattering_com, delete=False)[0]
        enc_list.position -= com_pos
        enc_list.velocity -= com_vel
        print (scattering_com.position - com_pos).in_(units.AU)
        print (enc_list.position).in_(units.AU)
        highest_mass_pair = enc_list.sorted_by_attribute('mass')[-2:]
        encounter_sep = (highest_mass_pair[0].position - highest_mass_pair[1].position).length()
        fh.write("Encounter Seperation:"+str(encounter_sep.in_(units.AU))+"\n")
        for particle in enc_list:
            str_mass = str((particle.mass).in_(units.MSun))
            fh.write(str(particle.id)+" ("+str_mass+"), ")
        fh.write("\n")
        for particle in enc_list:
            distance_from_com = (particle.position).length()
            fh.write(str(distance_from_com.in_(units.AU))+", ")
        fh.write("\n\n")
        fh.close
        # return true is necessary for the multiples code
        return True
        

# Awkward syntax here because multiples needs a function that resets
# and returns a small-N integrator.

SMALLN = None
def init_smalln(converter):
    global SMALLN
    SMALLN = SmallN(convert_nbody=converter)

def new_smalln():
    SMALLN.reset()
    return SMALLN

def stop_smalln():
    global SMALLN
    SMALLN.stop()

def print_diagnostics(grav, E0=None):

    # Simple diagnostics.

    ke = grav.kinetic_energy
    pe = grav.potential_energy
    Nmul, Nbin, Emul = grav.get_total_multiple_energy()
    print ''
    print 'Time =', grav.get_time().in_(units.Myr)
    print '    top-level kinetic energy =', ke
    print '    top-level potential energy =', pe
    print '    total top-level energy =', ke + pe
    print '   ', Nmul, 'multiples,', 'total energy =', Emul
    E = ke + pe + Emul
    print '    uncorrected total energy =', E
    
    # Apply known corrections.
    
    Etid = grav.multiples_external_tidal_correction \
            + grav.multiples_internal_tidal_correction  # tidal error
    Eerr = grav.multiples_integration_energy_error	# integration error

    E -= Etid + Eerr
    print '    corrected total energy =', E

    if E0 is not None: print '    relative energy error=', (E-E0)/E0
    
    return E

def integrate_system(N, t_end, seed=None):

    # Initialize an N-body module and load a stellar system.

    mass = N|units.MSun
    length = 1|units.parsec
    converter = nbody_system.nbody_to_si(mass, length)
    gravity = ph4(convert_nbody=converter)
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    gravity.parameters.epsilon_squared = (0.0|units.parsec)**2

    if seed is not None: numpy.random.seed(seed)
    stars = new_plummer_model(N, convert_nbody=converter)
    stars.mass = mass/N
    stars.scale_to_standard(convert_nbody=converter,
                            smoothing_length_squared
                             = gravity.parameters.epsilon_squared)

    # Star IDs are important, as they are used in multiples bookkeeping.
    id = numpy.arange(N)
    stars.id = id+1

    # Set dynamical radii.

    #stars.radius = 0.5/N | units.parsec
    stars.radius = 2000 | units.AU

    gravity.particles.add_particles(stars)

    # Add Planets
    do_planets = True
    if do_planets:
        systems_SI = create.planetary_systems(stars, N-1, 'test_planets', Jupiter=True)
        gravity.particles.add_particles(systems_SI)

    # Enable collision detection.

    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()

    SmallScaleConverter = nbody_system.nbody_to_si(2*mass/N, 4000 | units.AU)
    # Define the small-N code.

    init_smalln(SmallScaleConverter)

    # Define a Kepler module.

    kep = Kepler(unit_converter=SmallScaleConverter)
    kep.initialize_code()

    # Create the multiples module.
    global multiples_code
    multiples_code = multiples2.Multiples(gravity, new_smalln, kep,
                                         constants.G)
    multiples_code.neighbor_perturbation_limit = 0.05
    multiples_code.neighbor_veto = True

    multiples_code.global_debug = 1	    # 0: no output from multiples
                                        # 1: minimal output
                                        # 2: debugging output
                                        # 3: even more output
                                        
    # Setting Up the Encounter Handler
    encounters = EncounterHandler()
    multiples_code.callback = encounters.handle_encounter_v3

    # Print selected multiples settings.

    print ''
    print 'multiples_code.neighbor_veto =', \
        multiples_code.neighbor_veto
    print 'multiples_code.neighbor_perturbation_limit =', \
        multiples_code.neighbor_perturbation_limit
    print 'multiples_code.retain_binary_apocenter =', \
        multiples_code.retain_binary_apocenter
    print 'multiples_code.wide_perturbation_limit =', \
        multiples_code.wide_perturbation_limit

    # Advance the system.
    time = numpy.sqrt(length**3/(constants.G*mass))
    print '\ntime unit =', time.in_(units.Myr)
    
    E0 = print_diagnostics(multiples_code)
    dt = 0.1 | units.Myr
    t_current = 0.0 | units.Myr
    #gravity.particles[-1].mass += 1 | units.MSun
    while t_current <= t_end:
        t_current += dt
        multiples_code.evolve_model(t_current)
        print gravity.particles.index_in_code #, gravity.particles[0].x
        #print multiples_code.particles[-1].key, multiples_code.particles[-1].x
        print multiples_code._inmemory_particles.id, #multiples_code._inmemory_particles[-1].x
        #print stars[0].id
        # Testing the Multiples & Gravity Particle Set Differences
        #test_psystem_id = systems_SI[0].host_star
        #test_psystem_hs_grav = [star.x for star in gravity.particles if star.index_in_code == test_psystem_id]
        #test_psystem_hs_mult = [star.x for star in multiples_code.particles if star.idex_in_code == test_psystem_id]
        #print "-----------------------------"
        #print "X Position in PH4", test_pystem_hs_grav.in_(units.parsec)
        #print "X Position in Multiples", test_psystem_hs_mult.in_(units.parsec)
        #print "Difference in Particle Sets", test_pystem_hs_grav.in_(units.parsec) - test_psystem_hs_mult.in_(units.parsec)
        #print "-----------------------------"
    print_diagnostics(multiples_code, E0)

    gravity.stop()
    kep.stop()
    stop_smalln()
    
if __name__ in ('__main__'):
    N = 10
    t_end = 5.0 | units.Myr
    integrate_system(N, t_end) #, 42)
