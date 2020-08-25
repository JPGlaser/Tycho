# Python Classes/Functions containing the Stellar Systems Structure for Tycho
# Keep This Class Unitless!

# ------------------------------------- #
#        Python Package Importing       #
# ------------------------------------- #

# Importing Necessary System Packages
import sys, os, math
import numpy as np
import matplotlib.pyplot as plt
import time as tp
import random as rp
import hashlib
import scipy as sp
from scipy import optimize
from scipy import special

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
from amuse.lab import *
from tycho import util

from amuse.community.kepler.interface import Kepler
from amuse.community.sse.interface import SSE

# ------------------------------------- #
#           Defining Functions          #
# ------------------------------------- #

def get_system_id(planet):
    host_star = planet.host_star
    if host_star != None:
        return (planet.id%10000, host_star.id, host_star.mass.value_in(units.MSun))
    if host_star == None:
        return (planet.id%10000, 0, 0.0)

# Periods are More Often Cited than Semimajor Axii
def get_periods(host_star, planets):
    for planet in planets:
        mu = constants.G*(planet.mass+host_star.mass)
        a = planet.semimajor_axis
        print(mu, a, 2.0*np.pi/np.sqrt(mu)*a**(3./2.))
        planet.period = 2.0*np.pi/np.sqrt(mu)*a**(3./2.)

def update_orb_elem(host_star, planets, converter=None, kepler_worker=None):
    if kepler_worker == None:
        if converter == None:
            tot_sys = Particles(particles=(host_star, planets))
            converter = nbody_system.nbody_to_si(tot_sys.mass.sum(), 2 * host_star.radius)
        kep_p = Kepler(unit_converter = converter, redirection = 'none')
        kep_p.initialize_code()
    else:
        kep_p = kepler_worker
    for planet in planets:
        total_mass = host_star.mass + planet.mass
        kep_pos = host_star.position - planet.position
        kep_vel = host_star.velocity - planet.velocity
        kep_p.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])
        planet.semimajor_axis, planet.eccentricity = kep_p.get_elements()
        planet.period = kep_p.get_period()
        planet.true_anomaly, planet.mean_anomaly = kep_p.get_angles()
    if kepler_worker == None:
        kep_p.stop()

def update_host_star(system, converter=None, kepler_worker=None):
    if kepler_worker == None:
        if converter == None:
            converter = nbody_system.nbody_to_si(system.mass.sum(), 2 * np.max(system.radius.number) | system.radius.unit)
        kep_p = Kepler(unit_converter = converter, redirection = 'none')
        kep_p.initialize_code()
    else:
        kep_p = kepler_worker
    stars = util.get_stars(system)
    planets = util.get_planets(system)
    p_NearestStar = planets.nearest_neighbour(stars)
    for i, planet in enumerate(planets):
        likely_host = p_NearestStar[i]
        update_orb_elem(likely_host, [planet], converter=converter, kepler_worker=kep_p)
        if planet.eccentricity >= 1.0:
            for s in stars - likely_host:
                update_orb_elem(s, [planet], converter=converter, kepler_worker=kep_p)
                if planet.eccentricity < 1.0:
                    planet.host_star = s.id
                    break
                elif planet.eccentricity >= 1.0:
                    planet.host_star = -1
        else:
            planet.host_star = likely_host.id
    if kepler_worker == None:
        kep_p.stop()

def get_zaxis_inclination(host_star, planets):
    for planet in planets:
        rel_pos = planet.position - host_star.position
        rel_vel = planet.velocity - host_star.velocity
        mom_vec = np.cross(rel_pos.number, rel_vel.number)
        mom_norm = np.sqrt(mom_vec[0]**2 + mom_vec[1]**2 + mom_vec[2]**2)
        planet.z_inc = np.arccos(mom_vec[2]/mom_norm)*180.0/np.pi | units.deg

def calc_zaxis_inclination(host_star, planet):
    get_zaxis_inclination(host_star, [planet])
    return planet.z_inc

def get_rel_inclination(planets, method='Jovian'):
    if method == 'Jovian':
        largest_p = planets.sorted_by_attribute('mass')[-1]
        for planet in planets:
            planet.rel_inc = planet.z_inc - largest_p.z_inc

def equation_35(inner_e, gamma, alpha):
    return alpha*inner_e + gamma*inner_e/np.sqrt(alpha*(1.-inner_e**2) + gamma**2*inner_e**2) - 1. + alpha

def equation_99(one_minus_alpha, r, epsilon):
    return 729*one_minus_alpha**7. - 4608.*r*epsilon*one_minus_alpha**3 - 16384.*(r*epsilon)**2

class PlanetarySystem():
    def __init__(self, host_star, planets, system_name='', uncertainties=None):
        try:
            self.planets = (planets.copy()).sorted_by_attribute('semimajor_axis')
        except:
            print('Error: Planets do Not have Semimajor Axis Attribute.')
        self.host_star = host_star.copy()
        self.number_of_planets = len(planets)
        self.system_name = system_name
        self.r = (sp.special.kv(1, 2.0/3.0) + 2*sp.special.kv(0, 2.0/3.0))/np.pi
        try:
            self.planets.period.sorted_by_attribute('period')
        except:
            get_periods(self.host_star, self.planets)

    def get_RelAMD(self, p_index):
        """C/Lam_i for Planet p_index"""
        relAMD = 0
        period_i = self.planets[p_index].period
        mass_i = self.planets[p_index].mass
        e_i = self.planets[p_index].eccentricity
        for planet in self.planets:
            period = planet.period
            mass = planet.mass
            e = planet.eccentricity
            alpha = (period/period_i)**(2./3)
            gamma = mass/mass_i
            relAMD += gamma * np.sqrt(alpha) *(1 - np.sqrt(1 - e**2.0))
        return relAMD

    def get_SystemBetaValues(self):
        """Calculate the AMD Stability Coefficient, Beta, values for the system."""
        res = [self.get_AMDBeta(i) for i in range(self.number_of_planets)]
        return np.array(res)

    def get_AMDBeta(self, p_index):
        """Get the beta value for the pair i-1 and i"""
        eta = self.get_RelAMD(p_index)
        HCc = self.get_Hill_CritC(p_index)
        betaH = eta/HCc
        if betaH < 1.0:
            self.planets[p_index].AMDBeta = betaH
            return betaH
        elif betaH >= 1.0:
            LCc = self.get_Laskar_CritC(p_index)
            betaL = eta/LCc
            self.planets[p_index].AMDBeta = betaL
            return betaL

    def get_alpha(self, p_index):
        if p_index == 0:
            return 0.0
        inner_p = self.planets[p_index-1]
        outer_p = self.planets[p_index]
        return (inner_p.period/outer_p.period)**(2./3)


    def get_epsilon(self, p_index):
        inner_p = self.planets[p_index-1]
        outer_p = self.planets[p_index]
        if p_index == 0:
            return outer_p.mass/(self.host_star).mass
        return (inner_p.mass+outer_p.mass)/(self.host_star).mass

    def get_Laskar_CritC(self, p_index):
        """Get C_crit for the pair i-1 and i"""
        inner_p = self.planets[p_index-1]
        outer_p = self.planets[p_index]
        if p_index == 0:
            # Inner planet returns C_crit = 1
            outer_p.stability_type = "Star"
            return 1.
        alpha = (inner_p.period/outer_p.period)**(2./3)
        gamma = inner_p.mass/outer_p.mass
        epsilon = (inner_p.mass+outer_p.mass)/(self.host_star).mass
        alpha_R = sp.optimize.brentq(equation_99, 0.0, 1.0, args = (self.r, epsilon))
        # Preform Collision Restricted Critical AMD Calc
        if alpha < alpha_R:
            outer_p.stability_type = "Collision"
            inner_ec = sp.optimize.brentq(equation_35, 0.0, 1.0, args = (gamma, alpha))
            outer_ec = 1 - alpha - alpha*inner_ec
            return gamma*np.sqrt(alpha)*(1-np.sqrt(1-inner_ec**2.0)) + (1-np.sqrt(1-outer_ec**2.0))
        # Preform MMR-Overlap Restricted Critical AMD Calc
        elif alpha >= alpha_R:
            alpha_circ = 4/(3**(6/7))*(self.r*epsilon)**(2/7)
            outer_p.stability_type = "MMR"
            if alpha < alpha_circ:
                g = (81.*(1-alpha)**5.)/(512.*self.r*epsilon) - (32*self.r*epsilon)/(9*(1-alpha)**2)
            elif alpha >= alpha_circ:
                g = 0.0
            return (g**2/2.0)*(gamma*np.sqrt(alpha))/(1+gamma*np.sqrt(alpha))

    def get_Hill_CritC(self, p_index):
        inner_p = self.planets[p_index-1]
        outer_p = self.planets[p_index]
        if p_index == 0:
            outer_p.stability_type = "Star"
            return 1.
        alpha = (inner_p.period/outer_p.period)**(2./3)
        gamma = inner_p.mass/outer_p.mass
        epsilon = (inner_p.mass+outer_p.mass)/(self.host_star).mass
        c1 = alpha/(gamma+alpha)
        c2 = gamma*3**(4./3.)*epsilon**(2./3.)
        c3 = (1+gamma)
        outer_p.stability_type = "Hill"
        return 1 + gamma*np.sqrt(alpha) - c3**(3./2.)*np.sqrt(c1*(1+c2/c3**2.))

    def plot_system(self,zeropoint=1,fig=None,ax=None,clrbar=False):
        def _clrbar(ax,norm,cmap):
            import matplotlib
            import matplotlib.cm
            import matplotlib.colors as colors
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top',size='3%',pad=.05)
            cmap = matplotlib.cm.get_cmap(cmap)
            cb = matplotlib.colorbar.ColorbarBase(ax=cax,cmap=cmap,norm=norm,orientation='horizontal')
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')

        import matplotlib.colors as colors
        norm = colors.LogNorm(vmin=1e-2,vmax=1e2)
        if ax is None:
            fig,ax=plt.subplots(figsize=(8,3))
            ax.set_ylim(0,2)
            ax.set_xscale('log')
            ax.set_xlim(1e-3,1e5)
            ax.set_xlabel('Period [days]')
            ax.minorticks_on()
            ax.set_yticklabels('')

        if clrbar:
            _clrbar(ax,norm,'coolwarm')
        ax.axhline(zeropoint,c='k',ls=':',zorder=0)
        print('For ', self.system_name, ':')
        for i, planet in enumerate(self.planets):
            beta = self.get_AMDBeta(i)
            stest = planet.stability_type
            mass = planet.mass
            period = planet.period
            size = 200*np.log10(mass.value_in(units.MJupiter) * 1e-2/(3e-6) / (.1))
            print('Planet #', i, 'has a AMDBeta of', beta, 'from the', stest, 'stability test.')
            ax.scatter(period.value_in(units.day),zeropoint,s=size,c=beta,cmap='coolwarm',norm=norm)
        return fig,ax

#------------------------------------------------------------------------------#
#-The following function returns a list to match planets with their host stars-#
#------------------------------------------------------------------------------#

def get_planetary_systems_from_set(bodies, converter=None, RelativePosition=False):
    # Initialize Kepler
    if converter == None:
        converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
    kep_p = Kepler(unit_converter = converter, redirection = 'none')
    kep_p.initialize_code()
    kep_s = Kepler(unit_converter = converter, redirection = 'none')
    kep_s.initialize_code()
    # Seperate Out Planets and Stars from Bodies
    stars, planets = util.get_stars(bodies), util.get_planets(bodies)
    num_stars, num_planets = len(stars), len(planets)
    # Initialize the Dictionary that Contains all Planetary Systems
    systems = {}
    # Start Looping Through Stars to Find Bound Planets
    for star in stars:
        system_id = star.id
        #star.semimajor_axis, star.eccentricity, star.period, star.true_anomaly, star.mean_anomaly, star.kep_energy, star.angular_momentum = \
        #    None, None, None, None, None, None, None
        current_system = systems.setdefault(system_id, Particles())
        current_system.add_particle(star)
        for planet in planets:
            total_mass = star.mass + planet.mass
            kep_pos = star.position - planet.position
            kep_vel = star.velocity - planet.velocity
            kep_p.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])
            a_p, e_p = kep_p.get_elements()
            if e_p < 1.0:
                # Check to See if The Stellar System is a Binary
                # Note: Things get complicated if it is ...
                noStellarHeirarchy = False
                for other_star in (stars-star):
                    kep_s.initialize_from_dyn(star.mass + other_star.mass, star.x - other_star.x, star.y - other_star.y, star.z - other_star.z,
                                              star.vx - other_star.vx, star.vy - other_star.vy, star.vz - other_star.vz)
                    a_s, e_s = kep_s.get_elements()
                    r_apo = kep_s.get_apastron()
                    HillR = util.calc_HillRadius(a_s, e_s, other_star.mass, star.mass)
                    if e_s >= 1.0 or HillR < r_apo:
                        noStellarHeirarchy = True
                    else:
                        noStellarHeirarchy = False
                if noStellarHeirarchy:
                    # Get Additional Information on Orbit
                    planet.semimajor_axis = a_p
                    planet.eccentricity = e_p
                    planet.period = kep_p.get_period()
                    planet.true_anomaly, planet.mean_anomaly = kep_p.get_angles()
                    #planet.kep_energy, planet.angular_momentum = kep_p.get_integrals()
                    # Add the Planet to the System Set
                    current_system.add_particle(planet)
                else:
                    # Handling for Planetary Systems in Stellar Heirarchical Structures
                    # Note: This is empty for now, maybe consider doing it by the heaviest bound stellar object as the primary.
                    pass
            else:
                continue
    kep_p.stop()
    kep_s.stop()
    return systems

# Note: The below function is nearly identical to the above function. However,
#       it is to be used for determining "clumps" of systems for the CutOrAdvance
#       function primarily. ~ Joe G. 4/1/20
# Note: This was updated to deal with stars who are bound but not mutually their
#       respected closest neighbours. ~ Joe G. 8/21/20
def get_heirarchical_systems_from_set(bodies, kepler_workers=None, converter=None, RelativePosition=False):
    # Initialize Kepler
    if kepler_workers == None:
        if converter == None:
            converter = nbody_system.nbody_to_si(bodies.mass.sum(), 2 * np.max(bodies.radius.number) | bodies.radius.unit)
        kep_p = Kepler(unit_converter = converter, redirection = 'none')
        kep_p.initialize_code()
        kep_s = Kepler(unit_converter = converter, redirection = 'none')
        kep_s.initialize_code()
    else:
        kep_p = kepler_workers[0]
        kep_s = kepler_workers[1]
    # Seperate Out Planets and Stars from Bodies
    stars, planets = util.get_stars(bodies), util.get_planets(bodies)
    num_stars, num_planets = len(stars), len(planets)
    # Initialize the Dictionary that Contains all Planetary Systems
    systems = {}
    # Initialize the List Used to Check Star IDs Against Already Classified Binaries
    binary_ids = []
    # Find Nearest Neighbors of the Set
    closest_neighbours = stars.nearest_neighbour()
    # Start Looping Through Stars to Find Bound Planets
    for index, star in enumerate(stars):
        # If the star is already in Binary_IDs, just go to the next star.
        if star.id in binary_ids:
            continue
        # If not, Set the System ID and Set-up Data Structure.
        system_id = star.id
        current_system = systems.setdefault(system_id, Particles())
        current_system.add_particle(star)
        noStellarHierarchy = False
        # If there is only one stars, there is obviously no stellar heirarchy in
        # the encounter that is occuring.
        if len(stars) == 1:
            noStellarHierarchy = True
        if not noStellarHierarchy:
            # Check to see if the Nearest Neighbor is Mutual
            star_neighbour_id = closest_neighbours[index].id
            neighbour_neighbour_id = closest_neighbours[stars.id == star_neighbour_id].id[0]
            for other_star in (stars-star):
                # Check to see if the two stars are bound.
                kep_s.initialize_from_dyn(star.mass + other_star.mass, star.x - other_star.x, star.y - other_star.y, star.z - other_star.z,
                                      star.vx - other_star.vx, star.vy - other_star.vy, star.vz - other_star.vz)
                a_s, e_s = kep_s.get_elements()
                print(star.id, other_star.id, e_s)
                # If they ARE NOT bound ...
                if e_s >= 1.0:
                    noStellarHierarchy = True
                # If they ARE bound ...
                else:
                    # If the star is the star's neighbour's neighbour and visa-versa, then proceed.
                    print(star.id, other_star.id, star_neighbour_id, neighbour_neighbour_id)
                    if star.id == neighbour_neighbour_id and other_star.id == star_neighbour_id:
                        noStellarHierarchy = False
                        print("Binary composed of Star", star.id, "and Star", other_star.id, "has been detected!")
                        current_system.add_particle(other_star)
                        binary_ids.append(star.id)
                        binary_ids.append(other_star.id)
                    else:
                        print("!!! Alert: Bound Stars are not closest neighbours ...")
                        print("!!! Current Star:", star.id,"| Other Star:", other_star.id)
                        print("!!! CS's Neighbour:", star_neighbour_id, \
                              "| CS's Neighbour's Neighbour:", neighbour_neighbour_id)
    checked_planet_ids = []
    for KeyID in systems.keys():
        current_system = systems[KeyID]
        sys_stars = util.get_stars(current_system)
        noStellarHierarchy = False
        # If there is only one stars, there is obviously no stellar heirarchy in
        # the encounter that is occuring.
        if len(sys_stars) == 1:
            noStellarHierarchy = True
        for planet in planets:
            if planet.id in checked_planet_ids:
                continue
            star = sys_stars[sys_stars.id == KeyID][0]
            total_mass = star.mass + planet.mass
            kep_pos = star.position - planet.position
            kep_vel = star.velocity - planet.velocity
            kep_p.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])
            a_p, e_p = kep_p.get_elements()
            P_p =  kep_p.get_period()
            Ta_p, Ma_p = kep_p.get_angles()
            host_star_id = star.id
            if e_p < 1.0:
                # Check to See if The Planetary System is tied to a Stellar Binary
                # Note: Things get complicated if it is ...
                if noStellarHierarchy:
                    # Get Additional Information on Orbit
                    planet.semimajor_axis = a_p
                    planet.eccentricity = e_p
                    planet.period = P_p
                    planet.true_anomaly = Ta_p
                    planet.mean_anomaly = Ma_p
                    planet.host_star = star.id
                    # Add the Planet to the System Set
                    current_system.add_particle(planet)
                else:
                    # Handling for Planetary Systems in Stellar Heirarchical Structures
                    # Note: We check to see which other star in the current systems
                    #       have a better boundness with the planet and choose that
                    #       as the new host star.
                    for other_star in sys_stars-star:
                        total_mass = other_star.mass + planet.mass
                        kep_pos = other_star.position - planet.position
                        kep_vel = other_star.velocity - planet.velocity
                        kep_p.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])
                        a_p2, e_p2 = kep_p.get_elements()
                        # Check to see if the planet is more bound to 'star' or
                        # 'other_star'. If its more bound to 'other_star',
                        # set the attributes to the more bound object. This will
                        # replace *_p with the better values with each loop.
                        if e_p2 < e_p:
                            a_p = a_p2
                            e_p = e_p2
                            P_p =  kep_p.get_period()
                            Ta_p, Ma_p = kep_p.get_angles()
                            host_star_id = other_star.id
                    planet.semimajor_axis = a_p
                    planet.eccentricity = e_p
                    planet.period = P_p
                    planet.true_anomaly = Ta_p
                    planet.mean_anomaly = Ma_p
                    planet.host_star = host_star_id
                    # Add the Planet to the System Set
                    current_system.add_particle(planet)
                checked_planet_ids.append(planet.id)
            elif not noStellarHierarchy:
                # Handling for Planetary Systems in Stellar Heirarchical Structures
                # Note: We check to see which other star in the current systems
                #       have a better boundness with the planet and choose that
                #       as the new host star.
                for other_star in sys_stars-star:
                    total_mass = other_star.mass + planet.mass
                    kep_pos = other_star.position - planet.position
                    kep_vel = other_star.velocity - planet.velocity
                    kep_p.initialize_from_dyn(total_mass, kep_pos[0], kep_pos[1], kep_pos[2], kep_vel[0], kep_vel[1], kep_vel[2])
                    a_p2, e_p2 = kep_p.get_elements()
                    # Check to see if the planet is more bound to 'star' or
                    # 'other_star'. If its more bound to 'other_star',
                    # set the attributes to the more bound object. This will
                    # replace *_p with the better values with each loop.
                    if e_p2 < e_p:
                        a_p = a_p2
                        e_p = e_p2
                        P_p =  kep_p.get_period()
                        Ta_p, Ma_p = kep_p.get_angles()
                        host_star_id = other_star.id
                planet.semimajor_axis = a_p
                planet.eccentricity = e_p
                planet.period = P_p
                planet.true_anomaly = Ta_p
                planet.mean_anomaly = Ma_p
                planet.host_star = host_star_id
                # Add the Planet to the System Set
                current_system.add_particle(planet)
            else:
                print("!!! Alert: Planet is not bound nor is it bound to any other star.")
    if kepler_workers == None:
        kep_p.stop()
        kep_s.stop()
    return systems
