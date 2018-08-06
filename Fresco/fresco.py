# -*- coding: utf-8 -*-
"""
Fresco creates a "simulated observation" of a set of particles.
Particles can be "stars" (point sources emitting light) or "gas" (emitting,
reflecting and/or obscuring light). Gas may also be displayed with contour
lines.
"""

from __future__ import (
        print_function,
        division,
        absolute_import,
        )

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.io import read_set_from_file
from amuse.datamodel.rotation import rotate
#from amuse.ext.galactic_potentials import MWpotentialBovy2015

from fresco.ubvinew import rgb_frame
from fresco.fieldstars import new_field_stars

from scipy.ndimage import gaussian_filter

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse


def new_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--filetype',
            dest='filetype',
            default='amuse',
            help='filetype [amuse], valid are amuse,starlab,txt,...',
            )
    parser.add_argument(
            '-s',
            dest='starsfilename',
            default='',
            help='file containing stars (optional) []',
            )
    parser.add_argument(
            '-g',
            dest='gasfilename',
            default='',
            help='file containing gas (optional) []',
            )
    parser.add_argument(
            '-o',
            dest='imagefilename',
            default='test',
            help='write image to this file [test]',
            )
    parser.add_argument(
            '--imagetype',
            dest='imagetype',
            default='png',
            help='image file type [png]',
            )
    parser.add_argument(
            '-b',
            dest='sourcebands',
            default='ubvri',
            help='colour bands to use [ubvri]',
            )
    parser.add_argument(
            '-a',
            dest='age',
            default=100.,
            type=float,
            help='age of the stars in Myr [100]',
            )
    parser.add_argument(
            '-w',
            dest='width',
            default=5.,
            type=float,
            help='image width in parsec [5]',
            )
    parser.add_argument(
            '-x',
            dest='plot_axes',
            action='store_true',
            default=False,
            help='plot axes [False]',
            )

    parser.add_argument(
           '--comx',
           dest='comx',
           type=float,
           default=None,
           help='Center of Mass (in Parsec)'
           )
    parser.add_argument(
           '--comy',
           dest='comy',
           type=float,
           default=None,
           help='Center of Mass (in Parsec)'
           )
    parser.add_argument(
           '--comz',
           dest='comz',
           type=float,
           default=None,
           help='Center of Mass (in Parsec)'
           )
    parser.add_argument(
           '--new-ev',
           dest='evolution_flag',
           action='store_true',
           default=False,
           help='Toggle b/w old and new evolution'
           )
#    parser.add_argument(
#           '-t',
#           dest='elapsed_time',
#           type=float,
#           default=None,
#           help='time between steps (Myr)'
#           )

    parser.add_argument(
            '--ext',
            dest='calculate_extinction',
            action='store_true',
            default=False,
            help='include extinction by dust [False]',
            )
    parser.add_argument(
            '--seed',
            dest='seed',
            default=1701,
            type=int,
            help='random seed',
            )
    parser.add_argument(
            '--vmax',
            dest='vmax',
            default=0,
            type=float,
            help='vmax value',
            )
    parser.add_argument(
            '--field',
            dest='n_fieldstars',
            default=0,
            type=int,
            help='add N field stars (optional) [0]',
            )
    parser.add_argument(
            '--ax',
            dest='angle_x',
            default=0,
            type=float,
            help='Rotation step around x-axis in deg [0]',
            )
    parser.add_argument(
            '--ay',
            dest='angle_y',
            default=0,
            type=float,
            help='Rotation step around y-axis in deg [0]',
            )
    parser.add_argument(
            '--az',
            dest='angle_z',
            default=0,
            type=float,
            help='Rotation step around z-axis in deg [0]',
            )
    parser.add_argument(
            '--frames',
            dest='frames',
            default=1,
            type=int,
            help='Number of frames (>1: rotate around x,y,z) [1]',
            )
    parser.add_argument(
            '--px',
            dest='pixels',
            default=2048,
            type=int,
            help='Number of pixels along each axis [2048]',
            )
    parser.add_argument(
            '--psf',
            dest='psf_type',
            default='hubble',
            help='PSF type (valid: [hubble], gaussian)',
            )
    parser.add_argument(
            '--sigma',
            dest='psf_sigma',
            default=1.0,
            type=float,
            help='PSF sigma (if PSF type is gaussian)',
            )
    parser.add_argument(
            '--contours',
            dest='contours',
            action='store_true',
            default=False,
            help='Plot gas contour lines [False]',
            )
    return parser.parse_args()

def singleStarEvolution(star, end_time, se):
    # First attempt to evolve the star regularly
    try:
        # This method to evolve the stars could probably be changed to just be
        # se.evolve_model but im not confident in that
        star.mass = se.evolve_star(star.mass,end_time,0.02)[1]
        star.luminosity = se.evolve_star(star.mass,end_time,0.02)[3]
        star.physical_radius = se.evolve_star(star.mass,end_time,0.02)[2]
        star.age = se.evolve_star(star.mass,end_time,0.02)[0]
        star.temperature = se.evolve_star(star.mass,end_time,0.02)[4]

    # If that fails for some reason reset the stellar evolution code
    except:
	se.stop()
        if se == "SeBa":
            from amuse.community.seba.interface import SeBa
            se = SeBa()
        elif se == "SSE":
            from amuse.community.sse.interface import SSE
            se = SSE()
            # SSE can result in nan values for luminosity/radius

        star.mass = se.evolve_star(star.mass,end_time,0.02)[1]
        star.luminosity = se.evolve_star(star.mass,end_time,0.02)[3]
        star.physical_radius = se.evolve_star(star.mass,end_time,0.02)[2]
        star.age = se.evolve_star(star.mass,end_time,0.02)[0]
        star.temperature = se.evolve_star(star.mass,end_time,0.02)[4]
    # return the star and stellar evolution code
    return star, se

def evolve_to_age(stars, age, evolution_flag, se="SeBa"):
    if evolution_flag:
        for star in stars:
            if se == "SeBa":
                from amuse.community.seba.interface import SeBa
                se = SeBa()
            elif se == "SSE":
                from amuse.community.sse.interface import SSE
                se = SSE()
                # SSE can result in nan values for luminosity/radius
            if star.type!="star":
                pass
            elif star.type=="star":
                se.particles.add_particle(star)

                # Try to find the stars age and if that fails set the default to 8 Myr
                try:
                    age = star.age
                except:
                    age = 8 | units.Myr

                # Actual evolution of the star
                if age > 0 | units.Myr:
                    star, se = singleStarEvolution(star, age, se)

                star.luminosity = np.nan_to_num(
                        se.particles.luminosity.value_in(units.LSun)
                        ) | units.LSun
                star.radius = (1 | units.m) + (np.nan_to_num(
                        se.particles.radius.value_in(units.RSun)
                        ) | units.RSun)

        # Make sure the se is stopped
        try:
            se.stop()
        except:
            pass

    # The old method
    else:
        if se == "SeBa":
            from amuse.community.seba.interface import SeBa
            se = SeBa()
        elif se == "SSE":
            from amuse.community.sse.interface import SSE
            se = SSE()
        
        if age > 0 | units.yr:
            try:
                se.evolve_model(age)
            except:
                try:
                    se.stop()
                except:
                    pass
                if se == "SeBa":
                    from amuse.community.seba.interface import SeBa
                    se = SeBa()
                elif se == "SSE":
                    from amuse.community.sse.interface import SSE
                    se = SSE()
                    # SSE can result in nan values for luminosity/radius
                se.evolve_model(age)

        else:
            try:
                se.evolve_model(100 | units.Myr)
            except:
                try:
                    se.stop()
       	       	except:
       	       	    pass 
                if se == "SeBa":
                    from amuse.community.seba.interface import SeBa
                    se = SeBa()
                elif se == "SSE":
                    from amuse.community.sse.interface import SSE
                    se = SSE()
                    # SSE can result in nan values for luminosity/radius
                se.evolve_model(100 | units.Myr)

        stars.luminosity = np.nan_to_num(
                se.particles.luminosity.value_in(units.LSun)
                ) | units.LSun
        # Temp fix: add one meter to radius of stars, to prevent zero/nan radius.
        # TODO: Should fix this a better way, but it's ok for now.
        stars.radius = (1 | units.m) + (np.nan_to_num(
                se.particles.radius.value_in(units.RSun)
                ) | units.RSun)
        try:
            se.stop()
        except:
            pass
    return


def calculate_effective_temperature(luminosity, radius):
    temp = np.nan_to_num(
            (
                (
                    luminosity
                    / (
                        constants.four_pi_stefan_boltzmann
                        * radius**2
                        )
                    )**.25
                ).value_in(units.K)
            ) | units.K
    return temp


def make_image(
        stars,
        gas,
        mode=["stars", "gas"],
        converter=None,
        image_width=5. | units.parsec,
        image_size=[1024, 1024],
        percentile=0.9995,
        age=0. | units.Myr,
        sourcebands="ubvri",
        vmax=None,
        calc_temperature=True,
        mapper_code=None,  # "FiMap"
        zoom_factor=1.0,
        psf_type="hubble",
        psf_sigma=1.0,
        return_vmax=False,
        ):
    """
    Makes image from gas and stars
    """
    if ("extinction" in mode):
        # Extinction can currently only be handled with FiMap
        mapper_code = "FiMap"

    if mapper_code == "FiMap":
        def mapper():
            from amuse.community.fi.interface import FiMap
            mapper = FiMap(converter, mode="openmp")

            # mapper.parameters.minimum_distance = 1. | units.AU
            mapper.parameters.image_size = image_size
            # mapper.parameters.image_target = image_target

            mapper.parameters.image_width = image_width
            # mapper.parameters.projection_direction = (
            #         (image_target-viewpoint)
            #         / (image_target-viewpoint).length()
            #         )
            # mapper.parameters.projection_mode = projection
            # mapper.parameters.image_angle = horizontal_angle
            # mapper.parameters.viewpoint = viewpoint
            mapper.parameters.extinction_flag =\
                True if "extinction" in mode else False
            return mapper
    else:
        # Gridify as default
        mapper = None
        mapper_code = "gridify"

    if "stars" not in mode:
        image = column_density_map(
                gas,
                image_width=image_width,
                image_size=image_size,
                mapper_factory=mapper,
                mapper_code=mapper_code,
                zoom_factor=zoom_factor,
                psf_type=psf_type,
                psf_sigma=psf_sigma,
                return_vmax=return_vmax,
                )
    else:
        image = image_from_stars(
                stars,
                image_width=image_width,
                image_size=image_size,
                percentile=percentile,
                calc_temperature=calc_temperature,
                age=age,
                sourcebands=sourcebands,
                gas=gas,
                vmax=vmax,
                mapper_factory=mapper,
                mapper_code=mapper_code,
                zoom_factor=zoom_factor,
                psf_type=psf_type,
                psf_sigma=psf_sigma,
                return_vmax=return_vmax,
                )
    return image


def column_density_map(
        gas,
        image_width=10. | units.parsec,
        image_size=[1024, 1024],
        mapper_factory=None,
        mapper_code=None,
        zoom_factor=1.0,
        psf_type="gaussian",
        psf_sigma=10.0,
        return_vmax=False,
        ):
    if mapper_code == "FiMap":
        if callable(mapper_factory):
            mapper = mapper_factory()

        p = mapper.particles.add_particles(gas)
        p.weight = gas.mass.value_in(units.amu)
        projected = mapper.image.pixel_value
        mapper.stop()
        im = gaussian_filter(
                projected,
                sigma=psf_sigma*zoom_factor,
                order=0,
                )
    else:
        from fresco.gridify import map_to_grid
        gas_in_mapper = gas.copy()
        gas_in_mapper.weight = gas_in_mapper.mass.value_in(units.amu)
        raw_image = map_to_grid(
                gas_in_mapper.x,
                gas_in_mapper.y,
                weights=gas_in_mapper.weight,
                image_size=image_size,
                image_width=image_width,
                )
        im = gaussian_filter(
                raw_image,
                sigma=psf_sigma*zoom_factor,
                order=0,
                ).T
    if return_vmax:
        return (im, -1)
    else:
        return im


def image_from_stars(
        stars,
        image_width=10. | units.parsec,
        image_size=[1024, 1024],
        percentile=0.9995,
        calc_temperature=True,
        age=0. | units.Myr,
        sourcebands="ubvri",
        gas=None,
        vmax=None,
        mapper_factory=None,
        mapper_code=None,
        zoom_factor=1.0,
        psf_type="hubble",
        psf_sigma=1.0,
        return_vmax=False,
        ):
    if calc_temperature:
        # calculates the temperature of the stars from their total luminosity
        # and radius, calculates those first if needed
        stars.temperature = calculate_effective_temperature(
                stars.luminosity,
                stars.radius,
                )

    vmax, rgb = rgb_frame(
            stars,
            dryrun=False,
            image_width=image_width,
            vmax=vmax,
            multi_psf=False,  # True,
            image_size=image_size,
            percentile=percentile,
            sourcebands=sourcebands,
            mapper_factory=mapper_factory,
            gas=gas,
            mapper_code=mapper_code,
            zoom_factor=zoom_factor,
            psf_type=psf_type,
            psf_sigma=psf_sigma,
            )
    if return_vmax:
        return rgb['pixels'], vmax
    else:
        return rgb['pixels']


def initialise_image(
        fig=None,
        dpi=150,
        image_size=[2048, 2048],
        length_unit=units.parsec,
        image_width=5 | units.parsec,
        plot_axes=True,
        subplot=0,
        ):
    if fig is None:
        if plot_axes:
            left = 0.15
            bottom = 0.15
        else:
            left = 0.
            bottom = 0.
        right = 1.0
        top = 1.0
        figwidth = image_size[0] / dpi / (right - left)
        figheight = image_size[1] / dpi / (top - bottom)
        figsize = (figwidth, figheight)

        xmin = -0.5 * image_width.value_in(length_unit)
        xmax = 0.5 * image_width.value_in(length_unit)
        ymin = -0.5 * image_width.value_in(length_unit)
        ymax = 0.5 * image_width.value_in(length_unit)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    else:
        # Simply clear and re-use the old figure
        ax = fig.get_axes()[subplot]
        ax.cla()
    ax.set_xlabel("X (%s)" % (length_unit))
    ax.set_ylabel("Y (%s)" % (length_unit))
    ax.set_aspect(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('black')
    return fig

# My only changes to main were:
    # remove print statment about evolving stars to age b/c it was no longer true
        # this statment was moved into the stellar evolution class to show the actual age
    # add the com variables to fix a tracking issue we had

if __name__ == "__main__":
    mode = []

    # Fixed settings
    stellar_evolution = True
    se_code = "SeBa"
    length_unit = units.parsec
    dpi = 600
    percentile = 0.9995  # for determining vmax

    # Parse arguments
    args = new_argument_parser()
    starsfilename = args.starsfilename
    gasfilename = args.gasfilename
    imagefilename = args.imagefilename
    imagetype = args.imagetype
    vmax = args.vmax if args.vmax > 0 else None
    n_fieldstars = args.n_fieldstars
    filetype = args.filetype
    contours = args.contours
    np.random.seed(args.seed)
    plot_axes = args.plot_axes
    angle_x = args.angle_x | units.deg
    angle_y = args.angle_y | units.deg
    angle_z = args.angle_z | units.deg
    sourcebands = args.sourcebands
    psf_type = args.psf_type.lower()
    psf_sigma = args.psf_sigma
    age = args.age | units.Myr
    image_width = args.width | units.parsec
    pixels = args.pixels
    frames = args.frames
# The com variables are used to feed in a track spline to let us better track a cluster throughout
# its life span
    evolution_flag = args.evolution_flag
    comx = args.comx | units.parsec
    comy = args.comy | units.parsec
# comz isn't currently sed but it could be later so it is left in
    comz = args.comz | units.parsec
    #elapsed_time = args.elapsed_time | units.Myr

    # Derived settings
    if args.calculate_extinction:
        mode.append("extinction")
    if psf_type not in ["hubble", "gaussian"]:
        print(("Invalid PSF type: %s" % psf_type))
        exit()
    image_size = [pixels, pixels]
    # If the nr of pixels is changed, zoom the PSF accordingly.
    zoom_factor = pixels/2048.

    if starsfilename:
        stars = read_set_from_file(
                starsfilename,
                filetype,
                close_file=True,
                )
        if stellar_evolution and (age > 0 | units.Myr):
             # commented out the print statment that may have caused confusion
#            print((
#                    "Calculating luminosity/temperature for %s old stars..."
#                    % (age)
#                    ))
            evolve_to_age(stars, age, evolution_flag, se=se_code)
        # This is where I subtracted off the spline to keep the cluster centered in the snapshots
        if comx and comy:
            stars.x -= comx
            stars.y -= comy

    else:
        stars = Particles()

    if n_fieldstars:
        minage = 400 | units.Myr
        maxage = 12 | units.Gyr
        fieldstars = new_field_stars(
                n_fieldstars,
                width=image_width,
                height=image_width,
                )
        fieldstars.age = (
                minage
                + (
                    np.random.sample(n_fieldstars)
                    * (maxage - minage)
                    )
                )
        evolve_to_age(fieldstars, 0 | units.yr, evolution_flag, se=se_code)
        # TODO: add distance modulus
        stars.add_particles(fieldstars)
    if len(stars) > 0:
        mode.append("stars")

    if gasfilename:
        gas = read_set_from_file(
                gasfilename,
                filetype,
                close_file=True,
                )
        if "stars" not in mode:
            com = gas.center_of_mass()
        gas.position -= com
        # Gadget and Fi disagree on the definition of h_smooth.
        # For gadget, need to divide by 2 to get the Fi value (??)
        gas.h_smooth *= 0.5
        gas.radius = gas.h_smooth
    else:
        gas = Particles()
    if len(gas) > 0:
        mode.append("gas")
        if contours:
            mode.append("contours")
    # gas.h_smooth = 0.05 | units.parsec

    converter = nbody_system.nbody_to_si(
            stars.total_mass() if "stars" in mode else gas.total_mass(),
            image_width,
            )

    # Initialise figure and axes
    fig = initialise_image(
            dpi=dpi,
            image_size=image_size,
            length_unit=length_unit,
            image_width=image_width,
            plot_axes=plot_axes,
            )
    ax = fig.get_axes()[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for frame in range(frames):
        fig = initialise_image(fig)

        if (frame != 0) or (frames == 1):
            if len(stars) > 0:
                rotate(stars, angle_x, angle_y, angle_z)
            if len(gas) > 0:
                rotate(gas, angle_x, angle_y, angle_z)

        image, vmax = make_image(
                stars,
                gas,
                mode=mode,
                converter=converter,
                image_width=image_width,
                image_size=image_size,
                percentile=percentile,
                calc_temperature=True,
                age=age,
                vmax=vmax,
                sourcebands=sourcebands,
                zoom_factor=zoom_factor,
                psf_type=psf_type,
                psf_sigma=psf_sigma,
                return_vmax=True,
                )

        if "stars" in mode:
            ax.imshow(
                    image,
                    origin='lower',
                    extent=[
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        ],
                    )
            if ("contours" in mode) and ("gas" in mode):
                gascontours = column_density_map(
                        gas,
                        zoom_factor=zoom_factor,
                        image_width=image_width,
                        image_size=image_size,
                        )
                gascontours[np.isnan(gascontours)] = 0.0
                vmax = np.max(gascontours) / 2
                # vmin = np.min(image[np.where(image > 0.0)])
                vmin = vmax / 100
                levels = 10**(
                        np.linspace(
                            np.log10(vmin),
                            np.log10(vmax),
                            num=5,
                            )
                        )[1:]
                # print(vmin, vmax)
                # print(levels)
                ax.contour(
                        gascontours,
                        origin='lower',
                        levels=levels,
                        colors="white",
                        linewidths=0.1,
                        extent=[
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            ],
                        )
        else:
            image = column_density_map(
                    gas,
                    image_width=image_width,
                    image_size=image_size,
                    )

            ax.imshow(
                    image,
                    origin='lower',
                    extent=[
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        ],
                    cmap="gray",
                    )

        plt.savefig(
                "%s-%i.%s" % (
                    imagefilename,
                    frame,
                    imagetype,
                    ),
                dpi=dpi,
                )
