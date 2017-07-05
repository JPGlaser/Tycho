'''
encounter_db.py

Provides methods to write a list of stellar close encounters to files in a systematic way.
The provided output is meant to be later loaded in a systematic way.

Each star cluster run is saved in a directory, categorized by the cluster parameters.
To load multiple runs, `os.walk` may be used.
'''

import os
import time
import hashlib
import zlib
import json

from amuse.units import units

ENCOUNTER_FILENAME = 'encounters.txt'
CLUSTER_FILENAME = 'cluster.txt'
CLUSTER_PARAMS_FILENAME = 'cluster_params.json'

def make_run_id():
    return hex(zlib.crc32(str(time.mktime(time.gmtime()))) & 0xffffffff)

# Clean up all the bodies in an encounter
class EncounterBody(object):
    """
    A model for a single star of an encounter. All attributes should be in
    SI units.
    """
# Set the properties of self   
    def __init__(self, id, mass, radius, pos, velocity):
        self.id = id
        self.mass = mass
        self.radius = radius
        self.pos = pos
        self.velocity = velocity
# Copy the properties over from the particle to self
    def copy_to_particle(self, particle):
        particle.id = self.id
        particle.mass = self.mass
        particle.radius = self.radius
        particle.position = self.pos
        particle.velocity = self.velocity

# Convert all bodys to si units
    @staticmethod
    def from_particle(part, conv):
        id = part.id
        mass = conv.to_si(part.mass)
        radius = conv.to_si(part.radius)
        pos = [
            conv.to_si(part.x),
            conv.to_si(part.y),
            conv.to_si(part.z)
        ]
        velocity = [
            conv.to_si(part.vx),
            conv.to_si(part.vy),
            conv.to_si(part.vz)
        ]

        return EncounterBody(id, mass, radius, pos, velocity)

    def __repr__(self):
        return '<Body {0}: mass={1}>'.format(self.id, self.mass)


class OrbitParams(object):
    """
    A model for binary orbital parameters of an encounter.
    All units should be in SI.
    """
    def __init__(self, M, a, e, r, E, t):
        self.M = M
        self.a = a
        self.e = e
        self.r = r
        self.E = E
        self.t = t

# calulate peri and apo
    @property
    def peri(self):
        return abs(self.a.number* (1.0 - self.e)) | units.AU
    @property
    def apo(self):
        return abs(self.a.number * (1.0 + self.e)) | units.AU

# convert parameters to si
    @staticmethod
    def from_nbody_params(M, a, e, r, E, t, conv):
        return OrbitParams(
            conv.to_si(M),
            conv.to_si(a),
            conv.to_si(e),
            conv.to_si(r),
            conv.to_si(E),
            conv.to_si(t))
            
# Records all relevant data of an encounter
class Encounter(object):
    """
    All data for a single encounter.
    """
    def __init__(self, bodies, orbit_params, time, id=None):
        self.bodies = bodies
        self.orbit_params = orbit_params
        self.time = time
        self.id = id

# Return time, ids, object (star planet), and orbital params
    def __repr__(self):
        return \
'''<Encounter {8} @ t={0} 
\tperi={5}, r_init={6}, ecc={7}
\tBody {1}: {3}
\tBody {2}: {4}
>''' \
            .format(self.time.value_in(units.Myr), self.bodies[0].id,
                    self.bodies[1].id, self.bodies[0], self.bodies[1],
                    self.orbit_params.peri, self.orbit_params.r, self.orbit_params.e,
                    self.id)

# This class establishes how many n_bodies were in a cluster and what time the cluster ended
class ClusterParameters(object):
    def __init__(self, n_bodies, t_end):
        self.n_bodies = n_bodies
        self.t_end = t_end

# This finds the directory the cluster was stored in
    def get_dir_string(self):
        return 'king_n={0}'.format(self.n_bodies)


class EncounterDbWriter(object):
    """
    Writer class for emitting encounter records in the encounter db.
    A single EncounterDbWriter instance is intended to write all data for
    a given cluster.
    """
    def __init__(self, root_directory, cluster_params):
        self.root_directory = root_directory
        self.run_id = make_run_id()
        self.run_id_str = EncounterDbWriter.make_run_id_string(self.run_id)
        self.directory = self.root_directory
        self.cluster_params = cluster_params

        if cluster_params is not None:
            cluster_params_dir = cluster_params.get_dir_string()
            self.directory = os.path.join(self.root_directory, cluster_params_dir)

        self.directory = os.path.join(self.directory, self.run_id_str)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.encounter_filename = os.path.join(self.directory, ENCOUNTER_FILENAME)
        self.encounter_file = open(self.encounter_filename, 'w')

        self.next_encounter_id = 0

    def output_directory_root(self):
        return self.directory

    @staticmethod
    def make_run_id_string(run_id):
        date_part = time.strftime('%Y_%m_%d-%H%M%S')
        return '{0}_{1}'.format(date_part, run_id)
        
    def finalize(self, end_time, expected_end, ex=None):
        param_obj = {
            'end_time': end_time.number,
            'target_time': expected_end.number,
            'num_bodies': self.cluster_params.n_bodies,
            'exception': None
        }
        if ex is not None:
            param_obj['exception'] = ex
        filename = os.path.join(self.directory, CLUSTER_PARAMS_FILENAME)
        with open(filename, 'w') as json_file:
            json_file.write(json.dumps(param_obj))
        self.encounter_file.close()

# Write out key information about the cluster
    def write_cluster(self, stars, conv):
        cluster_filename = os.path.join(self.directory, CLUSTER_FILENAME)
        with open(cluster_filename, 'w') as f:
            for star in stars:
                fields = [
                    star.id,
                    conv.to_si(star.mass).value_in(units.MSun),
                    conv.to_si(star.radius).value_in(units.RSun),
                    conv.to_si(star.x).value_in(units.AU),
                    conv.to_si(star.y).value_in(units.AU),
                    conv.to_si(star.z).value_in(units.AU),
                    conv.to_si(star.vx).value_in(units.kms),
                    conv.to_si(star.vy).value_in(units.kms),
                    conv.to_si(star.vz).value_in(units.kms)
                ]
                f.write('\t'.join([str(x) for x in fields]) + '\n')

# Write out information about the encounters
    def write_encounter(self, encounter, conv):
        if encounter.id is None:
            encounter.id = self.next_encounter_id
        self.next_encounter_id += 1

        orbit = encounter.orbit_params
        fields = [
            encounter.time.value_in(units.Myr),
            orbit.peri.value_in(units.AU),
            orbit.apo.value_in(units.AU),
            orbit.r.value_in(units.AU),
            orbit.e,
            orbit.a.value_in(units.AU),
            orbit.M.value_in(units.MSun),
            orbit.E.value_in(units.m**2 / units.s**2),
            self.next_encounter_id
        ]
        for star in encounter.bodies:
            star_params = [
                star.mass.value_in(units.MSun),
                star.radius.value_in(units.km),
                star.pos[0].value_in(units.AU),
                star.pos[1].value_in(units.AU),
                star.pos[2].value_in(units.AU),
                star.velocity[0].value_in(units.kms),
                star.velocity[1].value_in(units.kms),
                star.velocity[2].value_in(units.kms),
                star.id
            ]
            fields.extend(star_params)

        self.encounter_file.write(
            '\t'.join([str(x) for x in fields]) + '\n')
        self.encounter_file.flush()

class EncounterDbReader(object):
    BODY_IDX_START = 9
    BODY_IDX_SIZE = 9

    def __init__(self, directory):
        self.directory = directory

        if not os.path.exists(self.directory):
            raise IOError('Specified encounter directory does not exist')
        
    def encounters(self):
        encounter_filename = os.path.join(self.directory, ENCOUNTER_FILENAME)
        with open(encounter_filename, 'r') as f:
            for line in f:
                fields = line.split('\t')
                orbit = self._parse_orbital_params(fields)
                star1 = self._parse_body(fields, 0)
                star2 = self._parse_body(fields, 1)
                time = float(fields[0]) | units.Myr
                encounter_id = int(fields[8])

                yield Encounter([star1, star2], orbit, time, encounter_id)

# Retrieve params from a cluster 
    def get_cluster_details(self):
        details_filename = os.path.join(self.directory, CLUSTER_PARAMS_FILENAME)
        with open(details_filename, 'r') as f:
            params = json.load(f)
            return params

# Retrieve information about the stars in the cluster
    def cluster_stars(self):
        cluster_filename = os.path.join(self.directory, CLUSTER_FILENAME)
        with open(cluster_filename, 'r') as f:
            for line in f:
                fields = line.split('\t')
                id = fields[0],
                mass = float(fields[1]) | units.MSun
                radius = float(fields[2]) | units.AU
                pos = [float(x) | units.AU for x in fields[3:6]]
                velocity = [float(x) | units.kms for x in fields[6:9]]

                yield EncounterBody(id, mass, radius, pos, velocity)

# Retrieve orbital params
    def _parse_orbital_params(self, fields):
        time = float(fields[0]) | units.Myr
        r = float(fields[3]) | units.AU
        e = float(fields[4])
        a = float(fields[5]) | units.AU
        M = float(fields[6]) | units.MSun
        E = float(fields[7]) | (units.m**2 / units.s**2)

        return OrbitParams(M, a, e, r, E, time)
# Sets different attributes their correct datatype then calls EncounterBody
    def _parse_body(self, fields, body_idx):
        idx_base = self.BODY_IDX_START + body_idx*self.BODY_IDX_SIZE

        id = str(fields[idx_base+8]).strip()
        mass = float(fields[idx_base+0]) | units.MSun
        radius = float(fields[idx_base+1]) | units.km
        pos = [float(fields[idx_base+i]) | units.AU for i in [2, 3, 4]]
        velocity = [float(fields[idx_base+i]) | units.kms for i in [5, 6, 7]]

        return EncounterBody(id, mass, radius, pos, velocity)
