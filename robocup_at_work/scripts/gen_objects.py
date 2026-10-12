#!/usr/bin/env python3
"""Generate the @Work object models and place them on the tables.

Writes models/, the object block in robocup_arena.world and
config/objects_manifest.json. Edit this, not the world.

    python3 scripts/gen_objects.py
    python3 scripts/gen_objects.py --scale 2
    python3 scripts/gen_objects.py --dry-run
"""

import argparse
import json
import math
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
MODELS_DIR = os.path.join(PKG, 'models')
WORLD = os.path.join(PKG, 'worlds', 'robocup_arena.world')
MANIFEST = os.path.join(PKG, 'config', 'objects_manifest.json')

# size multiplier, 1.0 = real @Work size
SCALE = 1.0

TABLE_TOP_Z = 0.10  # all tables are 0.1 m

BEGIN = '    <!-- objects: built by gen_objects.py, do not edit between markers -->'
END = '    <!-- end objects -->'


# @Work sizes in m, long axis on +X. Colours are not competition
# accurate, just distinct.
OBJECTS = {
    'F20_20_B': dict(
        model='f20_20_b',
        desc='Aluminium profile, 20x20 mm cross-section, 100 mm long.',
        mass=0.050,
        bbox=(0.100, 0.020, 0.020),
        rest_z=0.010,          # origin height above the table
        parts=[dict(kind='box', size=(0.100, 0.020, 0.020), pose=(0, 0, 0, 0, 0, 0),
                    rgba=(0.15, 0.30, 0.85, 1))],
    ),
    'S40_40_G': dict(
        model='s40_40_g',
        desc='Aluminium profile, 40x40 mm cross-section, 100 mm long.',
        mass=0.150,
        bbox=(0.100, 0.040, 0.040),
        rest_z=0.020,
        parts=[dict(kind='box', size=(0.100, 0.040, 0.040), pose=(0, 0, 0, 0, 0, 0),
                    rgba=(0.15, 0.65, 0.25, 1))],
    ),
    'M20_100': dict(
        model='m20_100',
        desc='M20 bolt, 100 mm long: 85 mm shaft plus a 15 mm head, lying down.',
        mass=0.200,
        bbox=(0.100, 0.029, 0.029),
        # rests on the head, shaft tilts ~2.8 deg
        rest_z=0.0145,
        parts=[
            # shaft along +X
            dict(kind='cylinder', radius=0.010, length=0.085,
                 pose=(0.0075, 0, 0, 0, math.pi / 2, 0),
                 rgba=(0.75, 0.75, 0.78, 1)),
            # head
            dict(kind='cylinder', radius=0.0145, length=0.015,
                 pose=(-0.0425, 0, 0, 0, math.pi / 2, 0),
                 rgba=(0.35, 0.35, 0.38, 1)),
        ],
    ),
    'R20': dict(
        model='r20',
        desc='R20 plastic tube, 20 mm diameter, 40 mm tall, standing upright.',
        mass=0.010,
        bbox=(0.020, 0.020, 0.040),
        rest_z=0.020,
        parts=[dict(kind='cylinder', radius=0.010, length=0.040,
                    pose=(0, 0, 0, 0, 0, 0), rgba=(0.85, 0.15, 0.15, 1))],
    ),
    'Bearing': dict(
        model='bearing',
        desc='Ball bearing, 32 mm outer diameter, 12 mm thick, lying flat.',
        mass=0.060,
        bbox=(0.032, 0.032, 0.012),
        rest_z=0.006,
        parts=[dict(kind='cylinder', radius=0.016, length=0.012,
                    pose=(0, 0, 0, 0, 0, 0), rgba=(0.25, 0.25, 0.28, 1))],
    ),
}

# (class, workstation, x, y, yaw) in world frame, z comes from rest_z
PLACEMENTS = [
    # WS01 - table centre (-2.0, 2.5), 0.8 x 0.5; robot at (-2.00, 1.75) yaw +90
    ('F20_20_B', 'WS01', -2.20, 2.37, 0.0),
    ('M20_100',  'WS01', -2.03, 2.37, 0.0),
    ('Bearing',  'WS01', -1.90, 2.40, 0.0),

    # WS02 - table centre (-3.3, 0.0), 0.5 x 0.8; robot at (-2.50, 0.00) yaw 180
    ('S40_40_G', 'WS02', -3.17, -0.18, math.pi / 2),
    ('Bearing',  'WS02', -3.17, -0.02, 0.0),
    ('M20_100',  'WS02', -3.17,  0.12, math.pi / 2),

    # WS03 - table centre (2.0, 2.5), 0.8 x 0.5; robot at (2.00, 1.75) yaw +90
    ('S40_40_G', 'WS03', 1.80, 2.38, 0.0),
    ('R20',      'WS03', 1.97, 2.38, 0.0),
    ('F20_20_B', 'WS03', 2.12, 2.38, 0.0),

    # WS04 - table centre (3.3, -1.5), 0.5 x 0.8; robot at (2.50, -1.50) yaw 0
    ('R20',      'WS04', 3.17, -1.60, 0.0),
    ('F20_20_B', 'WS04', 3.17, -1.44, math.pi / 2),
    ('S40_40_G', 'WS04', 3.17, -1.28, math.pi / 2),
]


def box_inertia(m, lx, ly, lz):
    return (m * (ly * ly + lz * lz) / 12.0,
            m * (lx * lx + lz * lz) / 12.0,
            m * (lx * lx + ly * ly) / 12.0)


def geometry_xml(part, s):
    if part['kind'] == 'box':
        x, y, z = (d * s for d in part['size'])
        return '<box><size>%.6g %.6g %.6g</size></box>' % (x, y, z)
    return ('<cylinder><radius>%.6g</radius><length>%.6g</length></cylinder>'
            % (part['radius'] * s, part['length'] * s))


def pose_xml(pose, s):
    """Scale the translation only."""
    x, y, z, r, p, yw = pose
    return '%.6g %.6g %.6g %.6g %.6g %.6g' % (x * s, y * s, z * s, r, p, yw)


# stiffer contacts so light objects don't jitter or sink
SURFACE = """          <surface>
            <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
            <contact><ode>
              <kp>1e6</kp><kd>100</kd>
              <min_depth>0.0005</min_depth><max_vel>0.0</max_vel>
            </ode></contact>
          </surface>"""


def model_sdf(name, spec, s):
    m = spec['mass'] * (s ** 3)
    lx, ly, lz = (d * s for d in spec['bbox'])
    ixx, iyy, izz = box_inertia(m, lx, ly, lz)

    body = []
    for i, part in enumerate(spec['parts']):
        geom = geometry_xml(part, s)
        pose = pose_xml(part['pose'], s)
        r, g, b, a = part['rgba']
        body.append(
            '        <collision name="c%d">\n'
            '          <pose>%s</pose>\n'
            '          <geometry>%s</geometry>\n'
            '%s\n'
            '        </collision>\n'
            '        <visual name="v%d">\n'
            '          <pose>%s</pose>\n'
            '          <geometry>%s</geometry>\n'
            '          <material>\n'
            '            <ambient>%.3g %.3g %.3g %.3g</ambient>\n'
            '            <diffuse>%.3g %.3g %.3g %.3g</diffuse>\n'
            '            <specular>0.2 0.2 0.2 1</specular>\n'
            '          </material>\n'
            '        </visual>'
            % (i, pose, geom, SURFACE, i, pose, geom, r, g, b, a, r, g, b, a))

    return (
        '<?xml version="1.0" ?>\n'
        '<!-- built by gen_objects.py at scale %.4g, edit that not this -->\n'
        '<sdf version="1.6">\n'
        '  <model name="%s">\n'
        '    <link name="link">\n'
        '      <inertial>\n'
        '        <mass>%.6g</mass>\n'
        '        <inertia>\n'
        '          <ixx>%.6g</ixx><ixy>0</ixy><ixz>0</ixz>\n'
        '          <iyy>%.6g</iyy><iyz>0</iyz>\n'
        '          <izz>%.6g</izz>\n'
        '        </inertia>\n'
        '      </inertial>\n'
        '%s\n'
        '      <velocity_decay><linear>0.01</linear><angular>0.01</angular></velocity_decay>\n'
        '    </link>\n'
        '    <self_collide>false</self_collide>\n'
        '  </model>\n'
        '</sdf>\n'
        % (s, spec['model'], m, ixx, iyy, izz, '\n'.join(body)))


def model_config(name, spec):
    return (
        '<?xml version="1.0"?>\n'
        '<model>\n'
        '  <name>%s</name>\n'
        '  <version>1.0</version>\n'
        '  <sdf version="1.6">model.sdf</sdf>\n'
        '  <description>\n'
        '    RoboCup@Work object %s. %s\n'
        '    Built by gen_objects.py - edit that, not this.\n'
        '  </description>\n'
        '</model>\n' % (spec['model'], name, spec['desc']))


def world_block(s):
    lines = [BEGIN,
             '    <!-- %d objects at scale %.4g -->' % (len(PLACEMENTS), s)]
    counts = {}
    instances = []
    for cls, ws, x, y, yaw in PLACEMENTS:
        spec = OBJECTS[cls]
        counts[cls] = counts.get(cls, 0) + 1
        inst = '%s_%d' % (spec['model'], counts[cls])
        z = TABLE_TOP_Z + spec['rest_z'] * s
        lines.append(
            '    <include>\n'
            '      <uri>model://%s</uri>\n'
            '      <name>%s</name>\n'
            '      <pose>%.6g %.6g %.6g 0 0 %.6g</pose>\n'
            '    </include>' % (spec['model'], inst, x, y, z, yaw))
        instances.append(dict(name=inst, cls=cls, workstation=ws,
                              pose=dict(x=x, y=y, z=z, yaw=yaw)))
    lines.append(END)
    return '\n'.join(lines) + '\n', instances


def splice(world_text, block):
    """Insert or replace the block, safe to re-run."""
    if BEGIN in world_text and END in world_text:
        head = world_text[:world_text.index(BEGIN)]
        tail = world_text[world_text.index(END) + len(END):].lstrip('\n')
        return head + block + tail
    idx = world_text.rindex('</world>')
    return world_text[:idx] + block + '\n' + world_text[idx:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, default=SCALE)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    s = args.scale

    block, instances = world_block(s)

    if args.dry_run:
        print(block)
        for name, spec in OBJECTS.items():
            print('--- models/%s/model.sdf ---' % spec['model'])
            print(model_sdf(name, spec, s))
        return

    for name, spec in OBJECTS.items():
        d = os.path.join(MODELS_DIR, spec['model'])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'model.sdf'), 'w') as f:
            f.write(model_sdf(name, spec, s))
        with open(os.path.join(d, 'model.config'), 'w') as f:
            f.write(model_config(name, spec))
        print('wrote models/%s/' % spec['model'])

    with open(WORLD) as f:
        text = f.read()
    if BEGIN not in text:
        bak = WORLD + '.bak_%s_objects' % time.strftime('%Y%m%d')
        if not os.path.exists(bak):
            shutil.copy2(WORLD, bak)
            print('backed up world -> %s' % os.path.basename(bak))
    with open(WORLD, 'w') as f:
        f.write(splice(text, block))
    print('spliced %d object instances into worlds/robocup_arena.world'
          % len(instances))

    manifest = dict(
        scale=s,
        table_top_z=TABLE_TOP_Z,
        built_by='scripts/gen_objects.py',
        classes=[dict(name=n,
                      model=sp['model'],
                      index=i,
                      bbox=[d * s for d in sp['bbox']],
                      mass=sp['mass'] * (s ** 3),
                      rest_z=sp['rest_z'] * s)
                 for i, (n, sp) in enumerate(sorted(OBJECTS.items()))],
        instances=instances,
    )
    with open(MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)
    print('wrote config/objects_manifest.json (%d classes, %d instances)'
          % (len(OBJECTS), len(instances)))


if __name__ == '__main__':
    main()
