#!/usr/bin/env python3
"""Write the campus world, its meshes, and the planner obstacles from park_inspection.campus.

Run from the package directory after editing the layout:

    python3 tools/generate_campus.py

Outputs:
    worlds/park.sdf               the Gazebo world
    worlds/meshes/*.obj, *.png    facades, trees, street furniture, terrain, textures
    config/campus_obstacles.yaml  planner boxes and inspection points

Every visual mesh is checked against the planner boxes of its object, so what
the sensors see never pokes outside what the planner avoids.
"""

import functools
import math
import sys
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meshgen import Mesh, seeded, textures  # noqa: E402
from park_inspection import campus  # noqa: E402

MESH_DIR = "meshes"

PALETTES = (
    # wall, roof
    ((0.72, 0.72, 0.70), (0.30, 0.31, 0.33)),
    ((0.82, 0.77, 0.66), (0.36, 0.33, 0.30)),
    ((0.58, 0.62, 0.66), (0.26, 0.28, 0.30)),
    ((0.64, 0.45, 0.36), (0.30, 0.24, 0.22)),
    ((0.88, 0.88, 0.86), (0.40, 0.41, 0.42)),
    ((0.50, 0.52, 0.54), (0.22, 0.23, 0.24)),
    ((0.76, 0.70, 0.60), (0.34, 0.31, 0.28)),
)
GLASS = ((0.20, 0.30, 0.38), (0.16, 0.25, 0.32), (0.24, 0.34, 0.40))
CAR_COLORS = (
    (0.75, 0.75, 0.76), (0.08, 0.08, 0.09), (0.62, 0.08, 0.07),
    (0.90, 0.90, 0.88), (0.12, 0.22, 0.45), (0.38, 0.40, 0.42),
)
SIGN_COLORS = ((0.08, 0.30, 0.62), (0.05, 0.42, 0.22), (0.86, 0.86, 0.84))

FLOOR_HEIGHT = 3.5
CORE = 0.3
PARAPET = 0.6
ROOF_SLAB = 0.15
LOBBY_DEPTH = 1.6
FRAME = 0.07

ROAD_Z = {"north_south": 0.02, "east_west": 0.024}
LOT_Z = 0.02
PAVEMENT_Z = 0.08
FIELD_Z = 0.03
PAINT = 0.004
TERRAIN_STEP = 2.5
TERRAIN_MIN = (-180.0, -160.0)
TERRAIN_MAX = (220.0, 160.0)
TERRAIN_AMPLITUDE = 0.3


# --------------------------------------------------------------------------- SDF helpers


def fmt(value):
    text = "%.3f" % value
    text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def vec(*values):
    return " ".join(fmt(value) for value in values)


def material(rgb, roughness=0.85, metalness=0.0, texture=None, emissive=None):
    ambient = tuple(channel * 0.85 for channel in rgb)
    albedo = "<albedo_map>%s/%s</albedo_map>" % (MESH_DIR, texture) if texture else ""
    glow = "<emissive>%s 1</emissive>" % vec(*emissive) if emissive else ""
    return (
        "<material><ambient>%s 1</ambient><diffuse>%s 1</diffuse><specular>0.03 0.03 0.03 1</specular>%s"
        "<pbr><metal>%s<roughness>%s</roughness><metalness>%s</metalness></metal></pbr></material>"
        % (vec(*ambient), vec(*rgb), glow, albedo, fmt(roughness), fmt(metalness))
    )


def surface(mu, restitution=0.0):
    return (
        "<surface><friction><ode><mu>%s</mu><mu2>%s</mu2></ode></friction>"
        "<bounce><restitution_coefficient>%s</restitution_coefficient></bounce></surface>"
        % (fmt(mu), fmt(mu), fmt(restitution))
    )


def box_geometry(size):
    return "<box><size>%s</size></box>" % vec(*size)


def cylinder_geometry(radius, length):
    return "<cylinder><radius>%s</radius><length>%s</length></cylinder>" % (fmt(radius), fmt(length))


def sphere_geometry(radius):
    return "<sphere><radius>%s</radius></sphere>" % fmt(radius)


def mesh_geometry(file_name):
    return "<mesh><uri>%s/%s</uri></mesh>" % (MESH_DIR, file_name)


def box_collision(name, low, high, mu=0.7):
    center = tuple((low[k] + high[k]) / 2.0 for k in range(3))
    size = tuple(high[k] - low[k] for k in range(3))
    return (name, center, box_geometry(size), surface(mu))


def inertial(mass, size, center=(0.0, 0.0, 0.0)):
    """Solid-box inertia. Static links ignore it, but every link still carries one."""
    sx, sy, sz = size
    return (
        "<inertial><pose>%s 0 0 0</pose><mass>%.6g</mass><inertia>"
        "<ixx>%.6g</ixx><ixy>0</ixy><ixz>0</ixz><iyy>%.6g</iyy><iyz>0</iyz><izz>%.6g</izz>"
        "</inertia></inertial>"
        % (
            vec(*center),
            mass,
            mass * (sy * sy + sz * sz) / 12.0,
            mass * (sx * sx + sz * sz) / 12.0,
            mass * (sx * sx + sy * sy) / 12.0,
        )
    )


def model_xml(name, pose, link, inertial_xml, visuals, collisions):
    """``visuals``: (name, geometry, material, shadows[, xyz]); ``collisions``: (name, center, geometry, surface)."""
    lines = [
        '    <model name="%s">\n' % name,
        "      <static>true</static>\n",
        "      <pose>%s</pose>\n" % vec(*pose),
        '      <link name="%s">\n' % link,
        "        %s\n" % inertial_xml,
    ]
    for collision_name, center, geometry, surface_xml in collisions:
        lines.append(
            '        <collision name="%s"><pose>%s 0 0 0</pose><geometry>%s</geometry>%s</collision>\n'
            % (collision_name, vec(*center), geometry, surface_xml)
        )
    for visual in visuals:
        visual_name, geometry, material_xml, shadows = visual[:4]
        pose = "<pose>%s 0 0 0</pose>" % vec(*visual[4]) if len(visual) > 4 else ""
        shadow = "" if shadows else "<cast_shadows>false</cast_shadows>"
        lines.append(
            '        <visual name="%s">%s%s<geometry>%s</geometry>%s</visual>\n'
            % (visual_name, pose, shadow, geometry, material_xml)
        )
    lines.append("      </link>\n    </model>\n")
    return "".join(lines)


def assert_inside(mesh, start, boxes, origin, label):
    """Every vertex added since ``start`` lies in one of the planner ``boxes``."""
    if len(mesh.vertices) == start:
        return
    points = np.asarray(mesh.vertices[start:]) + np.asarray(origin)
    covered = np.zeros(len(points), dtype=bool)
    for box in boxes:
        half = np.asarray(box.size) / 2.0 + 1e-6
        covered |= np.all(np.abs(points - np.asarray(box.center)) <= half, axis=1)
    if not covered.all():
        raise AssertionError("%s: mesh vertex %s outside its planner boxes" % (label, points[~covered][0]))


# --------------------------------------------------------------------------- buildings


def _faces(building):
    hx, hy = building.size_xy[0] / 2.0, building.size_xy[1] / 2.0
    # normal axis, outward sign, tangent axis, half extent along normal, tangent range
    return {
        "s": (1, -1.0, 0, hy, -hx, hx),
        "n": (1, 1.0, 0, hy, -hx, hx),
        "w": (0, -1.0, 1, hx, -hy + CORE, hy - CORE),
        "e": (0, 1.0, 1, hx, -hy + CORE, hy - CORE),
    }


def face_bounds(face, u0, u1, z0, z1, d0, d1):
    """Axis-aligned box on a face; ``d`` is the inset from the outer face, negative is outside."""
    n_axis, sign, t_axis, half = face[:4]
    first, second = sign * (half - d0), sign * (half - d1)
    low, high = [0.0, 0.0, z0], [0.0, 0.0, z1]
    low[n_axis], high[n_axis] = min(first, second), max(first, second)
    low[t_axis], high[t_axis] = u0, u1
    return tuple(low), tuple(high)


def face_box(mesh, face, u0, u1, z0, z1, d0, d1):
    if u1 - u0 > 1e-6 and z1 - z0 > 1e-6:
        mesh.box(*face_bounds(face, u0, u1, z0, z1, d0, d1))


def face_quad(mesh, face, u0, u1, z0, z1, d):
    n_axis, sign, t_axis, half = face[:4]

    def point(u_value, z_value):
        result = [0.0, 0.0, z_value]
        result[n_axis] = sign * (half - d)
        result[t_axis] = u_value
        return tuple(result)

    normal = [0.0, 0.0, 0.0]
    normal[n_axis] = sign
    mesh.polygon([point(u0, z0), point(u1, z0), point(u1, z1), point(u0, z1)], tuple(normal))


def window(parts, face, u0, u1, z0, z1):
    """Recessed window: glass at the back of the reveal, frame, mullion, and transom."""
    glass_d = CORE - 0.02
    face_quad(parts["glass"], face, u0, u1, z0, z1, glass_d)
    frame = parts["frame"]
    back, front = glass_d, glass_d - 0.12
    face_box(frame, face, u0, u0 + FRAME, z0, z1, front, back)
    face_box(frame, face, u1 - FRAME, u1, z0, z1, front, back)
    face_box(frame, face, u0, u1, z0, z0 + FRAME, front, back)
    face_box(frame, face, u0, u1, z1 - FRAME, z1, front, back)
    if u1 - u0 > 1.2:
        middle = (u0 + u1) / 2.0
        face_box(frame, face, middle - 0.03, middle + 0.03, z0, z1, front + 0.03, back)
    transom = z1 - min(0.55, (z1 - z0) * 0.3)
    face_box(frame, face, u0, u1, transom - 0.03, transom + 0.03, front + 0.03, back)


def punched_face(parts, building, face, z_start):
    wall = parts["wall"]
    u_min, u_max = face[4], face[5]
    length = u_max - u_min
    height = building.height
    if building.hall:
        rows = [(height - 4.4, height - 2.0)]
        bay_target, window_share, window_max = 6.0, 0.7, 4.5
    else:
        rows = []
        level = 0
        while level * FLOOR_HEIGHT + 2.6 <= height - PARAPET - 0.4:
            if level * FLOOR_HEIGHT >= z_start - 1e-6:
                rows.append((level * FLOOR_HEIGHT + 0.9, level * FLOOR_HEIGHT + 2.6))
            level += 1
        bay_target, window_share, window_max = 3.2, 0.6, 2.0
    bays = max(1, int(round(length / bay_target)))
    bay = length / bays
    width = min(window_share * bay, window_max)
    openings = [
        (u_min + (index + 0.5) * bay - width / 2.0, u_min + (index + 0.5) * bay + width / 2.0)
        for index in range(bays)
    ]
    z_value = z_start
    for sill, head in rows:
        face_box(wall, face, u_min, u_max, z_value, sill, 0.0, CORE)
        edges = [u_min] + [edge for opening in openings for edge in opening] + [u_max]
        for index in range(0, len(edges), 2):
            face_box(wall, face, edges[index], edges[index + 1], sill, head, 0.0, CORE)
        for u0, u1 in openings:
            window(parts, face, u0, u1, sill, head)
        z_value = head
    face_box(wall, face, u_min, u_max, z_value, height, 0.0, CORE)


def curtain_face(parts, building, face, z_start):
    wall, frame = parts["wall"], parts["frame"]
    u_min, u_max = face[4], face[5]
    top = building.height - PARAPET
    bottom = z_start
    if z_start == 0.0:
        bottom = 0.5
        face_box(wall, face, u_min, u_max, 0.0, bottom, 0.0, CORE)
    face_quad(parts["glass"], face, u_min, u_max, bottom, top, CORE - 0.02)
    count = max(2, int(round((u_max - u_min) / 1.5)))
    for index in range(1, count):
        u_value = u_min + index * (u_max - u_min) / count
        face_box(frame, face, u_value - 0.04, u_value + 0.04, bottom, top, 0.1, CORE - 0.02)
    level = FLOOR_HEIGHT
    while level < top - 0.5:
        if level > bottom + 0.5:
            face_box(wall, face, u_min, u_max, level - 0.15, level + 0.25, 0.04, CORE - 0.02)
        level += FLOOR_HEIGHT
    face_box(wall, face, u_min, u_max, top, building.height, 0.0, CORE)


def lobby(parts, building, face, lobby_height):
    """Recessed, glazed ground floor behind a colonnade, with double doors at the centre."""
    wall, frame = parts["wall"], parts["frame"]
    u_min, u_max = face[4], face[5]
    length = u_max - u_min
    columns = max(1, int(round(length / 4.5)))
    for index in range(columns + 1):
        u_value = u_min + index * length / columns
        face_box(wall, face, max(u_min, u_value - 0.25), min(u_max, u_value + 0.25), 0.0, lobby_height, 0.0, 0.5)
    face_box(wall, face, u_min, u_max, lobby_height - 0.6, lobby_height, 0.0, CORE)
    glass_d = LOBBY_DEPTH - 0.02
    face_quad(parts["glass"], face, u_min, u_max, 0.0, lobby_height, glass_d)
    door_width = min(4.0, building.entrance_width - 1.0)
    door_height = 2.7
    count = max(2, int(round(length / 1.5)))
    for index in range(1, count):
        u_value = u_min + index * length / count
        if abs(u_value) < door_width / 2.0 + 0.2:
            continue
        face_box(frame, face, u_value - 0.04, u_value + 0.04, 0.0, lobby_height, glass_d - 0.08, glass_d)
    face_box(frame, face, u_min, u_max, door_height, door_height + 0.1, glass_d - 0.08, glass_d)
    half = door_width / 2.0
    face_box(frame, face, -half - 0.12, -half, 0.0, door_height, glass_d - 0.16, glass_d)
    face_box(frame, face, half, half + 0.12, 0.0, door_height, glass_d - 0.16, glass_d)
    face_box(frame, face, -0.04, 0.04, 0.0, door_height, glass_d - 0.1, glass_d)
    face_quad(parts["door"], face, -half, -0.04, 0.05, door_height, glass_d - 0.03)
    face_quad(parts["door"], face, 0.04, half, 0.05, door_height, glass_d - 0.03)
    for u0 in (-0.22, 0.16):
        face_box(frame, face, u0, u0 + 0.06, 0.9, 1.4, glass_d - 0.12, glass_d - 0.03)


def building_geometry(building):
    """Meshes in the building frame (origin at the footprint centre) and local collision boxes."""
    parts = {
        "wall": Mesh(uv_scale=4.0),
        "glass": Mesh(),
        "frame": Mesh(),
        "roof": Mesh(uv_scale=4.0),
        "door": Mesh(),
    }
    hx, hy = building.size_xy[0] / 2.0, building.size_xy[1] / 2.0
    height = building.height
    faces = _faces(building)
    entrance = faces[building.entrance]
    if building.hall:
        lobby_height = 4.2
    else:
        lobby_height = 2 * FLOOR_HEIGHT if height >= 14.0 else FLOOR_HEIGHT
    core_top = height - PARAPET - ROOF_SLAB

    core_low, core_high = [-hx + CORE, -hy + CORE, 0.0], [hx - CORE, hy - CORE, core_top]
    lower_low, lower_high = list(core_low), list(core_high)
    lower_high[2] = lobby_height
    if entrance[1] < 0.0:
        lower_low[entrance[0]] = -(entrance[3] - LOBBY_DEPTH)
    else:
        lower_high[entrance[0]] = entrance[3] - LOBBY_DEPTH
    parts["wall"].box(lower_low, lower_high)
    upper_low = list(core_low)
    upper_low[2] = lobby_height
    parts["wall"].box(upper_low, core_high, bottom=True)
    parts["roof"].box((-hx + CORE, -hy + CORE, core_top), (hx - CORE, hy - CORE, core_top + ROOF_SLAB))

    for name, face in faces.items():
        z_start = lobby_height if face is entrance else 0.0
        if building.curtain_wall:
            curtain_face(parts, building, face, z_start)
        else:
            punched_face(parts, building, face, z_start)
    lobby(parts, building, entrance, lobby_height)
    if building.curtain_wall:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                xs = sorted((sx * hx, sx * (hx - 0.5)))
                ys = sorted((sy * hy, sy * (hy - 0.5)))
                parts["wall"].box((xs[0], ys[0], 0.0), (xs[1], ys[1], height))
    else:
        for offset in (-1.0, 1.0):
            x_value = offset * hx / 3.0
            parts["frame"].box(
                (x_value - 0.8, -0.5, core_top + ROOF_SLAB), (x_value + 0.8, 0.5, core_top + ROOF_SLAB + 0.45)
            )

    width = building.entrance_width
    canopy_low = campus.CANOPY_HEIGHT - 0.3
    canopy_boxes = [
        face_bounds(entrance, -width / 2.0, width / 2.0, canopy_low, campus.CANOPY_HEIGHT, -campus.CANOPY_DEPTH, 0.0),
        face_bounds(entrance, -width / 2.0, -width / 2.0 + 0.2, 0.0, canopy_low,
                    -campus.CANOPY_DEPTH, -campus.CANOPY_DEPTH + 0.2),
        face_bounds(entrance, width / 2.0 - 0.2, width / 2.0, 0.0, canopy_low,
                    -campus.CANOPY_DEPTH, -campus.CANOPY_DEPTH + 0.2),
    ]
    for low, high in canopy_boxes:
        parts["frame"].box(low, high, bottom=True)

    collisions = [box_collision("body", (-hx, -hy, 0.0), (hx, hy, height))]
    penthouse = building.penthouse_obstacle
    if penthouse is not None:
        pw, pd, _ = penthouse.size
        parts["wall"].box((-pw / 2.0, -pd / 2.0, height - PARAPET), (pw / 2.0, pd / 2.0, height + 3.0))
        parts["frame"].box((-pw / 2.0, -pd / 2.0, height + 2.8), (pw / 2.0, pd / 2.0, height + 3.0))
        collisions.append(box_collision("roof_unit", (-pw / 2.0, -pd / 2.0, height), (pw / 2.0, pd / 2.0, height + 3.0)))
    for index, (low, high) in enumerate(canopy_boxes):
        collisions.append(box_collision("canopy_%d" % index, low, high))

    boxes = [building.obstacle, building.canopy_obstacle]
    if penthouse is not None:
        boxes.append(penthouse)
    origin = (building.center_xy[0], building.center_xy[1], 0.0)
    for part, mesh in parts.items():
        assert_inside(mesh, 0, boxes, origin, "%s %s" % (building.name, part))
    return parts, collisions


def building_model(building, meshes):
    wall_rgb, roof_rgb = PALETTES[building.palette % len(PALETTES)]
    parts, collisions = building_geometry(building)
    materials = {
        "wall": material(wall_rgb, 0.85, texture="concrete.png"),
        "glass": material(GLASS[building.palette % len(GLASS)], 0.1, 0.5),
        "frame": material((0.20, 0.21, 0.22), 0.45, 0.6),
        "roof": material(roof_rgb, 0.9, texture="concrete.png"),
        "door": material((0.10, 0.12, 0.13), 0.15, 0.4),
    }
    visuals = []
    for part, mesh in parts.items():
        file_name = "%s_%s.obj" % (building.name, part)
        meshes[file_name] = mesh
        visuals.append((part, mesh_geometry(file_name), materials[part], True))
    size = building.obstacle.size
    return model_xml(
        building.name,
        (building.center_xy[0], building.center_xy[1], 0, 0, 0, 0),
        "body",
        inertial(250.0 * size[0] * size[1] * size[2], size, (0.0, 0.0, building.height / 2.0)),
        visuals,
        collisions,
    )


# --------------------------------------------------------------------------- trees

TREE_SHAPES = (
    {"fork": 2.4, "trunk": 0.2, "shell": (1.0, 0.8), "blob": 0.75, "blobs": 7},
    {"fork": 3.0, "trunk": 0.17, "shell": (0.8, 1.0), "blob": 0.7, "blobs": 6},
    {"fork": 2.0, "trunk": 0.22, "shell": (1.2, 0.7), "blob": 0.85, "blobs": 8},
)
LEAF_COLORS = ((0.92, 1.0, 0.80), (1.0, 0.96, 0.72), (0.80, 0.94, 0.78))


def tree_variant(index):
    shape = TREE_SHAPES[index]
    crown_z = campus.TREE_VARIANTS[index]["canopy_z"]
    shell_h, shell_v = shape["shell"]
    trunk = shape["trunk"]
    rng = seeded(100 + index)
    bark = Mesh()
    bark.cylinder((0.0, 0.0, 0.0), (0.0, 0.0, shape["fork"]), trunk, trunk * 0.75, 9)
    bark.cylinder((0.0, 0.0, shape["fork"] - 0.1), (0.0, 0.0, crown_z + 0.3), trunk * 0.6, trunk * 0.25, 6)
    for branch in range(4):
        angle = 2.0 * math.pi * branch / 4.0 + 0.4 * index
        bark.cylinder(
            (0.0, 0.0, shape["fork"] - 0.3),
            (0.55 * shell_h * math.cos(angle), 0.55 * shell_h * math.sin(angle), crown_z - 0.1),
            trunk * 0.5,
            trunk * 0.2,
            6,
        )
    leaves = Mesh(uv_scale=1.5)
    radius = shape["blob"]
    for blob in range(shape["blobs"]):
        angle = 2.0 * math.pi * blob / shape["blobs"] + rng.random() * 0.3
        lift = 0.25 * shell_v * (1.0 if blob % 2 else -1.0)
        leaves.blob(
            (shell_h * math.cos(angle), shell_h * math.sin(angle), crown_z + lift), radius, rng
        )
    leaves.blob((0.0, 0.0, crown_z + shell_v), radius, rng)
    leaves.blob((0.0, 0.0, crown_z), radius * 1.15, rng)
    limits = campus.TREE_VARIANTS[index]
    for mesh in (bark, leaves):
        low, high = mesh.bounds()
        reach = max(abs(low[0]), abs(low[1]), high[0], high[1])
        assert reach <= limits["radius"] and high[2] <= limits["top"], ("tree variant", index, reach, high)
    assert limits["canopy_z"] + limits["canopy_radius"] <= limits["top"]
    return bark, leaves


def trees_model(meshes):
    variants = [tree_variant(index) for index in range(len(campus.TREE_VARIANTS))]
    bark = Mesh()
    foliage = [Mesh(uv_scale=1.5) for _ in variants]
    collisions = []
    for tree in campus.TREES:
        variant_bark, variant_leaves = variants[tree.variant]
        offset = (tree.position_xy[0], tree.position_xy[1], 0.0)
        start = len(bark.vertices)
        bark.extend(variant_bark, tree.scale, tree.yaw, offset)
        assert_inside(bark, start, [tree.obstacle], (0, 0, 0), tree.name)
        start = len(foliage[tree.variant].vertices)
        foliage[tree.variant].extend(variant_leaves, tree.scale, tree.yaw, offset)
        assert_inside(foliage[tree.variant], start, [tree.obstacle], (0, 0, 0), tree.name)
        trunk_radius = TREE_SHAPES[tree.variant]["trunk"] * tree.scale
        collisions.append(
            (
                tree.name + "_trunk",
                (offset[0], offset[1], tree.canopy_center_z / 2.0),
                cylinder_geometry(trunk_radius, tree.canopy_center_z),
                surface(0.6, 0.05),
            )
        )
        collisions.append(
            (
                tree.name + "_canopy",
                (offset[0], offset[1], tree.canopy_center_z),
                sphere_geometry(tree.canopy_radius),
                surface(0.4, 0.05),
            )
        )
    meshes["trees_bark.obj"] = bark
    visuals = [("bark", mesh_geometry("trees_bark.obj"), material((0.33, 0.23, 0.15), 0.95), True)]
    for index, mesh in enumerate(foliage):
        file_name = "trees_foliage_%d.obj" % index
        meshes[file_name] = mesh
        visuals.append(
            ("foliage_%d" % index, mesh_geometry(file_name), material(LEAF_COLORS[index], 0.95, texture="leaves.png"), True)
        )
    return model_xml(
        "street_trees", (0, 0, 0, 0, 0, 0), "trees",
        inertial(400.0 * len(campus.TREES), (2.0, 2.0, 6.0), (0.0, 0.0, 3.0)), visuals, collisions,
    )


# --------------------------------------------------------------------------- street furniture


def furniture_model(meshes):
    metal = Mesh()
    lens = Mesh()
    panels = [Mesh() for _ in SIGN_COLORS]
    collisions = []
    top = campus.LIGHT_HEIGHT
    for light in campus.LIGHTS:
        x_value, y_value = light.position_xy
        fx, fy = light.facing
        start = len(metal.vertices)
        metal.box((x_value - 0.18, y_value - 0.18, 0.0), (x_value + 0.18, y_value + 0.18, 0.6))
        metal.cylinder((x_value, y_value, 0.6), (x_value, y_value, top - 0.1), 0.09, 0.06, 8)
        end_x, end_y = x_value + fx * campus.LIGHT_ARM, y_value + fy * campus.LIGHT_ARM
        metal.box(
            (min(x_value, end_x) - 0.04, min(y_value, end_y) - 0.04, top - 0.18),
            (max(x_value, end_x) + 0.04, max(y_value, end_y) + 0.04, top - 0.1),
        )
        metal.cylinder(
            (x_value, y_value, top - 0.8), (x_value + fx * 0.8, y_value + fy * 0.8, top - 0.14), 0.035, 0.035, 6
        )
        head_a = (x_value + fx * (campus.LIGHT_ARM - 0.55), y_value + fy * (campus.LIGHT_ARM - 0.55))
        head_b = (x_value + fx * (campus.LIGHT_ARM + 0.15), y_value + fy * (campus.LIGHT_ARM + 0.15))
        head_low = (min(head_a[0], head_b[0]) - (0.16 if fx == 0 else 0.0),
                    min(head_a[1], head_b[1]) - (0.16 if fy == 0 else 0.0))
        head_high = (max(head_a[0], head_b[0]) + (0.16 if fx == 0 else 0.0),
                     max(head_a[1], head_b[1]) + (0.16 if fy == 0 else 0.0))
        metal.box((head_low[0], head_low[1], top - 0.28), (head_high[0], head_high[1], top - 0.1), bottom=False)
        lens.polygon(
            [
                (head_low[0] + 0.03, head_low[1] + 0.03, top - 0.285),
                (head_high[0] - 0.03, head_low[1] + 0.03, top - 0.285),
                (head_high[0] - 0.03, head_high[1] - 0.03, top - 0.285),
                (head_low[0] + 0.03, head_high[1] - 0.03, top - 0.285),
            ],
            (0, 0, -1),
        )
        assert_inside(metal, start, [light.obstacle], (0, 0, 0), light.name)
        collisions.append(
            (light.name + "_pole", (x_value, y_value, top / 2.0), cylinder_geometry(0.18, top), surface(0.5))
        )
        collisions.append(
            box_collision(
                light.name + "_head",
                (min(x_value, head_low[0]), min(y_value, head_low[1]), top - 0.28),
                (max(x_value, head_high[0]), max(y_value, head_high[1]), top - 0.1),
                0.5,
            )
        )
    for sign in campus.SIGNS:
        x_value, y_value = sign.position_xy
        start = len(metal.vertices)
        metal.cylinder((x_value, y_value, 0.0), (x_value, y_value, campus.SIGN_HEIGHT), 0.045, 0.04, 8)
        panel = panels[sign.color]
        panel_start = len(panel.vertices)
        panel.box((x_value - 0.45, y_value + 0.05, 2.15), (x_value + 0.45, y_value + 0.09, 2.8), bottom=True)
        panels[(sign.color + 1) % len(panels)].box(
            (x_value - 0.35, y_value + 0.05, 1.6), (x_value + 0.35, y_value + 0.09, 2.0), bottom=True
        )
        assert_inside(metal, start, [sign.obstacle], (0, 0, 0), sign.name)
        assert_inside(panel, panel_start, [sign.obstacle], (0, 0, 0), sign.name)
        collisions.append(
            (sign.name + "_pole", (x_value, y_value, campus.SIGN_HEIGHT / 2.0),
             cylinder_geometry(0.05, campus.SIGN_HEIGHT), surface(0.5))
        )
        collisions.append(
            box_collision(sign.name + "_panel", (x_value - 0.45, y_value + 0.05, 1.6), (x_value + 0.45, y_value + 0.09, 2.8))
        )
    meshes["street_metal.obj"] = metal
    meshes["street_lamp_lens.obj"] = lens
    visuals = [
        ("metal", mesh_geometry("street_metal.obj"), material((0.26, 0.28, 0.30), 0.45, 0.7), True),
        ("lamp_lens", mesh_geometry("street_lamp_lens.obj"),
         material((0.92, 0.92, 0.86), 0.3, emissive=(0.35, 0.33, 0.28)), False),
    ]
    for index, mesh in enumerate(panels):
        file_name = "street_sign_panel_%d.obj" % index
        meshes[file_name] = mesh
        visuals.append(("sign_panel_%d" % index, mesh_geometry(file_name), material(SIGN_COLORS[index], 0.5), True))
    return model_xml(
        "street_furniture", (0, 0, 0, 0, 0, 0), "furniture",
        inertial(150.0 * (len(campus.LIGHTS) + len(campus.SIGNS)), (0.4, 0.4, 8.0), (0.0, 0.0, 4.0)),
        visuals, collisions,
    )


def hedges_model(meshes):
    mesh = Mesh(uv_scale=1.0)
    collisions = []
    for hedge in campus.HEDGES:
        x0, y0, x1, y1 = hedge.rect
        start = len(mesh.vertices)
        mesh.box((x0, y0, 0.0), (x1, y1, campus.HEDGE_HEIGHT - 0.15))
        mesh.box((x0 + 0.06, y0 + 0.06, campus.HEDGE_HEIGHT - 0.15), (x1 - 0.06, y1 - 0.06, campus.HEDGE_HEIGHT))
        assert_inside(mesh, start, [hedge.obstacle], (0, 0, 0), hedge.name)
        collisions.append(box_collision(hedge.name, (x0, y0, 0.0), (x1, y1, campus.HEDGE_HEIGHT), 0.5))
    meshes["hedges.obj"] = mesh
    visuals = [("shrubs", mesh_geometry("hedges.obj"), material((0.70, 0.84, 0.62), 0.95, texture="leaves.png"), True)]
    return model_xml(
        "hedges", (0, 0, 0, 0, 0, 0), "hedges",
        inertial(60.0 * len(campus.HEDGES), (10.0, 0.6, 0.9), (0.0, 0.0, 0.45)), visuals, collisions,
    )


# --------------------------------------------------------------------------- cars


def car_meshes():
    """A compact hatchback in its own frame: +x forward, wheels on the ground."""
    half_width = campus.CAR_WIDTH / 2.0
    body = Mesh()
    body.prism_y(
        [(-2.2, 0.3), (2.2, 0.3), (2.2, 0.72), (1.95, 0.9), (-2.15, 0.92), (-2.2, 0.8)], -half_width, half_width
    )
    body.box((-1.1, -0.74, 1.44), (0.2, 0.74, 1.5))
    for x_value in (-0.5, 0.35):
        for side in (-1.0, 1.0):
            ys = sorted((side * 0.77, side * 0.8))
            body.box((x_value, ys[0], 0.9), (x_value + 0.1, ys[1], 1.45))
    glass = Mesh()
    glass.prism_y([(-1.85, 0.9), (0.95, 0.9), (0.25, 1.45), (-1.15, 1.45)], -0.78, 0.78)
    trim = Mesh()
    for x_value in (-1.4, 1.4):
        for side in (-1.0, 1.0):
            inner, outer = side * 0.69, side * 0.91
            trim.cylinder((x_value, inner, 0.33), (x_value, outer, 0.33), 0.33, 0.33, 12)
    trim.box((-2.22, -0.85, 0.3), (-2.18, 0.85, 0.5))
    trim.box((2.18, -0.85, 0.3), (2.22, 0.85, 0.5))
    lamps = Mesh()
    for side in (-1.0, 1.0):
        ys = sorted((side * 0.55, side * 0.8))
        lamps.polygon([(2.205, ys[0], 0.58), (2.205, ys[1], 0.58), (2.205, ys[1], 0.68), (2.205, ys[0], 0.68)], (1, 0, 0))
        lamps.polygon(
            [(-2.205, ys[0], 0.66), (-2.205, ys[1], 0.66), (-2.205, ys[1], 0.76), (-2.205, ys[0], 0.76)], (-1, 0, 0)
        )
    return {"body": body, "glass": glass, "trim": trim, "lamps": lamps}


def car_model(car):
    parts = car_meshes()
    box = car.obstacle
    for name, mesh in parts.items():
        rotated = Mesh()
        rotated.extend(mesh, 1.0, car.yaw, (car.position_xy[0], car.position_xy[1], 0.0))
        assert_inside(rotated, 0, [box], (0, 0, 0), car.name + " " + name)
    half_width = campus.CAR_WIDTH / 2.0
    collisions = [
        box_collision("chassis", (-2.2, -half_width, 0.3), (2.2, half_width, 0.92)),
        box_collision("cabin", (-1.85, -0.8, 0.92), (0.95, 0.8, 1.5)),
        box_collision("wheels", (-1.75, -0.91, 0.0), (1.75, 0.91, 0.3), 0.9),
    ]
    visuals = [
        ("body", mesh_geometry("car_body.obj"), material(CAR_COLORS[car.color], 0.3, 0.6), True),
        ("glass", mesh_geometry("car_glass.obj"), material((0.08, 0.10, 0.12), 0.1, 0.4), True),
        ("trim", mesh_geometry("car_trim.obj"), material((0.05, 0.05, 0.05), 0.8), True),
        ("lamps", mesh_geometry("car_lamps.obj"), material((0.80, 0.25, 0.15), 0.3), False),
    ]
    return model_xml(
        car.name,
        (car.position_xy[0], car.position_xy[1], 0, 0, 0, car.yaw),
        "body",
        inertial(1300.0, (campus.CAR_LENGTH, campus.CAR_WIDTH, campus.CAR_HEIGHT), (0.0, 0.0, 0.7)),
        visuals,
        collisions,
    )


# --------------------------------------------------------------------------- fence, gate, goals


def fence_run(meshes, box):
    iron, concrete, brick = meshes
    low = (box.center[0] - box.size[0] / 2.0, box.center[1] - box.size[1] / 2.0)
    high = (box.center[0] + box.size[0] / 2.0, box.center[1] + box.size[1] / 2.0)
    along = 0 if box.size[0] >= box.size[1] else 1
    across = 1 - along
    middle = (low[across] + high[across]) / 2.0
    start, end = low[along], high[along]

    def strip(a0, a1, half, z0, z1, mesh):
        lo, hi = [0.0, 0.0, z0], [0.0, 0.0, z1]
        lo[along], hi[along] = a0, a1
        lo[across], hi[across] = middle - half, middle + half
        mesh.box(tuple(lo), tuple(hi))

    strip(start, end, 0.18, 0.0, 0.45, concrete)
    strip(start, end, 0.025, 0.85, 0.9, iron)
    strip(start, end, 0.025, 2.2, 2.26, iron)
    position = start + 0.09
    while position < end - 0.05:
        strip(position - 0.011, position + 0.011, 0.011, 0.45, 2.34, iron)
        position += 0.18
    position = start
    while position <= end + 1e-6:
        if round((position - start) / 2.5) % 5 == 0:
            strip(max(start, position - 0.23), min(end, position + 0.23), 0.23, 0.0, 2.25, brick)
            strip(max(start, position - 0.25), min(end, position + 0.25), 0.25, 2.25, campus.FENCE_HEIGHT, concrete)
        else:
            strip(max(start, position - 0.04), min(end, position + 0.04), 0.04, 0.45, 2.3, iron)
        position += 2.5


def enclosure_models(meshes):
    iron, concrete, brick = Mesh(), Mesh(uv_scale=2.0), Mesh()
    stone, dark, glass, sign, red, white = Mesh(uv_scale=2.0), Mesh(), Mesh(), Mesh(), Mesh(), Mesh()
    goal = Mesh()
    fence_collisions, gate_collisions, goal_collisions = [], [], []
    for item in campus.STRUCTURES:
        box = item.box
        low = tuple(box.center[k] - box.size[k] / 2.0 for k in range(3))
        high = tuple(box.center[k] + box.size[k] / 2.0 for k in range(3))
        starts = {id(mesh): len(mesh.vertices) for mesh in (iron, concrete, brick, stone, dark, glass, sign, red, white, goal)}
        target = gate_collisions
        if item.kind == "fence":
            fence_run((iron, concrete, brick), box)
            target = fence_collisions
        elif item.kind == "pillar":
            cx, cy = box.center[0], box.center[1]
            stone.box((cx - 0.5, cy - 0.5, 0.0), (cx + 0.5, cy + 0.5, high[2] - 0.3))
            stone.box((cx - 0.6, cy - 0.6, high[2] - 0.3), (cx + 0.6, cy + 0.6, high[2]), bottom=True)
        elif item.kind == "header":
            cy = box.center[1]
            dark.box((low[0], cy - 0.45, low[2]), (high[0], cy + 0.45, high[2]), bottom=True)
            for side in (-1.0, 1.0):
                y_value = cy + side * 0.47
                sign.polygon(
                    [(campus.GATE_X - 4.0, y_value, low[2] + 0.2), (campus.GATE_X + 4.0, y_value, low[2] + 0.2),
                     (campus.GATE_X + 4.0, y_value, high[2] - 0.2), (campus.GATE_X - 4.0, y_value, high[2] - 0.2)],
                    (0.0, side, 0.0),
                )
        elif item.kind == "booth":
            stone.box((low[0] + 0.5, low[1] + 0.5, 0.0), (high[0] - 0.5, high[1] - 0.5, high[2] - 0.3))
            dark.box((low[0], low[1], high[2] - 0.3), high, bottom=True)
            x0, y0, x1, y1 = low[0] + 0.49, low[1] + 0.49, high[0] - 0.49, high[1] - 0.49
            for points, normal in (
                ([(x0, y0, 1.0), (x1, y0, 1.0), (x1, y0, 2.3), (x0, y0, 2.3)], (0, -1, 0)),
                ([(x0, y1, 1.0), (x1, y1, 1.0), (x1, y1, 2.3), (x0, y1, 2.3)], (0, 1, 0)),
                ([(x0, y0, 1.0), (x0, y1, 1.0), (x0, y1, 2.3), (x0, y0, 2.3)], (-1, 0, 0)),
                ([(x1, y0, 1.0), (x1, y1, 1.0), (x1, y1, 2.3), (x1, y0, 2.3)], (1, 0, 0)),
            ):
                glass.polygon(points, normal)
            dark.polygon([(x0 - 0.005, y0 + 0.4, 0.0), (x0 - 0.005, y0 + 1.3, 0.0),
                          (x0 - 0.005, y0 + 1.3, 2.2), (x0 - 0.005, y0 + 0.4, 2.2)], (-1, 0, 0))
        elif item.kind == "barrier":
            cy = box.center[1]
            white.box((high[0] - 0.4, cy - 0.2, 0.0), (high[0], cy + 0.2, 1.0))
            position, stripe = high[0] - 0.4, 0
            while position > low[0] + 1e-6:
                left = max(low[0], position - 0.5)
                (red if stripe % 2 == 0 else white).box((left, cy - 0.04, 0.92), (position, cy + 0.04, 1.04), bottom=True)
                position, stripe = left, stripe + 1
        elif item.kind == "leaf":
            cy = box.center[1]
            iron.box((low[0], cy - 0.05, 0.1), (high[0], cy + 0.05, 0.2))
            iron.box((low[0], cy - 0.05, high[2] - 0.12), (high[0], cy + 0.05, high[2]))
            position = low[0]
            while position <= high[0] + 1e-6:
                iron.box((max(low[0], position - 0.015), cy - 0.015, 0.2), (min(high[0], position + 0.015), cy + 0.015, high[2] - 0.12))
                position += 0.12
            position = low[0] + 0.5
            while position < high[0]:
                dark.box((position - 0.1, cy - 0.15, 0.0), (position + 0.1, cy + 0.15, 0.1))
                position += 3.0
        elif item.kind == "goal":
            target = goal_collisions
            faces_east = item.name == "goal_w"
            post_x = (high[0] - 0.12, high[0]) if faces_east else (low[0], low[0] + 0.12)
            back_x = low[0] + 0.05 if faces_east else high[0] - 0.05
            for y_value in (low[1] + 0.06, high[1] - 0.06):
                goal.box((post_x[0], y_value - 0.06, 0.0), (post_x[1], y_value + 0.06, high[2]))
                goal.cylinder(
                    ((post_x[0] + post_x[1]) / 2.0, y_value, high[2] - 0.1), (back_x, y_value, 0.05), 0.03, 0.03, 6
                )
            goal.box((post_x[0], low[1], high[2] - 0.12), (post_x[1], high[1], high[2]), bottom=True)
            goal.box((low[0], low[1], 0.0), (high[0], low[1] + 0.04, 0.04))
            goal.box((low[0], high[1] - 0.04, 0.0), (high[0], high[1], 0.04))
            back_rail = (low[0], low[0] + 0.04) if faces_east else (high[0] - 0.04, high[0])
            goal.box((back_rail[0], low[1], 0.0), (back_rail[1], high[1], 0.04))
        for mesh in (iron, concrete, brick, stone, dark, glass, sign, red, white, goal):
            assert_inside(mesh, starts[id(mesh)], [box], (0, 0, 0), item.name)
        target.append(box_collision(item.name, low, high, 0.6))
    for name, mesh in (
        ("fence_iron.obj", iron), ("fence_concrete.obj", concrete), ("fence_brick.obj", brick),
        ("gate_stone.obj", stone), ("gate_dark.obj", dark), ("gate_glass.obj", glass), ("gate_sign.obj", sign),
        ("gate_barrier_red.obj", red), ("gate_barrier_white.obj", white), ("sports_goals.obj", goal),
    ):
        meshes[name] = mesh
    iron_mat = material((0.12, 0.13, 0.14), 0.5, 0.7)
    fence = model_xml(
        "perimeter_fence", (0, 0, 0, 0, 0, 0), "fence",
        inertial(2.0e5, (230.0, 190.0, 2.4), (20.0, 0.0, 1.2)),
        [
            ("iron", mesh_geometry("fence_iron.obj"), iron_mat, True),
            ("base", mesh_geometry("fence_concrete.obj"), material((0.66, 0.65, 0.62), 0.9, texture="concrete.png"), True),
            ("piers", mesh_geometry("fence_brick.obj"), material((0.55, 0.28, 0.20), 0.9), True),
        ],
        fence_collisions,
    )
    gate = model_xml(
        "main_gate", (0, 0, 0, 0, 0, 0), "gate",
        inertial(4.0e4, (16.0, 5.0, 6.4), (campus.GATE_X, campus.FENCE_MIN[1], 3.2)),
        [
            ("stone", mesh_geometry("gate_stone.obj"), material((0.78, 0.76, 0.72), 0.85, texture="concrete.png"), True),
            ("dark", mesh_geometry("gate_dark.obj"), material((0.22, 0.23, 0.25), 0.6, 0.3), True),
            ("glass", mesh_geometry("gate_glass.obj"), material(GLASS[0], 0.1, 0.5), True),
            ("sign", mesh_geometry("gate_sign.obj"), material((0.10, 0.28, 0.58), 0.4), False),
            ("barrier_red", mesh_geometry("gate_barrier_red.obj"), material((0.80, 0.10, 0.08), 0.5), True),
            ("barrier_white", mesh_geometry("gate_barrier_white.obj"), material((0.92, 0.92, 0.90), 0.5), True),
        ],
        gate_collisions,
    )
    goals = model_xml(
        "sports_goals", (0, 0, 0, 0, 0, 0), "goals",
        inertial(120.0, (1.0, 5.2, 2.1), (20.0, 80.0, 1.05)),
        [("frames", mesh_geometry("sports_goals.obj"), material((0.95, 0.95, 0.95), 0.4), True)],
        goal_collisions,
    )
    return fence + gate + goals


# --------------------------------------------------------------------------- ground


def terrain_meshes():
    """Lawn with gentle undulation (0 to 0.3 m), flat under and around paving.

    The collision mesh keeps only raised triangles; the ground plane covers the rest.
    """
    xs = np.arange(TERRAIN_MIN[0], TERRAIN_MAX[0] + 1e-6, TERRAIN_STEP)
    ys = np.arange(TERRAIN_MIN[1], TERRAIN_MAX[1] + 1e-6, TERRAIN_STEP)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance = np.full(grid_x.shape, np.inf)
    for x0, y0, x1, y1 in campus.flat_rects():
        dx = np.maximum(np.maximum(x0 - grid_x, 0.0), grid_x - x1)
        dy = np.maximum(np.maximum(y0 - grid_y, 0.0), grid_y - y1)
        distance = np.minimum(distance, np.hypot(dx, dy))
    mask = np.clip((distance - 2.0) / 5.0, 0.0, 1.0)
    edge = np.minimum(np.minimum(grid_x - xs[0], xs[-1] - grid_x), np.minimum(grid_y - ys[0], ys[-1] - grid_y))
    edge = np.clip(edge / 10.0, 0.0, 1.0)
    wave = (
        0.5
        + 0.22 * np.sin(0.11 * grid_x + 0.7) * np.sin(0.13 * grid_y + 1.3)
        + 0.16 * np.sin(0.31 * grid_x - 0.23 * grid_y)
        + 0.12 * np.sin(0.05 * grid_x + 0.07 * grid_y + 2.0)
    )
    height = np.round(TERRAIN_AMPLITUDE * mask * edge * np.clip(wave, 0.0, 1.0), 3)
    slope_y, slope_x = np.gradient(height, TERRAIN_STEP)
    normals = np.stack([-slope_x, -slope_y, np.ones_like(height)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    vertices = [tuple(map(float, point)) for point in zip(grid_x.ravel(), grid_y.ravel(), height.ravel())]
    normal_list = [tuple(map(float, normal)) for normal in normals.reshape(-1, 3)]
    columns = len(xs)
    flat_height = height.ravel()
    triangles, raised = [], []
    for row in range(len(ys) - 1):
        for column in range(columns - 1):
            a = row * columns + column
            for triangle in ((a, a + 1, a + columns + 1), (a, a + columns + 1, a + columns)):
                triangles.append(triangle)
                if max(flat_height[index] for index in triangle) > 0.0:
                    raised.append(triangle)
    visual = Mesh(uv_scale=6.0)
    visual.raw(vertices, normal_list, [(x / 6.0, y / 6.0) for x, y, _ in vertices], triangles)
    collision = Mesh()
    collision.raw(vertices, normal_list, None, raised)
    return visual, collision


def markings_mesh():
    paint = Mesh()
    half = campus.ROAD_HALF_WIDTH
    ns_lines = campus.STREET_X + campus.RING_X
    ew_lines = campus.STREET_Y + campus.RING_Y
    z_ns, z_ew = ROAD_Z["north_south"] + PAINT, ROAD_Z["east_west"] + PAINT

    def dashes_along_y(x_value, y_from, y_to, crossings):
        y_value = y_from
        while y_value + 3.0 <= y_to:
            if all(abs(y_value + 1.5 - crossing) > half + 5.0 for crossing in crossings):
                paint.quad_up(x_value - 0.08, y_value, x_value + 0.08, y_value + 3.0, z_ns)
            y_value += 6.0

    def dashes_along_x(y_value, x_from, x_to, crossings):
        x_value = x_from
        while x_value + 3.0 <= x_to:
            if all(abs(x_value + 1.5 - crossing) > half + 5.0 for crossing in crossings):
                paint.quad_up(x_value, y_value - 0.08, x_value + 3.0, y_value + 0.08, z_ew)
            x_value += 6.0

    def zebra_on_ns(x_value, y_from, y_to):
        stripe = x_value - half + 0.5
        while stripe + 0.5 <= x_value + half - 0.5 + 1e-6:
            paint.quad_up(stripe, y_from, stripe + 0.5, y_to, z_ns)
            stripe += 1.0

    def zebra_on_ew(y_value, x_from, x_to):
        stripe = y_value - half + 0.5
        while stripe + 0.5 <= y_value + half - 0.5 + 1e-6:
            paint.quad_up(x_from, stripe, x_to, stripe + 0.5, z_ew)
            stripe += 1.0

    ring_low_y, ring_high_y = campus.RING_Y[0] - half, campus.RING_Y[1] + half
    ring_low_x, ring_high_x = campus.RING_X[0] - half, campus.RING_X[1] + half
    for x_value in ns_lines:
        dashes_along_y(x_value, ring_low_y, ring_high_y, ew_lines)
    for y_value in ew_lines:
        crossings = list(ns_lines)
        dashes_along_x(y_value, ring_low_x, ring_high_x, crossings)
    dashes_along_y(campus.GATE_X, campus.PUBLIC_ROAD_Y + half, ring_low_y,
                   [campus.PUBLIC_ROAD_Y, campus.RING_Y[0], campus.FENCE_MIN[1]])
    dashes_along_x(campus.PUBLIC_ROAD_Y, -160.0, 200.0, [campus.GATE_X])
    for x_value in ns_lines:
        for y_value in ew_lines:
            if y_value < campus.RING_Y[1]:
                zebra_on_ns(x_value, y_value + half + 1.0, y_value + half + 4.0)
            if y_value > campus.RING_Y[0]:
                zebra_on_ns(x_value, y_value - half - 4.0, y_value - half - 1.0)
            if x_value < campus.RING_X[1]:
                zebra_on_ew(y_value, x_value + half + 1.0, x_value + half + 4.0)
            if x_value > campus.RING_X[0]:
                zebra_on_ew(y_value, x_value - half - 4.0, x_value - half - 1.0)
    zebra_on_ns(campus.GATE_X, ring_low_y - 4.0, ring_low_y - 1.0)

    lot_x0, lot_y0, lot_x1, lot_y1 = campus.PARKING_LOT
    lot_z = LOT_Z + PAINT
    x_value = lot_x0 + campus.PARKING_BAY_PITCH
    while x_value < lot_x1 - 2.0 + campus.PARKING_BAY_PITCH:
        paint.quad_up(x_value - 0.06, lot_y1 - 5.6, x_value + 0.06, lot_y1 - 0.4, lot_z)
        paint.quad_up(x_value - 0.06, lot_y0 + 0.4, x_value + 0.06, lot_y0 + 5.6, lot_z)
        x_value += campus.PARKING_BAY_PITCH
    paint.quad_up(lot_x0 + 1.0, (lot_y0 + lot_y1) / 2.0 - 0.06, lot_x1 - 1.0, (lot_y0 + lot_y1) / 2.0 + 0.06, lot_z)

    fx0, fy0, fx1, fy1 = campus.SPORTS_FIELD
    px0, py0, px1, py1 = fx0 + 2.0, fy0 + 2.0, fx1 - 2.0, fy1 - 2.0
    field_z = FIELD_Z + PAINT
    line = 0.06
    for x0, y0, x1, y1 in (
        (px0, py0 - line, px1, py0 + line), (px0, py1 - line, px1, py1 + line),
        (px0 - line, py0, px0 + line, py1), (px1 - line, py0, px1 + line, py1),
    ):
        paint.quad_up(x0, y0, x1, y1, field_z)
    middle_x, middle_y = (px0 + px1) / 2.0, (py0 + py1) / 2.0
    paint.quad_up(middle_x - line, py0, middle_x + line, py1, field_z)
    for x0, x1 in ((px0, px0 + 6.0), (px1 - 6.0, px1)):
        inner = x1 if x0 == px0 else x0
        paint.quad_up(x0, middle_y - 6.0 - line, x1, middle_y - 6.0 + line, field_z)
        paint.quad_up(x0, middle_y + 6.0 - line, x1, middle_y + 6.0 + line, field_z)
        paint.quad_up(inner - line, middle_y - 6.0, inner + line, middle_y + 6.0, field_z)
    segments = 40
    for index in range(segments):
        first = 2.0 * math.pi * index / segments
        second = 2.0 * math.pi * (index + 1) / segments
        paint.polygon(
            [
                (middle_x + 4.94 * math.cos(first), middle_y + 4.94 * math.sin(first), field_z),
                (middle_x + 5.06 * math.cos(first), middle_y + 5.06 * math.sin(first), field_z),
                (middle_x + 5.06 * math.cos(second), middle_y + 5.06 * math.sin(second), field_z),
                (middle_x + 4.94 * math.cos(second), middle_y + 4.94 * math.sin(second), field_z),
            ],
            (0, 0, 1),
        )
    return paint


def ground_model(meshes):
    terrain, terrain_collision = terrain_meshes()
    meshes["terrain.obj"] = terrain
    meshes["terrain_collision.obj"] = terrain_collision
    roads = Mesh(uv_scale=6.0)
    pavement = Mesh(uv_scale=2.0)
    lawn = Mesh(uv_scale=6.0)
    turf = Mesh(uv_scale=10.0)
    collisions = [
        ("plane", (0.0, 0.0, 0.0), "<plane><normal>0 0 1</normal><size>1 1</size></plane>", surface(0.85)),
        ("terrain_relief", (0.0, 0.0, 0.0), mesh_geometry("terrain_collision.obj"), surface(0.85)),
    ]
    for kind, rects in campus.road_rects().items():
        z_value = ROAD_Z[kind]
        for index, (x0, y0, x1, y1) in enumerate(rects):
            roads.quad_up(x0, y0, x1, y1, z_value)
            collisions.append(box_collision("road_%s_%d" % (kind, index), (x0, y0, z_value - 0.1), (x1, y1, z_value), 0.9))
    lot_x0, lot_y0, lot_x1, lot_y1 = campus.PARKING_LOT
    roads.quad_up(lot_x0, lot_y0, lot_x1, lot_y1, LOT_Z)
    collisions.append(box_collision("parking_lot", (lot_x0, lot_y0, LOT_Z - 0.1), (lot_x1, lot_y1, LOT_Z), 0.9))
    for index, (x0, y0, x1, y1) in enumerate(campus.pavement_rects()):
        pavement.box((x0, y0, 0.0), (x1, y1, PAVEMENT_Z))
        collisions.append(box_collision("pavement_%d" % index, (x0, y0, 0.0), (x1, y1, PAVEMENT_Z), 0.8))
    for corner_x in (-9.0, 9.0):
        for corner_y in (-9.0, 9.0):
            lawn.quad_up(corner_x - 3.5, corner_y - 3.5, corner_x + 3.5, corner_y + 3.5, PAVEMENT_Z + 0.005)
    fx0, fy0, fx1, fy1 = campus.SPORTS_FIELD
    turf.quad_up(fx0, fy0, fx1, fy1, FIELD_Z)
    collisions.append(box_collision("sports_field", (fx0, fy0, FIELD_Z - 0.1), (fx1, fy1, FIELD_Z), 0.9))
    collisions.append(("landing_pad", (0.0, 0.0, (PAVEMENT_Z + 0.01) / 2.0), cylinder_geometry(3.0, PAVEMENT_Z + 0.01), surface(0.9)))
    for name, mesh in (("roads.obj", roads), ("pavement.obj", pavement), ("plaza_lawn.obj", lawn),
                       ("sports_turf.obj", turf), ("markings.obj", markings_mesh())):
        meshes[name] = mesh
    white = (1.0, 1.0, 1.0)
    pad_top = PAVEMENT_Z + 0.01
    visuals = [
        ("far_grass", "<plane><normal>0 0 1</normal><size>3000 3000</size></plane>",
         material((0.30, 0.42, 0.19), 0.97), False, (0.0, 0.0, -0.03)),
        ("terrain", mesh_geometry("terrain.obj"), material(white, 0.97, texture="grass.png"), False),
        ("roads", mesh_geometry("roads.obj"), material(white, 0.92, texture="asphalt.png"), False),
        ("pavement", mesh_geometry("pavement.obj"), material(white, 0.88, texture="paving.png"), False),
        ("plaza_lawn", mesh_geometry("plaza_lawn.obj"), material(white, 0.97, texture="grass.png"), False),
        ("sports_turf", mesh_geometry("sports_turf.obj"), material(white, 0.9, texture="turf.png"), False),
        ("markings", mesh_geometry("markings.obj"), material((0.88, 0.88, 0.84), 0.6), False),
        ("pad", "<cylinder><radius>3</radius><length>0.01</length></cylinder>",
         material((0.52, 0.53, 0.54), 0.8), False, (0.0, 0.0, pad_top - 0.005)),
        ("pad_marking", "<cylinder><radius>1.4</radius><length>0.01</length></cylinder>",
         material((0.15, 0.35, 0.75), 0.5), False, (0.0, 0.0, pad_top + 0.001)),
    ]
    return model_xml(
        "ground", (0, 0, 0, 0, 0, 0), "link",
        inertial(1.0e6, (400.0, 320.0, 1.0), (20.0, 0.0, -0.5)), visuals, collisions,
    )


# --------------------------------------------------------------------------- world


WORLD_HEADER = """<?xml version="1.0"?>
<!-- Generated by tools/generate_campus.py from park_inspection/campus.py. Edit the layout there. -->
<sdf version="1.9">
  <world name="park">
    <!-- PX4 loads the physics system itself. 250 Hz matches its SITL worlds. -->
    <physics name="park_physics" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <!-- ISA sea-level air, consistent with the field elevation below. -->
    <atmosphere type="adiabatic">
      <temperature>288.15</temperature>
      <pressure>101325</pressure>
      <temperature_gradient>-0.0065</temperature_gradient>
    </atmosphere>
    <!-- Light breeze, about 2 m/s. PX4's motor model subtracts this from the body airspeed. -->
    <wind>
      <linear_velocity>1.8 0.7 0</linear_velocity>
    </wind>
    <scene>
      <ambient>0.34 0.37 0.41 1</ambient>
      <background>0.45 0.62 0.82 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
      <sky>
        <clouds>
          <speed>4</speed>
        </clouds>
      </sky>
    </scene>
    <!-- Mid-morning sun from the south-east; the only light, with shadows. -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 200 0 0 0</pose>
      <intensity>1.1</intensity>
      <diffuse>1.0 0.96 0.88 1</diffuse>
      <specular>0.25 0.24 0.22 1</specular>
      <direction>-0.45 0.35 -0.82</direction>
      <attenuation>
        <range>2000</range>
        <constant>1</constant>
        <linear>0</linear>
        <quadratic>0</quadratic>
      </attenuation>
    </light>
"""

WORLD_FOOTER = """    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
  </world>
</sdf>
"""


@functools.lru_cache(maxsize=1)
def _scene():
    meshes = {}
    parts = [WORLD_HEADER, ground_model(meshes)]
    parts.extend(building_model(building, meshes) for building in campus.BUILDINGS)
    parts.append(trees_model(meshes))
    parts.append(hedges_model(meshes))
    parts.append(furniture_model(meshes))
    parts.append(enclosure_models(meshes))
    for name, mesh in car_meshes().items():
        meshes["car_%s.obj" % name] = mesh
    parts.extend(car_model(car) for car in campus.CARS)
    parts.append(WORLD_FOOTER)
    return "".join(parts), meshes


def world_sdf():
    return _scene()[0]


def mesh_files():
    """OBJ text keyed by file name under worlds/meshes."""
    return {name: mesh.text() for name, mesh in sorted(_scene()[1].items())}


def obstacles_yaml():
    lines = [
        "# Generated by tools/generate_campus.py from park_inspection/campus.py. Edit the layout there.",
        "# Each obstacle is [center x, center y, center z, size x, size y, size z] in ENU metres.",
        "/**:",
        "  ros__parameters:",
        "    inspection_points: [%s]" % ", ".join("%.1f" % value for value in campus.flat_inspection_points()),
        "    bounds_min: [%s]" % ", ".join("%.1f" % value for value in campus.BOUNDS_MIN),
        "    bounds_max: [%s]" % ", ".join("%.1f" % value for value in campus.BOUNDS_MAX),
        "    return_altitude: %.1f" % campus.RETURN_ALTITUDE,
        "    obstacles: [",
    ]
    named = campus.obstacles()
    for index, (name, box) in enumerate(named):
        values = ", ".join("%.2f" % value for value in box.center + box.size)
        separator = "," if index < len(named) - 1 else ""
        lines.append("      %s%s  # %s" % (values, separator, name))
    lines.append("    ]")
    return "\n".join(lines) + "\n"


def main():
    from PIL import Image

    world_path = PACKAGE_DIR / "worlds" / "park.sdf"
    mesh_dir = PACKAGE_DIR / "worlds" / MESH_DIR
    yaml_path = PACKAGE_DIR / "config" / "campus_obstacles.yaml"
    mesh_dir.mkdir(exist_ok=True)
    meshes = mesh_files()
    for stale in mesh_dir.glob("*.obj"):
        if stale.name not in meshes:
            stale.unlink()
    for name, text in meshes.items():
        (mesh_dir / name).write_text(text, encoding="utf-8")
    for name, pixels in textures().items():
        Image.fromarray(pixels).save(mesh_dir / name)
    world_path.write_text(world_sdf(), encoding="utf-8")
    yaml_path.write_text(obstacles_yaml(), encoding="utf-8")
    triangles = sum(len(mesh.faces) for mesh in _scene()[1].values())
    print(
        "Wrote %s, %d meshes (%d triangles) and %s: %d buildings, %d trees, %d cars, %d planner boxes"
        % (world_path, len(meshes), triangles, yaml_path, len(campus.BUILDINGS), len(campus.TREES),
           len(campus.CARS), len(campus.obstacles()))
    )


if __name__ == "__main__":
    main()
