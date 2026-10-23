"""Small OBJ mesh builder and procedural textures for the campus generator."""

import math
import random

import numpy as np


def _sub(first, second):
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _cross(first, second):
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first, second):
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _unit(vector):
    length = math.sqrt(_dot(vector, vector))
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _newell(points):
    normal = [0.0, 0.0, 0.0]
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        normal[0] += (current[1] - following[1]) * (current[2] + following[2])
        normal[1] += (current[2] - following[2]) * (current[0] + following[0])
        normal[2] += (current[0] - following[0]) * (current[1] + following[1])
    return tuple(normal)


class Mesh:
    """Triangle mesh with flat or per-vertex normals, written as OBJ.

    With ``uv_scale`` set, texture coordinates are planar projections of the
    vertex position divided by ``uv_scale`` metres, so textures tile in world
    units regardless of the polygon size.
    """

    def __init__(self, uv_scale=None):
        self.uv_scale = uv_scale
        self.vertices = []
        self.uvs = []
        self.normals = []
        self.faces = []
        self._normal_ids = {}

    def _normal(self, normal):
        key = tuple(round(value, 4) + 0.0 for value in normal)
        index = self._normal_ids.get(key)
        if index is None:
            self.normals.append(key)
            index = len(self.normals)
            self._normal_ids[key] = index
        return index

    def _uv(self, point, normal):
        axis = max(range(3), key=lambda item: abs(normal[item]))
        if axis == 2:
            u_value, v_value = point[0], point[1]
        elif axis == 0:
            u_value, v_value = point[1], point[2]
        else:
            u_value, v_value = point[0], point[2]
        return (u_value / self.uv_scale, v_value / self.uv_scale)

    def polygon(self, points, normal=None):
        """Convex planar polygon. ``normal`` only fixes the facing; the winding follows it."""
        computed = _newell(points)
        if _dot(computed, computed) < 1e-12:
            return
        if normal is not None and _dot(computed, normal) < 0.0:
            points = points[::-1]
            computed = (-computed[0], -computed[1], -computed[2])
        normal = _unit(computed)
        normal_id = self._normal(normal)
        first = len(self.vertices) + 1
        for point in points:
            self.vertices.append(tuple(point))
            if self.uv_scale:
                self.uvs.append(self._uv(point, normal))
        for index in range(1, len(points) - 1):
            self.faces.append(
                ((first, normal_id), (first + index, normal_id), (first + index + 1, normal_id))
            )

    def box(self, low, high, bottom=False):
        x0, y0, z0 = low
        x1, y1, z1 = high
        self.polygon([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1))
        if bottom:
            self.polygon([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)], (0, 0, -1))
        self.polygon([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0))
        self.polygon([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)], (-1, 0, 0))
        self.polygon([(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)], (0, 1, 0))
        self.polygon([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0))

    def quad_up(self, x0, y0, x1, y1, z_value):
        self.polygon([(x0, y0, z_value), (x1, y0, z_value), (x1, y1, z_value), (x0, y1, z_value)], (0, 0, 1))

    def prism_y(self, profile, y0, y1):
        """Extrude a convex side profile of (x, z) points across y0..y1."""
        self.polygon([(x, y1, z) for x, z in profile], (0, 1, 0))
        self.polygon([(x, y0, z) for x, z in profile], (0, -1, 0))
        center = (
            sum(x for x, _ in profile) / len(profile),
            sum(z for _, z in profile) / len(profile),
        )
        for index, (x_a, z_a) in enumerate(profile):
            x_b, z_b = profile[(index + 1) % len(profile)]
            outward = ((x_a + x_b) / 2.0 - center[0], 0.0, (z_a + z_b) / 2.0 - center[1])
            self.polygon([(x_a, y0, z_a), (x_b, y0, z_b), (x_b, y1, z_b), (x_a, y1, z_a)], outward)

    def cylinder(self, start, end, radius_start, radius_end, segments=8, caps=True):
        axis = _unit(_sub(end, start))
        helper = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
        first = _unit(_cross(axis, helper))
        second = _cross(axis, first)
        directions = []
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            directions.append(
                tuple(math.cos(angle) * first[k] + math.sin(angle) * second[k] for k in range(3))
            )
        ring_a = [tuple(start[k] + radius_start * d[k] for k in range(3)) for d in directions]
        ring_b = [tuple(end[k] + radius_end * d[k] for k in range(3)) for d in directions]
        for index in range(segments):
            following = (index + 1) % segments
            outward = tuple(directions[index][k] + directions[following][k] for k in range(3))
            self.polygon([ring_a[index], ring_a[following], ring_b[following], ring_b[index]], outward)
        if caps:
            self.polygon(ring_b, axis)
            self.polygon(ring_a, (-axis[0], -axis[1], -axis[2]))

    def blob(self, center, radius, rng, rings=4, sectors=7, jitter=0.1):
        """Low-poly lumpy sphere. Every vertex stays within ``radius * (1 + jitter)``."""
        def point(polar, azimuth):
            scale = radius * (1.0 - jitter + 2.0 * jitter * rng.random())
            return (
                center[0] + scale * math.sin(polar) * math.cos(azimuth),
                center[1] + scale * math.sin(polar) * math.sin(azimuth),
                center[2] + scale * math.cos(polar),
            )

        top = point(0.0, 0.0)
        bottom = point(math.pi, 0.0)
        offset = rng.random() * 2.0 * math.pi
        layers = [
            [
                point(math.pi * ring / rings, offset + 2.0 * math.pi * (sector + 0.5 * (ring % 2)) / sectors)
                for sector in range(sectors)
            ]
            for ring in range(1, rings)
        ]

        def face(points):
            centroid = tuple(sum(p[k] for p in points) / len(points) for k in range(3))
            self.polygon(points, _sub(centroid, center))

        for sector in range(sectors):
            following = (sector + 1) % sectors
            face([top, layers[0][sector], layers[0][following]])
            face([bottom, layers[-1][following], layers[-1][sector]])
            for ring in range(len(layers) - 1):
                upper, lower = layers[ring], layers[ring + 1]
                face([upper[sector], lower[sector], lower[following]])
                face([upper[sector], lower[following], upper[following]])

    def extend(self, other, scale=1.0, yaw=0.0, offset=(0.0, 0.0, 0.0)):
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        def rotate(vector):
            return (
                cos_yaw * vector[0] - sin_yaw * vector[1],
                sin_yaw * vector[0] + cos_yaw * vector[1],
                vector[2],
            )

        base = len(self.vertices)
        for vertex in other.vertices:
            rotated = rotate(vertex)
            moved = tuple(rotated[k] * scale + offset[k] for k in range(3))
            self.vertices.append(moved)
        if self.uv_scale:
            if other.uv_scale:
                self.uvs.extend(other.uvs)
            else:
                self.uvs.extend((vertex[0] / self.uv_scale, vertex[2] / self.uv_scale)
                                for vertex in self.vertices[base:])
        normal_map = [self._normal(rotate(normal)) for normal in other.normals]
        for face in other.faces:
            self.faces.append(tuple((base + vertex, normal_map[normal - 1]) for vertex, normal in face))

    def raw(self, vertices, normals, uvs, triangles):
        """Append an indexed mesh with per-vertex normals."""
        base = len(self.vertices)
        self.vertices.extend(vertices)
        if self.uv_scale:
            self.uvs.extend(uvs)
        normal_ids = [self._normal(normal) for normal in normals]
        for triangle in triangles:
            self.faces.append(tuple((base + index + 1, normal_ids[index]) for index in triangle))

    def bounds(self, start=0):
        points = self.vertices[start:]
        return (
            tuple(min(point[k] for point in points) for k in range(3)),
            tuple(max(point[k] for point in points) for k in range(3)),
        )

    def text(self):
        lines = ["# Generated by tools/generate_campus.py"]
        lines.extend("v %.3f %.3f %.3f" % vertex for vertex in self.vertices)
        if self.uv_scale:
            lines.extend("vt %.4f %.4f" % uv for uv in self.uvs)
        lines.extend("vn %.4f %.4f %.4f" % normal for normal in self.normals)
        if self.uv_scale:
            corner = "%d/%d/%d"
            lines.extend(
                "f " + " ".join(corner % (vertex, vertex, normal) for vertex, normal in face)
                for face in self.faces
            )
        else:
            lines.extend(
                "f " + " ".join("%d//%d" % (vertex, normal) for vertex, normal in face)
                for face in self.faces
            )
        return "\n".join(lines) + "\n"


def seeded(seed):
    return random.Random(seed)


def _periodic_noise(size, rng, falloff, low_cut=1.0):
    """Tileable noise: random phases with a 1/f^falloff spectrum, normalised to 0..1."""
    frequency_x = np.fft.fftfreq(size)[:, None] * size
    frequency_y = np.fft.fftfreq(size)[None, :] * size
    radius = np.sqrt(frequency_x ** 2 + frequency_y ** 2)
    radius[0, 0] = 1.0
    amplitude = 1.0 / np.maximum(radius, low_cut) ** falloff
    amplitude[0, 0] = 0.0
    phases = rng.uniform(0.0, 2.0 * np.pi, (size, size))
    field = np.real(np.fft.ifft2(amplitude * np.exp(1j * phases)))
    field -= field.min()
    return field / field.max()


def _colorize(base, variation, *fields):
    image = np.zeros(fields[0].shape + (3,))
    for channel in range(3):
        value = base[channel]
        for weight, field in zip(variation, fields):
            value = value + weight[channel] * (field - 0.5)
        image[..., channel] = value
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def textures():
    """Tileable albedo maps keyed by file name."""
    rng = np.random.default_rng(7)
    size = 512
    coarse = _periodic_noise(size, rng, 2.2)
    medium = _periodic_noise(size, rng, 1.2, 8.0)
    fine = rng.random((size, size))
    grass = _colorize(
        (0.31, 0.43, 0.19),
        ((0.10, 0.10, 0.04), (0.08, 0.10, 0.03), (0.07, 0.09, 0.04)),
        coarse, medium, fine,
    )
    asphalt = _colorize(
        (0.20, 0.20, 0.21),
        ((0.04, 0.04, 0.04), (0.03, 0.03, 0.03), (0.10, 0.10, 0.10)),
        _periodic_noise(size, rng, 2.0), _periodic_noise(size, rng, 1.0, 16.0), rng.random((size, size)),
    )
    # 2 m texture tile holding 4 x 4 pavers of 0.5 m with grout lines.
    paving_shade = rng.uniform(-0.04, 0.04, (4, 4)).repeat(size // 4, 0).repeat(size // 4, 1)
    paving = _colorize(
        (0.63, 0.61, 0.57),
        ((0.03, 0.03, 0.03), (0.06, 0.06, 0.06)),
        _periodic_noise(size, rng, 1.0, 16.0), rng.random((size, size)),
    ).astype(np.float64)
    paving += paving_shade[..., None] * 255.0
    grout = np.zeros((size, size), dtype=bool)
    for line in range(0, size, size // 4):
        grout[line:line + 3, :] = True
        grout[:, line:line + 3] = True
    paving[grout] *= 0.62
    paving = np.clip(paving, 0, 255).astype(np.uint8)
    leaves = _colorize(
        (0.36, 0.56, 0.24),
        ((0.16, 0.20, 0.08), (0.12, 0.14, 0.06)),
        _periodic_noise(256, rng, 0.9, 6.0), rng.random((256, 256)),
    )
    stripes = (np.arange(size)[:, None] // (size // 2)) % 2 * np.ones((1, size))
    turf = _colorize(
        (0.22, 0.47, 0.20),
        ((0.06, 0.08, 0.04), (0.03, 0.04, 0.02), (0.04, 0.05, 0.03)),
        stripes, _periodic_noise(size, rng, 1.5), rng.random((size, size)),
    )
    concrete = _colorize(
        (0.80, 0.80, 0.80),
        ((0.10, 0.10, 0.10), (0.05, 0.05, 0.05)),
        _periodic_noise(256, rng, 1.6), rng.random((256, 256)),
    )
    return {
        "grass.png": grass,
        "asphalt.png": asphalt,
        "paving.png": paving,
        "leaves.png": leaves,
        "turf.png": turf,
        "concrete.png": concrete,
    }
