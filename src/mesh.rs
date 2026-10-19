//! Vertex/index mesh storage and procedural mesh generators.
//! Made by AeroForger

use crate::math::Vec3;

/// A single mesh vertex: position plus surface normal, both in model space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    /// Position in model space.
    pub pos: Vec3,
    /// Surface normal in model space (should be unit length).
    pub normal: Vec3,
}

impl Vertex {
    /// Builds a vertex from a position and normal.
    pub fn new(pos: Vec3, normal: Vec3) -> Self {
        Self { pos, normal }
    }
}

/// An indexed triangle mesh.
///
/// `indices` form triangles in groups of three. Winding must be
/// counter-clockwise when viewed from outside (backface culling depends on it).
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// Unique vertices.
    pub vertices: Vec<Vertex>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Builds a mesh from raw vertex and index buffers.
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Raw triangle soup: every group of 3 vertices becomes a face with its
    /// geometric normal (vertices are not shared).
    pub fn from_triangles(tris: &[(Vec3, Vec3, Vec3)]) -> Mesh {
        let mut vertices = Vec::with_capacity(tris.len() * 3);
        let mut indices = Vec::with_capacity(tris.len() * 3);
        for (a, b, c) in tris {
            let n = b.sub(*a).cross(c.sub(*a)).normalize();
            let base = vertices.len() as u32;
            indices.extend([base, base + 1, base + 2]);
            vertices.push(Vertex::new(*a, n));
            vertices.push(Vertex::new(*b, n));
            vertices.push(Vertex::new(*c, n));
        }
        Mesh::new(vertices, indices)
    }

    /// Axis-aligned cube centered at the origin, side length 2.
    pub fn cube() -> Mesh {
        let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let faces: [[Vec3; 4]; 6] = [
            // +X, -X, +Y, -Y, +Z, -Z. All wound CCW from outside.
            [
                v(1.0, -1.0, -1.0),
                v(1.0, 1.0, -1.0),
                v(1.0, 1.0, 1.0),
                v(1.0, -1.0, 1.0),
            ],
            [
                v(-1.0, -1.0, 1.0),
                v(-1.0, 1.0, 1.0),
                v(-1.0, 1.0, -1.0),
                v(-1.0, -1.0, -1.0),
            ],
            [
                v(-1.0, 1.0, -1.0),
                v(-1.0, 1.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(1.0, 1.0, -1.0),
            ],
            [
                v(-1.0, -1.0, 1.0),
                v(-1.0, -1.0, -1.0),
                v(1.0, -1.0, -1.0),
                v(1.0, -1.0, 1.0),
            ],
            [
                v(1.0, -1.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(-1.0, 1.0, 1.0),
                v(-1.0, -1.0, 1.0),
            ],
            [
                v(-1.0, -1.0, -1.0),
                v(-1.0, 1.0, -1.0),
                v(1.0, 1.0, -1.0),
                v(1.0, -1.0, -1.0),
            ],
        ];
        let mut tris = Vec::with_capacity(12);
        for f in faces {
            tris.push((f[0], f[1], f[2]));
            tris.push((f[0], f[2], f[3]));
        }
        Mesh::from_triangles(&tris)
    }

    /// UV sphere centered at the origin, radius 1.
    pub fn sphere(stacks: usize, slices: usize) -> Mesh {
        let mut mesh = Mesh::default();
        for i in 0..=stacks {
            let theta = std::f32::consts::PI * i as f32 / stacks as f32;
            for j in 0..=slices {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
                let st = theta.sin();
                let n = Vec3::new(st * phi.cos(), theta.cos(), st * phi.sin());
                mesh.vertices.push(Vertex::new(n, n));
            }
        }
        let idx = |i: usize, j: usize| (i * (slices + 1) + j) as u32;
        for i in 0..stacks {
            for j in 0..slices {
                let a = idx(i, j);
                let b = idx(i + 1, j);
                let c = idx(i + 1, j + 1);
                let d = idx(i, j + 1);
                // Winding chosen so face normals point outward.
                if i != 0 {
                    mesh.indices.extend([a, d, b]);
                }
                if i + 1 != stacks {
                    mesh.indices.extend([b, d, c]);
                }
            }
        }
        mesh
    }

    /// Torus centered at the origin: major radius `big_r`, minor radius
    /// `small_r`, lying flat in the XZ plane.
    pub fn torus(major: usize, minor: usize, big_r: f32, small_r: f32) -> Mesh {
        let mut mesh = Mesh::default();
        for i in 0..=major {
            let u = 2.0 * std::f32::consts::PI * i as f32 / major as f32;
            let (su, cu) = u.sin_cos();
            for j in 0..=minor {
                let v = 2.0 * std::f32::consts::PI * j as f32 / minor as f32;
                let (sv, cv) = v.sin_cos();
                let center = Vec3::new(big_r * cu, 0.0, big_r * su);
                let n = Vec3::new(cv * cu, sv, cv * su);
                mesh.vertices
                    .push(Vertex::new(center.add(n.scale(small_r)), n));
            }
        }
        let idx = |i: usize, j: usize| (i * (minor + 1) + j) as u32;
        for i in 0..major {
            for j in 0..minor {
                let a = idx(i, j);
                let b = idx(i + 1, j);
                let c = idx(i + 1, j + 1);
                let d = idx(i, j + 1);
                mesh.indices.extend([a, d, b]);
                mesh.indices.extend([b, d, c]);
            }
        }
        mesh
    }

    /// Number of triangles in the index buffer.
    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Mesh-space bounding sphere radius (distance of the farthest vertex).
    pub fn bounding_radius(&self) -> f32 {
        self.vertices
            .iter()
            .map(|v| v.pos.length())
            .fold(0.0f32, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Winding of each triangle must agree with its face normal so backface
    /// culling and lighting don't fight each other.
    pub(super) fn assert_winding_matches_normals(mesh: &Mesh, name: &str) {
        for tri in mesh.indices.chunks(3) {
            let a = mesh.vertices[tri[0] as usize];
            let b = mesh.vertices[tri[1] as usize];
            let c = mesh.vertices[tri[2] as usize];
            let face_n = b.pos.sub(a.pos).cross(c.pos.sub(a.pos)).normalize();
            for v in [a, b, c] {
                assert!(
                    face_n.dot(v.normal) > 0.0,
                    "{name}: triangle winding disagrees with vertex normal"
                );
            }
        }
    }

    #[test]
    fn cube_has_six_faces_and_outward_winding() {
        let cube = Mesh::cube();
        assert_eq!(cube.vertices.len(), 36, "triangle soup: 12 tris x 3 verts");
        assert_eq!(cube.tri_count(), 12);
        assert!((cube.bounding_radius() - 3.0f32.sqrt()).abs() < 1e-4);
        assert_winding_matches_normals(&cube, "cube");
    }

    #[test]
    fn cube_covers_all_six_axis_directions() {
        let cube = Mesh::cube();
        for axis in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ] {
            for sign in [1.0f32, -1.0] {
                let target = axis.scale(sign);
                let hits = cube
                    .indices
                    .chunks(3)
                    .filter(|t| cube.vertices[t[0] as usize].normal.dot(target) > 0.99)
                    .count();
                assert_eq!(hits, 2, "face {target:?} must have exactly 2 triangles");
            }
        }
    }

    #[test]
    fn sphere_normals_point_outward() {
        let sphere = Mesh::sphere(8, 12);
        // Pole rows emit one triangle per quad, middle rows two.
        assert_eq!(sphere.tri_count(), 8 * 12 * 2 - 2 * 12);
        assert_winding_matches_normals(&sphere, "sphere");
        assert!((sphere.bounding_radius() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn torus_has_expected_topology_and_outward_winding() {
        let t = Mesh::torus(24, 12, 0.9, 0.35);
        assert_eq!(t.tri_count(), 24 * 12 * 2);
        assert_winding_matches_normals(&t, "torus");
        assert!((t.bounding_radius() - 1.25).abs() < 1e-4);
    }
}
