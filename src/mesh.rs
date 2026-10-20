//! Vertex/index mesh storage, UVs, and procedural mesh generators.
//! Made by AeroForger

use crate::math::Vec3;
use std::fmt;

/// A problem with a mesh's vertex or index data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// The index buffer is not a whole number of triangles.
    IncompleteTriangle {
        /// Number of entries in the index buffer.
        index_count: usize,
    },
    /// A triangle references a vertex that does not exist.
    IndexOutOfRange {
        /// Zero-based triangle number.
        triangle: usize,
        /// The bad index.
        index: u32,
        /// Number of vertices in the mesh.
        vertex_count: usize,
    },
    /// A vertex position, normal, or UV contains NaN or infinity.
    NonFiniteVertex {
        /// Zero-based vertex number.
        vertex: usize,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompleteTriangle { index_count } => write!(
                f,
                "index buffer has {index_count} entries; a mesh needs a multiple of 3"
            ),
            Self::IndexOutOfRange {
                triangle,
                index,
                vertex_count,
            } => write!(
                f,
                "triangle {triangle} uses vertex index {index}, but the mesh has {vertex_count} vertices"
            ),
            Self::NonFiniteVertex { vertex } => {
                write!(
                    f,
                    "vertex {vertex} has a non-finite position, normal, or UV"
                )
            }
        }
    }
}

impl std::error::Error for MeshError {}

/// A single mesh vertex: model-space position, normal, and texture coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    /// Position in model space.
    pub pos: Vec3,
    /// Surface normal in model space (should be unit length).
    pub normal: Vec3,
    /// Texture coordinate, usually in `[0,1]`.
    pub uv: [f32; 2],
}

impl Vertex {
    /// Builds an untextured vertex from a position and normal (`uv = [0,0]`).
    pub fn new(pos: Vec3, normal: Vec3) -> Self {
        Self {
            pos,
            normal,
            uv: [0.0, 0.0],
        }
    }

    /// Builds a vertex with a texture coordinate.
    pub fn with_uv(pos: Vec3, normal: Vec3, uv: [f32; 2]) -> Self {
        Self { pos, normal, uv }
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
    ///
    /// # Panics
    ///
    /// Panics if [`Mesh::validate`] rejects the data. Use [`Mesh::try_new`] for
    /// a recoverable error.
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self::try_new(vertices, indices).unwrap_or_else(|error| panic!("invalid mesh: {error}"))
    }

    /// Builds a mesh, returning an error instead of panicking on bad data.
    pub fn try_new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Result<Self, MeshError> {
        let mesh = Self { vertices, indices };
        mesh.validate()?;
        Ok(mesh)
    }

    /// Checks that indices form whole triangles, point at real vertices, and
    /// that every vertex component is finite.
    pub fn validate(&self) -> Result<(), MeshError> {
        if !self.indices.len().is_multiple_of(3) {
            return Err(MeshError::IncompleteTriangle {
                index_count: self.indices.len(),
            });
        }
        for (triangle, tri) in self.indices.as_chunks::<3>().0.iter().enumerate() {
            for &index in tri {
                if index as usize >= self.vertices.len() {
                    return Err(MeshError::IndexOutOfRange {
                        triangle,
                        index,
                        vertex_count: self.vertices.len(),
                    });
                }
            }
        }
        for (vertex, v) in self.vertices.iter().enumerate() {
            if !is_finite(v.pos)
                || !is_finite(v.normal)
                || !v.uv[0].is_finite()
                || !v.uv[1].is_finite()
            {
                return Err(MeshError::NonFiniteVertex { vertex });
            }
        }
        Ok(())
    }

    /// Raw triangle soup: every group of 3 vertices becomes a face with its
    /// geometric normal (vertices are not shared).
    pub fn from_triangles(tris: &[(Vec3, Vec3, Vec3)]) -> Mesh {
        let mut vertices = Vec::with_capacity(tris.len() * 3);
        let mut indices = Vec::with_capacity(tris.len() * 3);
        for (a, b, c) in tris {
            let cross = b.sub(*a).cross(c.sub(*a));
            let n = if cross.length() > 1e-12 {
                cross.normalize()
            } else {
                // A zero-area triangle cannot be seen from any side. Keep it
                // with a usable normal so the data stays valid.
                Vec3::new(0.0, 1.0, 0.0)
            };
            let base = vertices.len() as u32;
            indices.extend([base, base + 1, base + 2]);
            vertices.push(Vertex::new(*a, n));
            vertices.push(Vertex::new(*b, n));
            vertices.push(Vertex::new(*c, n));
        }
        Mesh::new(vertices, indices)
    }

    /// Axis-aligned cube centered at the origin, side length 2.
    ///
    /// Each face is a separate quad with UVs covering the full `[0,1]` square,
    /// so a single texture appears once per side.
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
        let face_uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        let mut mesh = Mesh::default();
        for face in faces {
            let normal = face[1].sub(face[0]).cross(face[2].sub(face[0])).normalize();
            let base = mesh.vertices.len() as u32;
            for (corner, uv) in face.into_iter().zip(face_uvs) {
                mesh.vertices.push(Vertex::with_uv(corner, normal, uv));
            }
            mesh.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Mesh::new(mesh.vertices, mesh.indices)
    }

    /// UV sphere centered at the origin, radius 1.
    ///
    /// `stacks` and `slices` are clamped to the smallest values that still make
    /// a closed sphere.
    pub fn sphere(stacks: usize, slices: usize) -> Mesh {
        let stacks = stacks.max(2);
        let slices = slices.max(3);
        let mut mesh = Mesh::default();
        for i in 0..=stacks {
            let theta = std::f32::consts::PI * i as f32 / stacks as f32;
            for j in 0..=slices {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
                let st = theta.sin();
                let n = Vec3::new(st * phi.cos(), theta.cos(), st * phi.sin());
                let uv = [j as f32 / slices as f32, i as f32 / stacks as f32];
                mesh.vertices.push(Vertex::with_uv(n, n, uv));
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
        Mesh::new(mesh.vertices, mesh.indices)
    }

    /// Torus centered at the origin: major radius `big_r`, minor radius
    /// `small_r`, lying flat in the XZ plane.
    ///
    /// Segment counts are clamped to at least 3.
    pub fn torus(major: usize, minor: usize, big_r: f32, small_r: f32) -> Mesh {
        assert!(
            big_r.is_finite() && small_r.is_finite(),
            "torus radii must be finite"
        );
        let major = major.max(3);
        let minor = minor.max(3);
        let mut mesh = Mesh::default();
        for i in 0..=major {
            let u = 2.0 * std::f32::consts::PI * i as f32 / major as f32;
            let (su, cu) = u.sin_cos();
            for j in 0..=minor {
                let v = 2.0 * std::f32::consts::PI * j as f32 / minor as f32;
                let (sv, cv) = v.sin_cos();
                let center = Vec3::new(big_r * cu, 0.0, big_r * su);
                let n = Vec3::new(cv * cu, sv, cv * su);
                let uv = [i as f32 / major as f32, j as f32 / minor as f32];
                mesh.vertices
                    .push(Vertex::with_uv(center.add(n.scale(small_r)), n, uv));
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
        Mesh::new(mesh.vertices, mesh.indices)
    }

    /// Flat, upward-facing plane centered at the origin in the XZ plane.
    ///
    /// `size` is the total side length and `subdivisions` is the number of
    /// quads per side. UVs cover `[0,1]`, so a texture is stretched across the
    /// whole plane. A subdivided plane also gives the vertex lighting enough
    /// vertices to show local point lights and shadow-map shading across a
    /// floor.
    pub fn plane(size: f32, subdivisions: usize) -> Mesh {
        assert!(size.is_finite(), "plane size must be finite");
        let n = subdivisions.max(1);
        let half = size * 0.5;
        let mut mesh = Mesh::default();
        let normal = Vec3::new(0.0, 1.0, 0.0);
        for i in 0..=n {
            let v = i as f32 / n as f32;
            for j in 0..=n {
                let u = j as f32 / n as f32;
                let pos = Vec3::new(-half + size * u, 0.0, -half + size * v);
                mesh.vertices.push(Vertex::with_uv(pos, normal, [u, v]));
            }
        }
        let idx = |i: usize, j: usize| (i * (n + 1) + j) as u32;
        for i in 0..n {
            for j in 0..n {
                let a = idx(i, j);
                let b = idx(i + 1, j);
                let c = idx(i + 1, j + 1);
                let d = idx(i, j + 1);
                // (a, b, c) and (a, c, d) point up (+Y).
                mesh.indices.extend([a, b, c, a, c, d]);
            }
        }
        Mesh::new(mesh.vertices, mesh.indices)
    }

    /// Number of triangles in the index buffer.
    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Parses a Wavefront OBJ file from a string.
    ///
    /// See [`Mesh::load_obj`] for the file-based version.
    pub fn from_obj(source: &str) -> Result<Mesh, crate::obj::ObjError> {
        crate::obj::parse(source)
    }

    /// Loads a Wavefront OBJ file from disk.
    pub fn load_obj(path: impl AsRef<std::path::Path>) -> Result<Mesh, crate::obj::ObjError> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|e| {
            crate::obj::ObjError::at(0, format!("cannot read {}: {e}", path.display()))
        })?;
        crate::obj::parse(&source)
    }

    /// Mesh-space bounding sphere radius (distance of the farthest vertex).
    pub fn bounding_radius(&self) -> f32 {
        self.vertices
            .iter()
            .map(|v| v.pos.length())
            .fold(0.0f32, f32::max)
    }

    /// Mesh-space axis-aligned bounds `(min, max)`, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.vertices.iter().map(|v| v.pos);
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }

    /// Center of the mesh-space bounding box, or [`Vec3::ZERO`] when empty.
    pub fn center(&self) -> Vec3 {
        match self.bounds() {
            Some((lo, hi)) => lo.add(hi).scale(0.5),
            None => Vec3::ZERO,
        }
    }
}

fn is_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
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
        assert_eq!(cube.vertices.len(), 24, "4 corners per face");
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

    #[test]
    fn plane_faces_up_and_has_uvs() {
        let plane = Mesh::plane(10.0, 4);
        assert_eq!(plane.tri_count(), 4 * 4 * 2);
        assert_winding_matches_normals(&plane, "plane");
        for v in &plane.vertices {
            assert!(v.normal.y > 0.99);
            assert!((0.0..=1.0).contains(&v.uv[0]) && (0.0..=1.0).contains(&v.uv[1]));
        }
        let (lo, hi) = plane.bounds().unwrap();
        assert!((hi.x - lo.x - 10.0).abs() < 1e-4);
        assert!((hi.z - lo.z - 10.0).abs() < 1e-4);
    }

    #[test]
    fn generated_meshes_have_uvs_in_range() {
        for mesh in [
            Mesh::cube(),
            Mesh::sphere(8, 12),
            Mesh::torus(12, 8, 1.0, 0.3),
            Mesh::plane(4.0, 2),
        ] {
            assert!(
                mesh.vertices
                    .iter()
                    .all(|v| v.uv[0].is_finite() && v.uv[1].is_finite())
            );
        }
    }

    #[test]
    fn try_new_reports_incomplete_and_out_of_range_indices() {
        let v = vec![Vertex::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
        assert!(matches!(
            Mesh::try_new(v.clone(), vec![0, 0]),
            Err(MeshError::IncompleteTriangle { index_count: 2 })
        ));
        assert!(matches!(
            Mesh::try_new(v, vec![0, 0, 1]),
            Err(MeshError::IndexOutOfRange {
                triangle: 0,
                index: 1,
                vertex_count: 1
            })
        ));
    }

    #[test]
    fn try_new_reports_non_finite_vertices() {
        let v = vec![Vertex::new(
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )];
        assert!(matches!(
            Mesh::try_new(v, vec![0, 0, 0]),
            Err(MeshError::NonFiniteVertex { vertex: 0 })
        ));
    }

    #[test]
    fn zero_segment_generators_still_make_finite_geometry() {
        for mesh in [Mesh::sphere(0, 0), Mesh::torus(0, 0, 1.0, 0.3)] {
            assert!(mesh.tri_count() > 0);
            assert!(mesh.validate().is_ok());
            assert!(mesh.vertices.iter().all(|v| is_finite(v.pos)));
        }
    }

    #[test]
    fn degenerate_triangle_soup_gets_a_fallback_normal() {
        let p = Vec3::new(0.0, 0.0, 0.0);
        let mesh = Mesh::from_triangles(&[(p, p, p)]);
        assert_eq!(mesh.tri_count(), 1);
        assert!(mesh.validate().is_ok());
        assert!(mesh.vertices[0].normal.length() > 0.99);
    }
}
