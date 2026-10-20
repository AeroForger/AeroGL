//! A small Wavefront OBJ loader: positions, UVs, normals, and polygon faces.
//! Made by AeroForger

use crate::math::Vec3;
use crate::mesh::{Mesh, Vertex};
use std::collections::HashMap;
use std::fmt;

/// A parse or I/O failure while loading an OBJ file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjError {
    line: usize,
    message: String,
}

impl ObjError {
    /// The 1-based source line, or `0` for I/O errors.
    pub fn line(&self) -> usize {
        self.line
    }

    /// A human-readable explanation.
    pub fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn at(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "OBJ: {}", self.message)
        } else {
            write!(f, "OBJ line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for ObjError {}

/// Parses OBJ source text into an indexed [`Mesh`].
///
/// Supports `v`, `vt`, `vn`, `f` (triangles and convex polygons), comments,
/// and negative (relative) indices. Missing normals are generated as smooth
/// vertex normals; missing UVs default to `[0,0]`. Material and group
/// statements are ignored.
pub(crate) fn parse(source: &str) -> Result<Mesh, ObjError> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    let mut mesh = Mesh::default();
    let mut needs_normal: Vec<bool> = Vec::new();
    let mut normal_sum: Vec<Vec3> = Vec::new();
    let mut map: HashMap<(usize, usize, usize), u32> = HashMap::new();

    for (line_index, raw) in source.lines().enumerate() {
        let line_no = line_index + 1;
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(tag) = parts.next() else {
            continue;
        };
        match tag {
            "v" => {
                let x = parse_f32(parts.next(), line_no, "vertex x")?;
                let y = parse_f32(parts.next(), line_no, "vertex y")?;
                let z = parse_f32(parts.next(), line_no, "vertex z")?;
                positions.push(Vec3::new(x, y, z));
            }
            "vn" => {
                let x = parse_f32(parts.next(), line_no, "normal x")?;
                let y = parse_f32(parts.next(), line_no, "normal y")?;
                let z = parse_f32(parts.next(), line_no, "normal z")?;
                normals.push(Vec3::new(x, y, z).normalize());
            }
            "vt" => {
                let u = parse_f32(parts.next(), line_no, "uv u")?;
                let v = parts
                    .next()
                    .map(|t| parse_f32(Some(t), line_no, "uv v"))
                    .transpose()?
                    .unwrap_or(0.0);
                uvs.push([u, v]);
            }
            "f" => {
                let face: Vec<&str> = parts.collect();
                if face.len() < 3 {
                    return Err(ObjError::at(line_no, "face needs at least 3 vertices"));
                }
                let mut resolved = Vec::with_capacity(face.len());
                for token in face {
                    let (v, vt, vn) = parse_face_vertex(
                        token,
                        line_no,
                        positions.len(),
                        uvs.len(),
                        normals.len(),
                    )?;
                    resolved.push((v, vt, vn));
                }
                for i in 1..resolved.len() - 1 {
                    for &(v, vt, vn) in &[resolved[0], resolved[i], resolved[i + 1]] {
                        let key = (v, vt, vn);
                        let index = *map.entry(key).or_insert_with(|| {
                            let pos = positions[v];
                            let normal = if vn < usize::MAX {
                                normals[vn]
                            } else {
                                Vec3::ZERO
                            };
                            let uv = if vt < usize::MAX { uvs[vt] } else { [0.0, 0.0] };
                            let idx = mesh.vertices.len() as u32;
                            mesh.vertices.push(Vertex::with_uv(pos, normal, uv));
                            needs_normal.push(vn == usize::MAX);
                            normal_sum.push(Vec3::ZERO);
                            idx
                        });
                        mesh.indices.push(index);
                    }
                }
            }
            // Groups, smoothing, object names, materials: not needed to render.
            _ => {}
        }
    }

    if mesh.indices.is_empty() {
        return Err(ObjError::at(0, "no faces found"));
    }

    // Generate smooth normals for faces that did not supply any.
    if needs_normal.iter().any(|&n| n) {
        for tri in mesh.indices.as_chunks::<3>().0 {
            let a = mesh.vertices[tri[0] as usize].pos;
            let b = mesh.vertices[tri[1] as usize].pos;
            let c = mesh.vertices[tri[2] as usize].pos;
            let face = b.sub(a).cross(c.sub(a));
            for &i in tri {
                normal_sum[i as usize] = normal_sum[i as usize].add(face);
            }
        }
        for (i, vertex) in mesh.vertices.iter_mut().enumerate() {
            if needs_normal[i] {
                let n = normal_sum[i].normalize();
                vertex.normal = if n.length() > 0.0 {
                    n
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };
            }
        }
    }

    mesh.validate()
        .map_err(|error| ObjError::at(0, error.to_string()))?;
    Ok(mesh)
}

fn parse_f32(token: Option<&str>, line: usize, what: &str) -> Result<f32, ObjError> {
    let token = token.ok_or_else(|| ObjError::at(line, format!("missing {what}")))?;
    token
        .parse::<f32>()
        .map_err(|_| ObjError::at(line, format!("invalid {what}: {token:?}")))
}

/// Resolves `v/vt/vn` (or any shorter form) to zero-based indices.
/// `usize::MAX` means "absent".
fn parse_face_vertex(
    token: &str,
    line: usize,
    position_count: usize,
    uv_count: usize,
    normal_count: usize,
) -> Result<(usize, usize, usize), ObjError> {
    let mut fields = token.split('/');
    let v = parse_index(fields.next(), position_count, line, "position")?
        .ok_or_else(|| ObjError::at(line, "face vertex is missing a position"))?;
    let vt = parse_index(fields.next(), uv_count, line, "uv")?.unwrap_or(usize::MAX);
    let vn = parse_index(fields.next(), normal_count, line, "normal")?.unwrap_or(usize::MAX);
    Ok((v, vt, vn))
}

fn parse_index(
    token: Option<&str>,
    count: usize,
    line: usize,
    what: &str,
) -> Result<Option<usize>, ObjError> {
    let Some(token) = token else {
        return Ok(None);
    };
    if token.is_empty() {
        return Ok(None);
    }
    let raw: i64 = token
        .parse()
        .map_err(|_| ObjError::at(line, format!("invalid {what} index: {token:?}")))?;
    if raw == 0 {
        return Err(ObjError::at(line, "OBJ indices are 1-based; 0 is invalid"));
    }
    let index = if raw > 0 { raw - 1 } else { count as i64 + raw };
    if index < 0 || index >= count as i64 {
        return Err(ObjError::at(
            line,
            format!("{what} index {raw} is out of range (have {count})"),
        ));
    }
    Ok(Some(index as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUAD: &str = "\
# a quad with UVs and normals
v -1 0 -1
v  1 0 -1
v  1 0  1
v -1 0  1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 1 0
f 1/1/1 2/2/1 3/3/1 4/4/1
";

    #[test]
    fn parses_quad_into_two_triangles() {
        let mesh = parse(QUAD).unwrap();
        assert_eq!(mesh.tri_count(), 2);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.vertices[0].uv, [0.0, 0.0]);
        assert!((mesh.vertices[0].normal.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn negative_indices_are_relative() {
        let src = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
        let mesh = parse(src).unwrap();
        assert_eq!(mesh.tri_count(), 1);
        assert_eq!(mesh.vertices.len(), 3);
    }

    #[test]
    fn missing_normals_are_generated_and_normalized() {
        let src = "\
v -1 0 -1
v  1 0 -1
v  1 0  1
v -1 0  1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3 4/4
";
        let mesh = parse(src).unwrap();
        for v in &mesh.vertices {
            assert!((v.normal.length() - 1.0).abs() < 1e-4, "{:?}", v.normal);
        }
    }

    #[test]
    fn bad_indices_report_the_line() {
        let src = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n";
        let err = parse(src).unwrap_err();
        assert_eq!(err.line(), 4);
        assert!(err.message().contains("out of range"), "{err}");
    }

    #[test]
    fn empty_files_are_an_error() {
        assert!(parse("# nothing here\n").is_err());
    }
}
