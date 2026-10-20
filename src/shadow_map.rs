//! Directional shadow mapping: a depth-only pass from the sun, sampled when
//! shading vertices.
//! Made by AeroForger

use crate::draw::triangle::draw_triangle_depth;
use crate::math::{Mat4, Vec3};
use crate::scene::Scene;

/// Shadow-map quality and response settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowSettings {
    /// Depth-map resolution (one side; the map is square).
    pub size: usize,
    /// Depth bias in NDC units, used to fight shadow acne.
    pub bias: f32,
    /// How dark a fully shadowed surface becomes (`0` = black, `1` = no
    /// shadow). The visibility factor is `1 - darkness * coverage`.
    pub darkness: f32,
    /// Percentage-closer filtering radius in texels (`0` = one sample).
    pub pcf_radius: usize,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            size: 1024,
            bias: 0.0025,
            darkness: 0.72,
            pcf_radius: 1,
        }
    }
}

impl ShadowSettings {
    /// A shadow map of `size` x `size` texels.
    pub fn new(size: usize) -> Self {
        Self {
            size: size.max(16),
            ..Self::default()
        }
    }

    /// Sets the depth bias (builder style).
    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    /// Sets the shadow darkness (builder style).
    pub fn with_darkness(mut self, darkness: f32) -> Self {
        self.darkness = darkness.clamp(0.0, 1.0);
        self
    }

    /// Sets the PCF radius in texels (builder style).
    pub fn with_pcf_radius(mut self, radius: usize) -> Self {
        self.pcf_radius = radius.min(4);
        self
    }
}

/// A reusable directional shadow map.
///
/// The map is rendered from an orthographic light camera fitted to the scene's
/// world-space bounds. Shading samples it per vertex, so it works best with
/// reasonably tessellated geometry (the built-in plane and sphere generators
/// are).
pub struct ShadowMap {
    size: usize,
    depth: Vec<f32>,
    view_proj: Mat4,
    bias: f32,
    darkness: f32,
    pcf_radius: usize,
}

impl ShadowMap {
    /// Allocates an empty shadow map.
    pub fn new(settings: ShadowSettings) -> Self {
        let size = settings.size.max(16);
        Self {
            size,
            depth: vec![f32::INFINITY; size * size],
            view_proj: Mat4::identity(),
            bias: settings.bias.max(0.0),
            darkness: settings.darkness.clamp(0.0, 1.0),
            pcf_radius: settings.pcf_radius.min(4),
        }
    }

    /// Resolution in texels.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Renders the depth pass for `scene`, using [`Scene::shadow_map`] settings
    /// and the primary directional light. A no-op when shadow mapping is not
    /// enabled.
    pub fn render(&mut self, scene: &Scene) {
        let Some(settings) = scene.shadow_map else {
            return;
        };
        self.resize(settings);
        self.depth.fill(f32::INFINITY);

        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;
        for object in &scene.objects {
            for v in &object.mesh.vertices {
                let p = object.transform.transform_point(v.pos);
                lo = lo.min(p);
                hi = hi.max(p);
                any = true;
            }
        }
        if !any {
            return;
        }

        let center = lo.add(hi).scale(0.5);
        let radius = hi.sub(lo).length() * 0.5 + 0.001;
        let mut dir = scene.light.direction.normalize();
        if dir.length() < 1e-4 {
            dir = Vec3::new(0.0, -1.0, 0.0);
        }
        let up = if dir.y.abs() > 0.95 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let eye = center.sub(dir.scale(radius * 2.0 + 1.0));
        let view = Mat4::look_at(eye, center, up);
        let extent = radius * 1.1 + 0.5;
        let proj = Mat4::orthographic(-extent, extent, -extent, extent, 0.05, radius * 4.0 + 2.0);
        self.view_proj = proj.multiply(&view);

        let size = self.size;
        let mvp = self.view_proj;
        for object in scene.objects.iter().filter(|o| o.casts_shadow) {
            let model_view_proj = mvp.multiply(&object.transform);
            debug_assert!(
                object.mesh.indices.len().is_multiple_of(3),
                "mesh index buffer is not a whole number of triangles"
            );
            for tri in object.mesh.indices.as_chunks::<3>().0 {
                let (Some(v0), Some(v1), Some(v2)) = (
                    object.mesh.vertices.get(tri[0] as usize),
                    object.mesh.vertices.get(tri[1] as usize),
                    object.mesh.vertices.get(tri[2] as usize),
                ) else {
                    debug_assert!(false, "mesh triangle references a missing vertex");
                    continue;
                };
                let p = [
                    model_view_proj.transform_h(v0.pos),
                    model_view_proj.transform_h(v1.pos),
                    model_view_proj.transform_h(v2.pos),
                ];
                let mut poly = PolyZ::new(p);
                poly = clip_z(poly, |q| q[2] + 1.0);
                if poly.len < 3 {
                    continue;
                }
                poly = clip_z(poly, |q| 1.0 - q[2]);
                if poly.len < 3 {
                    continue;
                }
                let verts = poly.as_slice();
                for i in 1..verts.len() - 1 {
                    let a = project_depth(verts[0], size);
                    let b = project_depth(verts[i], size);
                    let c = project_depth(verts[i + 1], size);
                    draw_triangle_depth(&mut self.depth, size, size, a, b, c);
                }
            }
        }
    }

    fn resize(&mut self, settings: ShadowSettings) {
        let size = settings.size.max(16);
        if size != self.size {
            self.size = size;
            self.depth.resize(size * size, f32::INFINITY);
        }
        self.bias = settings.bias.max(0.0);
        self.darkness = settings.darkness.clamp(0.0, 1.0);
        self.pcf_radius = settings.pcf_radius.min(4);
    }

    /// Returns the lighting factor for a world-space point: `1.0` fully lit,
    /// `0.0` fully shadowed. Points outside the light frustum are lit.
    pub fn visibility(&self, world: Vec3) -> f32 {
        if self.size == 0 {
            return 1.0;
        }
        let c = self.view_proj.transform_h(world);
        if c[3].abs() < 1e-9 {
            return 1.0;
        }
        let inv_w = 1.0 / c[3];
        let ndc = [c[0] * inv_w, c[1] * inv_w, c[2] * inv_w];
        if ndc[0] < -1.0 || ndc[0] > 1.0 || ndc[1] < -1.0 || ndc[1] > 1.0 || ndc[2] > 1.0 {
            return 1.0;
        }
        let u = ndc[0] * 0.5 + 0.5;
        let v = 1.0 - (ndc[1] * 0.5 + 0.5);
        let center_x = (u * self.size as f32).floor() as i32;
        let center_y = (v * self.size as f32).floor() as i32;

        let radius = self.pcf_radius as i32;
        let mut shadowed = 0u32;
        let mut total = 0u32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = (center_x + dx).clamp(0, self.size as i32 - 1) as usize;
                let y = (center_y + dy).clamp(0, self.size as i32 - 1) as usize;
                let stored = self.depth[y * self.size + x];
                if stored.is_finite() && ndc[2] > stored + self.bias {
                    shadowed += 1;
                }
                total += 1;
            }
        }
        let coverage = shadowed as f32 / total.max(1) as f32;
        1.0 - self.darkness * coverage
    }
}

fn project_depth(p: [f32; 4], size: usize) -> [f32; 3] {
    let inv_w = if p[3].abs() > 1e-9 { 1.0 / p[3] } else { 1.0 };
    let x = p[0] * inv_w;
    let y = p[1] * inv_w;
    let z = p[2] * inv_w;
    [
        (x * 0.5 + 0.5) * size as f32,
        (1.0 - (y * 0.5 + 0.5)) * size as f32,
        z,
    ]
}

/// Small clip-space polygon for the light frustum's near/far planes.
#[derive(Clone, Copy)]
struct PolyZ {
    verts: [[f32; 4]; 8],
    len: usize,
}

impl PolyZ {
    fn new(tri: [[f32; 4]; 3]) -> Self {
        let mut verts = [[0.0; 4]; 8];
        verts[..3].copy_from_slice(&tri);
        Self { verts, len: 3 }
    }

    fn push(&mut self, v: [f32; 4]) {
        if self.len < self.verts.len() {
            self.verts[self.len] = v;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[[f32; 4]] {
        &self.verts[..self.len]
    }
}

/// Sutherland-Hodgman against the plane where `signed_distance >= 0`.
fn clip_z(poly: PolyZ, signed_distance: impl Fn([f32; 4]) -> f32) -> PolyZ {
    let mut out = PolyZ {
        verts: [[0.0; 4]; 8],
        len: 0,
    };
    let verts = poly.as_slice();
    if verts.is_empty() {
        return out;
    }
    for i in 0..verts.len() {
        let a = verts[i];
        let b = verts[(i + 1) % verts.len()];
        let da = signed_distance(a);
        let db = signed_distance(b);
        if da >= 0.0 {
            out.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            out.push([
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                a[3] + (b[3] - a[3]) * t,
            ]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::mesh::Mesh;
    use crate::scene::Object;

    #[test]
    fn shadow_map_darkens_the_projected_area_and_leaves_far_geometry_lit() {
        let mut scene = Scene::new();
        scene.ground.shadows = false;
        scene.ground.grid = false;
        scene.shadow_map = Some(ShadowSettings::new(256));
        scene.add(Object::new(Mesh::cube(), Material::new([0.9, 0.4, 0.2])));
        let mut map = ShadowMap::new(ShadowSettings::new(256));
        map.render(&scene);

        let dir = scene.light.direction.normalize();
        // Point on the floor directly under the cube along the light ray.
        let t = -1.35 / dir.y;
        let under = Vec3::new(0.0, 0.0, 0.0).add(dir.scale(t));
        assert!(
            map.visibility(under) < 0.6,
            "floor under the cube should be shadowed, got {}",
            map.visibility(under)
        );
        assert!(
            (map.visibility(Vec3::new(20.0, -1.35, 20.0)) - 1.0).abs() < 1e-6,
            "far outside the frustum stays lit"
        );
    }

    #[test]
    fn disabled_shadow_map_leaves_everything_lit() {
        let scene = Scene::new();
        let mut map = ShadowMap::new(ShadowSettings::default());
        map.render(&scene);
        assert!((map.visibility(Vec3::ZERO) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pcf_radius_is_clamped_to_a_sane_maximum() {
        let s = ShadowSettings::default().with_pcf_radius(99);
        assert_eq!(s.pcf_radius, 4);
        assert_eq!(ShadowSettings::new(1).size, 16);
    }
}
