//! Scene pipeline: transforms, clipping, lighting, grid, planar shadow masks,
//! and draw dispatch. Resolution independent, since everything works off a
//! `Target`.
//! Made by AeroForger

use crate::camera::Camera;
use crate::draw::triangle::draw_triangle_cover;
use crate::draw::{ShadedVertex, draw_line_z, draw_triangle_gouraud, lerp, rgb};
use crate::framebuffer::Target;
use crate::light::DirectionalLight;
use crate::material::Material;
use crate::math::{Mat4, Vec3};
use crate::scene::{DrawMode, Object, Scene};

/// Per-frame lighting state shared by every object.
pub struct RenderContext {
    /// World-space camera position, for specular highlights.
    pub camera_pos: Vec3,
    /// Normalized direction the light travels (toward the scene).
    pub light_dir: Vec3,
    /// Light color, linear RGB.
    pub light_color: [f32; 3],
    /// Ambient fill.
    pub ambient: f32,
}

impl RenderContext {
    /// Builds the context for a frame from a camera position and a light.
    pub fn new(camera_pos: Vec3, light: &DirectionalLight) -> Self {
        Self {
            camera_pos,
            light_dir: light.direction.normalize(),
            light_color: light.color,
            ambient: light.ambient,
        }
    }
}

/// Blinn-Phong vertex lighting: ambient + diffuse + specular + rim.
///
/// Colors are linear and may exceed `1.0`; the rasterizer clamps when packing.
pub fn shade_vertex(pos: Vec3, normal: Vec3, material: &Material, ctx: &RenderContext) -> [f32; 3] {
    let to_light = -ctx.light_dir;
    let to_cam = (ctx.camera_pos - pos).normalize();
    let half = (to_light + to_cam).normalize();

    let ndotl = normal.dot(to_light).max(0.0);
    let spec = if ndotl > 0.0 {
        normal.dot(half).max(0.0).powf(material.shininess.max(1.0))
    } else {
        0.0
    };
    let rim = (1.0 - normal.dot(to_cam).max(0.0)).powi(2) * material.rim;
    let ambient = ctx.ambient;

    [
        material.color[0] * (ambient + (1.0 - ambient) * ndotl) * ctx.light_color[0]
            + spec * material.specular * ctx.light_color[0]
            + rim * material.color[0],
        material.color[1] * (ambient + (1.0 - ambient) * ndotl) * ctx.light_color[1]
            + spec * material.specular * ctx.light_color[1]
            + rim * material.color[1],
        material.color[2] * (ambient + (1.0 - ambient) * ndotl) * ctx.light_color[2]
            + spec * material.specular * ctx.light_color[2]
            + rim * material.color[2],
    ]
}

/// One vertex in clip space, carrying its precomputed vertex color.
#[derive(Clone, Copy)]
struct ClipVertex {
    pos: [f32; 4], // x, y, z, w, with no perspective divide yet
    color: [f32; 3],
}

/// Fixed-capacity polygon from clipping a triangle against one plane: at most
/// four vertices. Keeps the per-triangle path allocation-free.
#[derive(Clone, Copy)]
struct Poly<V: Copy> {
    verts: [V; 4],
    len: usize,
}

impl<V: Copy> Poly<V> {
    /// An empty polygon seeded with `fill` so the array is initialized without
    /// requiring `V: Default`.
    fn new(fill: V) -> Self {
        Self {
            verts: [fill; 4],
            len: 0,
        }
    }

    fn push(&mut self, v: V) {
        debug_assert!(
            self.len < 4,
            "a triangle clipped by one plane has <= 4 verts"
        );
        self.verts[self.len] = v;
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[V] {
        &self.verts[..self.len]
    }
}

impl ClipVertex {
    fn ndc(&self) -> Vec3 {
        let inv_w = 1.0 / self.pos[3];
        Vec3::new(
            self.pos[0] * inv_w,
            self.pos[1] * inv_w,
            self.pos[2] * inv_w,
        )
    }

    fn shaded(self, ndc: Vec3, width: usize, height: usize) -> ShadedVertex {
        ShadedVertex {
            x: (ndc.x * 0.5 + 0.5) * width as f32,
            y: (1.0 - (ndc.y * 0.5 + 0.5)) * height as f32,
            z: ndc.z,
            color: self.color,
        }
    }

    fn lerp(a: ClipVertex, b: ClipVertex, t: f32) -> ClipVertex {
        ClipVertex {
            pos: [
                lerp(a.pos[0], b.pos[0], t),
                lerp(a.pos[1], b.pos[1], t),
                lerp(a.pos[2], b.pos[2], t),
                lerp(a.pos[3], b.pos[3], t),
            ],
            color: [
                lerp(a.color[0], b.color[0], t),
                lerp(a.color[1], b.color[1], t),
                lerp(a.color[2], b.color[2], t),
            ],
        }
    }
}

/// Sutherland-Hodgman clip of one triangle against the near plane
/// (`w >= near`). Vertices behind the camera would otherwise divide by a
/// negative `w` and produce garbage triangles wrapping the screen.
fn clip_near(tri: [ClipVertex; 3], near: f32) -> Poly<ClipVertex> {
    let mut out = Poly::new(tri[0]);
    for i in 0..3 {
        let (a, b) = (tri[i], tri[(i + 1) % 3]);
        let da = a.pos[3] - near;
        let db = b.pos[3] - near;
        if da >= 0.0 {
            out.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            out.push(ClipVertex::lerp(a, b, t));
        }
    }
    out
}

fn ndc_to_screen(ndc: Vec3, width: usize, height: usize) -> (f32, f32) {
    (
        (ndc.x * 0.5 + 0.5) * width as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * height as f32,
    )
}

/// Renders one meshed object, honoring its [`DrawMode`].
pub fn render_object(
    target: &mut Target,
    object: &Object,
    view: Mat4,
    proj: Mat4,
    near: f32,
    ctx: &RenderContext,
) {
    let model = object.transform;
    let mv = view.multiply(&model);
    let mvp = proj.multiply(&mv);
    let base = object.material.color;
    let wire_col = rgb(
        base[0] * 0.3 + 0.5,
        base[1] * 0.3 + 0.5,
        base[2] * 0.3 + 0.5,
    );

    let shaded = matches!(object.mode, DrawMode::Shaded | DrawMode::ShadedWireframe);
    let wireframe = matches!(object.mode, DrawMode::Wireframe | DrawMode::ShadedWireframe);

    for tri in object.mesh.indices.chunks(3) {
        let clip_vert = |i: u32| -> ClipVertex {
            let v = &object.mesh.vertices[i as usize];
            let world = model.transform_point(v.pos);
            let world_n = model.transform_dir(v.normal).normalize();
            ClipVertex {
                pos: mvp.transform_h(v.pos),
                color: shade_vertex(world, world_n, &object.material, ctx),
            }
        };

        let clipped = clip_near(
            [clip_vert(tri[0]), clip_vert(tri[1]), clip_vert(tri[2])],
            near,
        );
        if clipped.len() < 3 {
            continue;
        }

        // Fan the clipped polygon (3 or 4 verts) into triangles.
        let verts = clipped.as_slice();
        for i in 1..verts.len() - 1 {
            let v = [verts[0], verts[i], verts[i + 1]];
            let n: [Vec3; 3] = [v[0].ndc(), v[1].ndc(), v[2].ndc()];
            let winding =
                (n[1].x - n[0].x) * (n[2].y - n[0].y) - (n[2].x - n[0].x) * (n[1].y - n[0].y);

            if shaded && winding > 0.0 {
                draw_triangle_gouraud(
                    target,
                    v[0].shaded(n[0], target.width, target.height),
                    v[1].shaded(n[1], target.width, target.height),
                    v[2].shaded(n[2], target.width, target.height),
                );
            }

            if wireframe {
                let s = [
                    ndc_to_screen(n[0], target.width, target.height),
                    ndc_to_screen(n[1], target.width, target.height),
                    ndc_to_screen(n[2], target.width, target.height),
                ];
                for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                    draw_line_z(
                        target, s[a].0, s[a].1, n[a].z, s[b].0, s[b].1, n[b].z, wire_col,
                    );
                }
            }
        }
    }
}

/// Depth-tested ground grid: minor lines every 1 unit, major every 5.
pub fn draw_grid(target: &mut Target, view: Mat4, proj: Mat4, near: f32, y: f32, extent: i32) {
    let minor = rgb(0.13, 0.19, 0.25);
    let major = rgb(0.20, 0.30, 0.38);
    let mvp = proj.multiply(&view);

    let mut segment = |a: Vec3, b: Vec3, color: u32| {
        let ca = mvp.transform_h(a);
        let cb = mvp.transform_h(b);
        // Clip the segment against the near plane before dividing.
        let (da, db) = (ca[3] - near, cb[3] - near);
        if da < 0.0 && db < 0.0 {
            return;
        }
        let (ca, cb) = if da < 0.0 {
            (lerp_clip(ca, cb, da / (da - db)), cb)
        } else if db < 0.0 {
            (ca, lerp_clip(cb, ca, db / (db - da)))
        } else {
            (ca, cb)
        };
        let za = ca[2] / ca[3];
        let zb = cb[2] / cb[3];
        let sa = ndc_to_screen(
            Vec3::new(ca[0] / ca[3], ca[1] / ca[3], za),
            target.width,
            target.height,
        );
        let sb = ndc_to_screen(
            Vec3::new(cb[0] / cb[3], cb[1] / cb[3], zb),
            target.width,
            target.height,
        );
        draw_line_z(target, sa.0, sa.1, za, sb.0, sb.1, zb, color);
    };

    for i in -extent..=extent {
        let t = i as f32;
        let c = if i % 5 == 0 { major } else { minor };
        segment(
            Vec3::new(t, y, -extent as f32),
            Vec3::new(t, y, extent as f32),
            c,
        );
        segment(
            Vec3::new(-extent as f32, y, t),
            Vec3::new(extent as f32, y, t),
            c,
        );
    }
}

fn lerp_clip(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    ]
}

/// Offscreen mask used to compose planar shadows exactly once per pixel.
///
/// Rendering shadow geometry into a mask (instead of straight into the frame)
/// is what prevents overlapping shadow triangles from darkening the same pixel
/// repeatedly, which is a common source of blotchy shadows.
#[derive(Default)]
pub struct ShadowMask {
    width: usize,
    height: usize,
    /// Non-zero where a shadow triangle covered the pixel.
    mask: Vec<u32>,
    /// Bounding box of pixels touched this frame, so compositing can skip the
    /// rest of the image.
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

impl ShadowMask {
    /// An empty mask; it sizes itself on first use.
    pub fn new() -> Self {
        Self {
            min_x: usize::MAX,
            min_y: usize::MAX,
            ..Self::default()
        }
    }

    /// Resizes (if needed) and clears the mask.
    pub fn prepare(&mut self, width: usize, height: usize) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.mask = vec![0; width * height];
        } else if self.min_x <= self.max_x && self.min_y <= self.max_y {
            // Only the area touched last frame can be non-zero, so clearing
            // that box is enough. Shadows usually cover a small part of the
            // screen, which makes this much cheaper than a full clear at
            // supersampled resolutions.
            for y in self.min_y..=self.max_y {
                let row = y * self.width;
                self.mask[row + self.min_x..=row + self.max_x].fill(0);
            }
        }
        self.min_x = usize::MAX;
        self.min_y = usize::MAX;
        self.max_x = 0;
        self.max_y = 0;
    }

    /// Fills one shadow triangle into the mask. There is no depth test because
    /// the mask only cares whether a pixel is covered at all.
    fn cover_triangle(&mut self, a: [f32; 2], b: [f32; 2], c: [f32; 2]) {
        draw_triangle_cover(
            &mut self.mask,
            self.width,
            self.height,
            a,
            b,
            c,
            0x00FF_FFFF,
        );
    }

    /// Expands the touched-pixel bounds to include a projected triangle.
    fn include_bounds(&mut self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let x0 = min_x.floor().clamp(0.0, self.width as f32 - 1.0) as usize;
        let y0 = min_y.floor().clamp(0.0, self.height as f32 - 1.0) as usize;
        let x1 = max_x.ceil().clamp(0.0, self.width as f32 - 1.0) as usize;
        let y1 = max_y.ceil().clamp(0.0, self.height as f32 - 1.0) as usize;
        self.min_x = self.min_x.min(x0);
        self.min_y = self.min_y.min(y0);
        self.max_x = self.max_x.max(x1);
        self.max_y = self.max_y.max(y1);
    }

    /// The touched-pixel box, or `None` when nothing was drawn.
    fn bounds(&self) -> Option<(usize, usize, usize, usize)> {
        if self.min_x > self.max_x || self.min_y > self.max_y {
            None
        } else {
            Some((self.min_x, self.min_y, self.max_x, self.max_y))
        }
    }

    /// Whether a pixel index is shadowed.
    #[inline]
    pub fn is_shadowed(&self, idx: usize) -> bool {
        self.mask[idx] != 0
    }
}

/// Projects a world point onto a horizontal plane along the light direction.
pub fn project_to_plane(p: Vec3, plane_y: f32, light_dir: Vec3) -> Vec3 {
    let t = (plane_y - p.y) / light_dir.y;
    p + light_dir.scale(t)
}

/// Renders the hard shadow of one object into the mask.
///
/// The caller must have [`ShadowMask::prepare`]d the mask for the current
/// resolution (and cleared it) first; [`render_scene`] does this per frame.
pub fn render_shadow_into(
    mask: &mut ShadowMask,
    object: &Object,
    view: Mat4,
    proj: Mat4,
    near: f32,
    plane_y: f32,
    light_dir: Vec3,
) {
    if light_dir.y >= -1e-3 {
        return; // light points up/sideways: no ground shadow
    }
    let mvp = proj.multiply(&view);
    let model = object.transform;
    let mut bounds: Option<(f32, f32, f32, f32)> = None;

    for tri in object.mesh.indices.chunks(3) {
        let project = |i: u32| -> Option<[f32; 4]> {
            let world = model.transform_point(object.mesh.vertices[i as usize].pos);
            if world.y < plane_y - 1e-4 {
                return None; // clearly below the ground: would cast backwards
            }
            let q = project_to_plane(world, plane_y, light_dir);
            Some(mvp.transform_h(q))
        };
        let (Some(a), Some(b), Some(c)) = (project(tri[0]), project(tri[1]), project(tri[2]))
        else {
            continue;
        };
        let clipped = clip_near_proj([a, b, c], near);
        if clipped.len() < 3 {
            continue;
        }
        let verts = clipped.as_slice();
        for i in 1..verts.len() - 1 {
            let p = [verts[0], verts[i], verts[i + 1]];
            let s = [
                ndc_to_screen(
                    Vec3::new(p[0][0] / p[0][3], p[0][1] / p[0][3], 0.0),
                    mask.width,
                    mask.height,
                ),
                ndc_to_screen(
                    Vec3::new(p[1][0] / p[1][3], p[1][1] / p[1][3], 0.0),
                    mask.width,
                    mask.height,
                ),
                ndc_to_screen(
                    Vec3::new(p[2][0] / p[2][3], p[2][1] / p[2][3], 0.0),
                    mask.width,
                    mask.height,
                ),
            ];
            let x0 = s[0].0.min(s[1].0).min(s[2].0);
            let x1 = s[0].0.max(s[1].0).max(s[2].0);
            let y0 = s[0].1.min(s[1].1).min(s[2].1);
            let y1 = s[0].1.max(s[1].1).max(s[2].1);
            bounds = Some(match bounds {
                Some((bx0, by0, bx1, by1)) => (bx0.min(x0), by0.min(y0), bx1.max(x1), by1.max(y1)),
                None => (x0, y0, x1, y1),
            });
            mask.cover_triangle([s[0].0, s[0].1], [s[1].0, s[1].1], [s[2].0, s[2].1]);
        }
    }
    if let Some((x0, y0, x1, y1)) = bounds {
        mask.include_bounds(x0, y0, x1, y1);
    }
}

/// Near-plane clip for raw clip-space polygons (shadow geometry).
fn clip_near_proj(pts: [[f32; 4]; 3], near: f32) -> Poly<[f32; 4]> {
    let mut out = Poly::new(pts[0]);
    for i in 0..3 {
        let (a, b) = (pts[i], pts[(i + 1) % 3]);
        let da = a[3] - near;
        let db = b[3] - near;
        if da >= 0.0 {
            out.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            out.push(lerp_clip(a, b, t));
        }
    }
    out
}

/// Darkens the frame buffer wherever the shadow mask says "shadowed".
///
/// Runs once per pixel, so overlapping shadow geometry cannot stack up.
pub fn composite_shadow(target: &mut Target, mask: &ShadowMask, tint: [f32; 3]) {
    debug_assert_eq!(mask.mask.len(), target.color.len());
    let Some((x0, y0, x1, y1)) = mask.bounds() else {
        return;
    };
    for y in y0..=y1 {
        let row = y * target.width;
        for x in x0..=x1 {
            let i = row + x;
            if mask.is_shadowed(i) {
                let [r, g, b] = crate::draw::pixel::unpack(target.color[i]);
                target.color[i] = rgb(r * tint[0], g * tint[1], b * tint[2]);
            }
        }
    }
}

/// Renders an entire scene: ground grid, planar shadows, then objects.
pub fn render_scene(target: &mut Target, scene: &Scene, camera: &Camera, shadow: &mut ShadowMask) {
    let view = camera.view();
    let proj = camera.projection(target.width as f32 / target.height as f32);
    let near = camera.near;
    let ctx = RenderContext::new(camera.eye(), &scene.light);

    if scene.ground.grid {
        draw_grid(target, view, proj, near, scene.ground.y, 10);
    }

    if scene.ground.shadows {
        shadow.prepare(target.width, target.height);
        for object in scene.objects.iter().filter(|o| o.casts_shadow) {
            render_shadow_into(
                shadow,
                object,
                view,
                proj,
                near,
                scene.ground.y,
                ctx.light_dir,
            );
        }
        composite_shadow(target, shadow, scene.ground.shadow_tint);
    }

    for object in &scene.objects {
        render_object(target, object, view, proj, near, &ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framebuffer::Framebuffer;
    use crate::scene::Ground;

    fn cv(w: f32) -> ClipVertex {
        ClipVertex {
            pos: [0.0, 0.0, 0.0, w],
            color: [0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn clip_near_splits_straddling_triangle() {
        let out = clip_near([cv(5.0), cv(2.0), cv(6.0)], 4.0);
        assert_eq!(out.len(), 4, "one vertex behind -> quad");
        assert!(out.as_slice().iter().all(|v| v.pos[3] >= 4.0 - 1e-4));
    }

    #[test]
    fn clip_near_drops_fully_behind_triangle() {
        assert_eq!(clip_near([cv(1.0), cv(2.0), cv(3.0)], 4.0).len(), 0);
    }

    #[test]
    fn clip_near_keeps_fully_in_front_triangle() {
        assert_eq!(clip_near([cv(5.0), cv(6.0), cv(7.0)], 4.0).len(), 3);
    }

    #[test]
    fn shadow_projection_lands_on_the_plane() {
        let light = Vec3::new(-0.45, -1.0, -0.35).normalize();
        let q = project_to_plane(Vec3::new(2.0, 3.0, -1.0), -1.5, light);
        assert!((q.y - (-1.5)).abs() < 1e-4);
        assert!(q.x < 2.0, "light leans -X, shadow must shift -X");
    }

    #[test]
    fn shadows_do_not_stack_when_triangles_overlap() {
        // Two overlapping shadow quads must darken a pixel exactly as much as
        // one. The mask guarantees a single composite.
        let mut fb = Framebuffer::new(32, 32);
        fb.clear(0x808080);
        let mut mask = ShadowMask::new();
        mask.prepare(32, 32);
        mask.cover_triangle([0.0, 0.0], [31.0, 0.0], [0.0, 31.0]);
        mask.cover_triangle([0.0, 0.0], [31.0, 0.0], [0.0, 31.0]);
        mask.include_bounds(0.0, 0.0, 31.0, 31.0);
        {
            let mut target = fb.target();
            composite_shadow(&mut target, &mask, [0.5, 0.5, 0.5]);
        }
        let [r, _, _] = crate::draw::pixel::unpack(fb.color[5 * 32 + 5]);
        assert!(
            (r - 0.25).abs() < 0.02,
            "one composite darkening, got {r} (double-composite would be 0.125)"
        );
    }

    #[test]
    fn prepare_clears_the_previous_frame_bounds() {
        // The mask only clears the box it touched last frame, so make sure a
        // second prepare leaves no stale coverage behind.
        let mut mask = ShadowMask::new();
        mask.prepare(16, 16);
        mask.cover_triangle([2.0, 2.0], [8.0, 2.0], [2.0, 8.0]);
        mask.include_bounds(2.0, 2.0, 8.0, 8.0);
        assert!(mask.bounds().is_some());
        mask.prepare(16, 16);
        assert!(mask.mask.iter().all(|&p| p == 0), "stale shadow pixels");
        assert!(mask.bounds().is_none(), "bounds should reset each frame");
    }

    #[test]
    fn blend_phong_lit_side_is_brighter() {
        let ctx = RenderContext::new(Vec3::new(0.0, 5.0, 8.0), &DirectionalLight::default());
        let up = -ctx.light_dir;
        let m = Material::new([1.0, 0.3, 0.2]);
        let lit = shade_vertex(Vec3::ZERO, up, &m, &ctx);
        let dark = shade_vertex(Vec3::ZERO, -up, &m, &ctx);
        assert!(lit[0] > dark[0] * 2.0, "lit {lit:?} vs dark {dark:?}");
    }

    #[test]
    fn shadow_tint_defaults_come_from_ground() {
        let g = Ground::default();
        assert!(g.shadows, "shadows on by default");
        assert!(g.grid, "grid on by default");
    }
}
