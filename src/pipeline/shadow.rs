//! Planar ground shadows: projection, mask coverage, and compositing.

use crate::draw::triangle::draw_triangle_cover;
use crate::framebuffer::Target;
use crate::math::{Mat4, Vec3, finite_or};
use crate::scene::{Object, Scene};

use super::clip::{Poly, lerp_clip, ndc_to_screen};

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

/// A screen-space box: `(min_x, min_y, max_x, max_y)` in pixels.
pub(super) type ScreenBounds = (f32, f32, f32, f32);

/// Value written into the shadow mask for covered pixels.
const SHADOW_COVER: u32 = 0x00FF_FFFF;

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
            let len = width
                .checked_mul(height)
                .expect("shadow mask size overflows usize");
            self.width = width;
            self.height = height;
            self.mask = vec![0; len];
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

    /// Fills one shadow triangle into the mask. Used by the shadow tests, where
    /// driving the full projection machinery would only add noise.
    #[cfg(test)]
    fn cover_triangle(&mut self, a: [f32; 2], b: [f32; 2], c: [f32; 2]) {
        draw_triangle_cover(
            &mut self.mask,
            self.width,
            0,
            self.height,
            a,
            b,
            c,
            SHADOW_COVER,
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

    /// Mask width in pixels, or `0` before the first [`ShadowMask::prepare`].
    pub fn width(&self) -> usize {
        self.width
    }

    /// Mask height in pixels, or `0` before the first [`ShadowMask::prepare`].
    pub fn height(&self) -> usize {
        self.height
    }

    /// Whether a raw pixel index is shadowed. Out-of-range indices are `false`.
    #[inline]
    pub fn is_shadowed(&self, idx: usize) -> bool {
        self.mask.get(idx).is_some_and(|&value| value != 0)
    }

    /// Whether the pixel at `(x, y)` is shadowed. Out-of-range coordinates are
    /// `false`.
    #[inline]
    pub fn is_shadowed_at(&self, x: usize, y: usize) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.is_shadowed(y * self.width + x)
    }

    /// Covers the whole mask with prepared screen-space triangles.
    pub(super) fn cover_tris(&mut self, tris: &[ShadowTri]) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        for tri in tris {
            draw_triangle_cover(
                &mut self.mask,
                self.width,
                0,
                self.height,
                tri.a,
                tri.b,
                tri.c,
                SHADOW_COVER,
            );
        }
    }

    /// Sets the touched box from projected floating-point bounds.
    pub(super) fn adopt_bounds_f32(&mut self, bounds: Option<ScreenBounds>) {
        match bounds {
            Some((x0, y0, x1, y1)) => self.include_bounds(x0, y0, x1, y1),
            None => {
                self.min_x = usize::MAX;
                self.min_y = usize::MAX;
                self.max_x = 0;
                self.max_y = 0;
            }
        }
    }

    /// Splits the mask into horizontal bands for parallel shadow projection.
    ///
    /// Call [`ShadowMask::prepare`] first. Each band owns a disjoint slice of
    /// the mask, so bands can be covered on separate threads.
    pub(super) fn split_bands(&mut self, count: usize) -> Vec<ShadowBand<'_>> {
        if self.width == 0 || self.height == 0 {
            return Vec::new();
        }
        let count = count.clamp(1, self.height);
        let base = self.height / count;
        let extra = self.height % count;
        let mut mask: &mut [u32] = &mut self.mask[..];
        let mut out = Vec::with_capacity(count);
        let mut y = 0;
        for i in 0..count {
            let rows = base + usize::from(i < extra);
            let len = rows * self.width;
            let (band_mask, rest) = mask.split_at_mut(len);
            out.push(ShadowBand {
                mask: band_mask,
                width: self.width,
                y_offset: y,
                rows,
                min_x: usize::MAX,
                min_y: usize::MAX,
                max_x: 0,
                max_y: 0,
            });
            mask = rest;
            y += rows;
        }
        out
    }
}

/// One horizontal band of a [`ShadowMask`], used by the parallel shadow pass.
pub(super) struct ShadowBand<'a> {
    mask: &'a mut [u32],
    width: usize,
    y_offset: usize,
    rows: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

impl ShadowBand<'_> {
    /// Covers this band with prepared screen-space shadow triangles.
    pub(super) fn cover(&mut self, tris: &[ShadowTri]) {
        let band_start = self.y_offset as f32;
        let band_end = (self.y_offset + self.rows) as f32;
        for tri in tris {
            if tri.max_y < band_start || tri.min_y >= band_end {
                continue;
            }
            draw_triangle_cover(
                self.mask,
                self.width,
                self.y_offset,
                self.rows,
                tri.a,
                tri.b,
                tri.c,
                SHADOW_COVER,
            );
        }
    }

    /// Restricts the band's composite box to the projected shadow bounds.
    pub(super) fn set_bounds(&mut self, bounds: Option<ScreenBounds>) {
        let Some((x0, y0, x1, y1)) = bounds else {
            return;
        };
        let band_end = self.y_offset + self.rows;
        let y0 = (y0.floor().max(0.0) as usize).max(self.y_offset);
        let y1 = (y1.ceil().max(0.0) as usize).min(band_end - 1);
        if y0 > y1 || self.width == 0 {
            return;
        }
        self.min_x = x0.floor().clamp(0.0, self.width as f32 - 1.0) as usize;
        self.max_x = x1.ceil().clamp(0.0, self.width as f32 - 1.0) as usize;
        self.min_y = y0;
        self.max_y = y1;
    }

    /// Darkens the matching slice of the color buffer once.
    pub(super) fn composite_into(&self, color: &mut [u32], tint: [f32; 3]) {
        if self.min_x > self.max_x || self.min_y > self.max_y {
            return;
        }
        let tables = tint_tables(tint);
        let band_end = self.y_offset + self.rows;
        let y0 = self.min_y.max(self.y_offset);
        let y1 = self.max_y.min(band_end - 1);
        if y0 > y1 {
            return;
        }
        for y in y0..=y1 {
            let row = (y - self.y_offset) * self.width;
            let mask = &self.mask[row + self.min_x..=row + self.max_x];
            let pixels = &mut color[row + self.min_x..=row + self.max_x];
            for (covered, pixel) in mask.iter().zip(pixels) {
                if *covered != 0 {
                    *pixel = tint_pixel(*pixel, &tables);
                }
            }
        }
    }
}

/// Builds per-channel 8-bit lookup tables for a shadow tint.
///
/// The composite pass runs over every shadowed pixel. A table lookup replaces
/// an unpack, three float multiplies and a clamp.
fn tint_tables(tint: [f32; 3]) -> [[u8; 256]; 3] {
    let mut tables = [[0u8; 256]; 3];
    for (table, scale) in tables.iter_mut().zip(tint) {
        let scale = finite_or(scale, 0.0).max(0.0);
        for (value, out) in table.iter_mut().enumerate() {
            *out = (value as f32 * scale).clamp(0.0, 255.0) as u8;
        }
    }
    tables
}

#[inline]
fn tint_pixel(color: u32, tables: &[[u8; 256]; 3]) -> u32 {
    let r = u32::from(tables[0][((color >> 16) & 0xFF) as usize]);
    let g = u32::from(tables[1][((color >> 8) & 0xFF) as usize]);
    let b = u32::from(tables[2][(color & 0xFF) as usize]);
    (r << 16) | (g << 8) | b
}

/// Projects a world point onto a horizontal plane along the light direction.
///
/// Returns `None` when the light is parallel to the plane (or the inputs are
/// not finite), because the projection would be at infinity.
pub fn try_project_to_plane(p: Vec3, plane_y: f32, light_dir: Vec3) -> Option<Vec3> {
    if !plane_y.is_finite() || !light_dir.y.is_finite() || light_dir.y.abs() < 1e-4 {
        return None;
    }
    let t = (plane_y - p.y) / light_dir.y;
    let projected = p + light_dir.scale(t);
    (projected.x.is_finite() && projected.y.is_finite() && projected.z.is_finite())
        .then_some(projected)
}

/// Projects a world point onto a horizontal plane along the light direction.
///
/// When the light has no usable downward component, this returns `p` unchanged.
/// Prefer [`try_project_to_plane`] when you need to detect that case.
pub fn project_to_plane(p: Vec3, plane_y: f32, light_dir: Vec3) -> Vec3 {
    try_project_to_plane(p, plane_y, light_dir).unwrap_or(p)
}

/// A ground-shadow triangle already projected to screen space.
///
/// Projecting once per frame instead of once per band is the difference
/// between roughly 2,000 and 16,000 transforms for the demo scene.
#[derive(Clone, Copy)]
pub(super) struct ShadowTri {
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    min_y: f32,
    max_y: f32,
}

/// Renders the hard shadow of one object into the mask.
///
/// The caller must have [`ShadowMask::prepare`]d the mask for the current
/// resolution (and cleared it) first; [`crate::pipeline::render_scene`] does this per frame.
pub fn render_shadow_into(
    mask: &mut ShadowMask,
    object: &Object,
    view: Mat4,
    proj: Mat4,
    near: f32,
    plane_y: f32,
    light_dir: Vec3,
) {
    debug_assert!(
        mask.width > 0 && mask.height > 0,
        "call ShadowMask::prepare before render_shadow_into"
    );
    let (width, height) = (mask.width, mask.height);
    if width == 0 || height == 0 || light_dir.y >= -1e-3 {
        return;
    }
    let view_proj = proj.multiply(&view);
    let mut tris = Vec::new();
    let mut bounds = None;
    prepare_shadow_object(
        object,
        view_proj,
        near,
        width,
        height,
        plane_y,
        light_dir,
        &mut tris,
        &mut bounds,
    );
    mask.cover_tris(&tris);
    mask.adopt_bounds_f32(bounds);
}

/// Projects one object's ground shadow to screen-space triangles.
#[allow(clippy::too_many_arguments)] // a projection plus the buffers it fills
fn prepare_shadow_object(
    object: &Object,
    view_proj: Mat4,
    near: f32,
    width: usize,
    height: usize,
    plane_y: f32,
    light_dir: Vec3,
    out: &mut Vec<ShadowTri>,
    bounds: &mut Option<ScreenBounds>,
) {
    let model = object.transform;

    debug_assert!(
        object.mesh.indices.len().is_multiple_of(3),
        "mesh index buffer is not a whole number of triangles"
    );
    for tri in object.mesh.indices.as_chunks::<3>().0 {
        let project = |i: u32| -> Option<[f32; 4]> {
            let vertex = object.mesh.vertices.get(i as usize)?;
            let world = model.transform_point(vertex.pos);
            if world.y < plane_y - 1e-4 {
                return None; // clearly below the ground: would cast backwards
            }
            let q = try_project_to_plane(world, plane_y, light_dir)?;
            Some(view_proj.transform_h(q))
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
                    width,
                    height,
                ),
                ndc_to_screen(
                    Vec3::new(p[1][0] / p[1][3], p[1][1] / p[1][3], 0.0),
                    width,
                    height,
                ),
                ndc_to_screen(
                    Vec3::new(p[2][0] / p[2][3], p[2][1] / p[2][3], 0.0),
                    width,
                    height,
                ),
            ];
            let x0 = s[0].0.min(s[1].0).min(s[2].0);
            let x1 = s[0].0.max(s[1].0).max(s[2].0);
            let y0 = s[0].1.min(s[1].1).min(s[2].1);
            let y1 = s[0].1.max(s[1].1).max(s[2].1);
            if x1 < 0.0 || x0 >= width as f32 || y1 < 0.0 || y0 >= height as f32 {
                continue; // no visible pixels, so no bounds entry either
            }
            *bounds = Some(match *bounds {
                Some((bx0, by0, bx1, by1)) => (bx0.min(x0), by0.min(y0), bx1.max(x1), by1.max(y1)),
                None => (x0, y0, x1, y1),
            });
            out.push(ShadowTri {
                a: [s[0].0, s[0].1],
                b: [s[1].0, s[1].1],
                c: [s[2].0, s[2].1],
                min_y: y0,
                max_y: y1,
            });
        }
    }
}

/// Projects every shadow caster once per frame.
pub(super) fn prepare_shadows(
    scene: &Scene,
    view_proj: Mat4,
    near: f32,
    width: usize,
    height: usize,
    plane_y: f32,
    light_dir: Vec3,
) -> (Vec<ShadowTri>, Option<ScreenBounds>) {
    if light_dir.y >= -1e-3 {
        return (Vec::new(), None); // light points up/sideways: no ground shadow
    }
    let mut tris = Vec::new();
    let mut bounds = None;
    for object in &scene.objects {
        if !object.casts_shadow {
            continue;
        }
        prepare_shadow_object(
            object,
            view_proj,
            near,
            width,
            height,
            plane_y,
            light_dir,
            &mut tris,
            &mut bounds,
        );
    }
    (tris, bounds)
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
    let Some(screen_len) = target.width.checked_mul(target.height) else {
        return;
    };
    if mask.width != target.width || mask.height != target.height || mask.mask.len() < screen_len {
        return;
    }
    let Some((x0, y0, x1, y1)) = mask.bounds() else {
        return;
    };
    // The target may be one band of the screen; only touch its own rows.
    let y0 = y0.max(target.band_offset());
    let y1 = y1.min(target.band_end() - 1);
    if y0 > y1 {
        return;
    }
    let tables = tint_tables(tint);
    for y in y0..=y1 {
        let mask_row = y * target.width;
        let out_row = (y - target.band_offset()) * target.width;
        let covered = &mask.mask[mask_row + x0..=mask_row + x1];
        let pixels = &mut target.color[out_row + x0..=out_row + x1];
        for (is_set, pixel) in covered.iter().zip(pixels) {
            if *is_set != 0 {
                *pixel = tint_pixel(*pixel, &tables);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framebuffer::Framebuffer;
    use crate::math::Vec3;

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
    fn horizontal_light_has_no_plane_projection() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let side = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(try_project_to_plane(p, -1.0, side), None);
        assert_eq!(project_to_plane(p, -1.0, side), p);
    }

    #[test]
    fn shadow_mask_accessors_are_bounds_checked() {
        let mut mask = ShadowMask::new();
        mask.prepare(16, 16);
        mask.cover_triangle([0.0, 0.0], [15.0, 0.0], [0.0, 15.0]);
        mask.include_bounds(0.0, 0.0, 15.0, 15.0);
        assert_eq!((mask.width(), mask.height()), (16, 16));
        assert!(mask.is_shadowed_at(2, 2));
        assert!(!mask.is_shadowed_at(100, 100));
        assert!(!mask.is_shadowed(usize::MAX));
    }

    #[test]
    fn mismatched_shadow_masks_are_ignored() {
        let mut fb = Framebuffer::new(16, 16);
        fb.clear(0x808080);
        let before = fb.color.clone();
        let mut mask = ShadowMask::new();
        mask.prepare(8, 8);
        mask.cover_triangle([0.0, 0.0], [7.0, 0.0], [0.0, 7.0]);
        mask.include_bounds(0.0, 0.0, 7.0, 7.0);
        {
            let mut target = fb.target();
            composite_shadow(&mut target, &mask, [0.5, 0.5, 0.5]);
        }
        assert_eq!(fb.color, before);
    }
}
