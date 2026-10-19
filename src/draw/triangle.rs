//! Triangle rasterization: barycentric fill with z-buffer and shading modes.
//! Made by AeroForger

use crate::draw::pixel::{edge, rgb};
use crate::framebuffer::Target;

/// A projected vertex for Gouraud shading: screen position, NDC depth and a
/// precomputed linear RGB color (lit per-vertex in the pipeline).
#[derive(Clone, Copy, Debug)]
pub struct ShadedVertex {
    /// Screen x in pixels.
    pub x: f32,
    /// Screen y in pixels.
    pub y: f32,
    /// NDC depth (near = -1, far = +1).
    pub z: f32,
    /// Linear RGB, may exceed 1.0 (clamped when packed).
    pub color: [f32; 3],
}

/// Gouraud-shaded, z-buffered triangle: interpolates per-vertex colors and
/// depth across the face.
///
/// Depth and color are linear in screen space, so they are stepped along each
/// scanline. Recomputing them from three barycentric weights per pixel is the
/// obvious implementation, but it costs nine extra multiplies per pixel.
pub fn draw_triangle_gouraud(
    target: &mut Target,
    a: ShadedVertex,
    b: ShadedVertex,
    c: ShadedVertex,
) {
    // Clamp the bounding box to the screen and then to this band. A vertex
    // clipped at the near plane can project millions of pixels off-screen.
    let band_start = target.band_offset() as i32;
    let band_end = target.band_end() as i32 - 1;
    let min_x = (a.x.min(b.x).min(c.x).floor() as i32).max(0);
    let max_x = (a.x.max(b.x).max(c.x).ceil() as i32).min(target.width as i32 - 1);
    // Step from the triangle's own top for the same rounding sequence in every
    // band, so banded output matches a single-threaded render exactly.
    let bbox_min_y = (a.y.min(b.y).min(c.y).floor() as i32).max(0);
    let bbox_max_y = (a.y.max(b.y).max(c.y).ceil() as i32).min(target.height as i32 - 1);
    let min_y = bbox_min_y.max(band_start);
    let max_y = bbox_max_y.min(band_end);
    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
    if area.abs() < f32::EPSILON {
        return;
    }
    let inv_area = 1.0 / area;

    let px = min_x as f32 + 0.5;
    let py = bbox_min_y as f32 + 0.5;
    let dw0_dx = (b.y - c.y) * inv_area;
    let dw0_dy = (c.x - b.x) * inv_area;
    let dw1_dx = (c.y - a.y) * inv_area;
    let dw1_dy = (a.x - c.x) * inv_area;
    let dw2_dx = (a.y - b.y) * inv_area;
    let dw2_dy = (b.x - a.x) * inv_area;
    let mut w0_row = edge(b.x, b.y, c.x, c.y, px, py) * inv_area;
    let mut w1_row = edge(c.x, c.y, a.x, a.y, px, py) * inv_area;
    let mut w2_row = edge(a.x, a.y, b.x, b.y, px, py) * inv_area;

    let dz_dx = dw0_dx * a.z + dw1_dx * b.z + dw2_dx * c.z;
    let dr_dx = dw0_dx * a.color[0] + dw1_dx * b.color[0] + dw2_dx * c.color[0];
    let dg_dx = dw0_dx * a.color[1] + dw1_dx * b.color[1] + dw2_dx * c.color[1];
    let db_dx = dw0_dx * a.color[2] + dw1_dx * b.color[2] + dw2_dx * c.color[2];

    let width = target.width;
    let color = &mut target.color[..];
    let depth = &mut target.depth[..];

    for y in bbox_min_y..=bbox_max_y {
        if !(band_start..=band_end).contains(&y) {
            w0_row += dw0_dy;
            w1_row += dw1_dy;
            w2_row += dw2_dy;
            continue;
        }
        let Some((lo, hi)) = row_span(
            [w0_row, w1_row, w2_row],
            [dw0_dx, dw1_dx, dw2_dx],
            min_x,
            max_x,
        ) else {
            w0_row += dw0_dy;
            w1_row += dw1_dy;
            w2_row += dw2_dy;
            continue;
        };

        let step = (lo - min_x) as f32;
        let w0 = w0_row + dw0_dx * step;
        let w1 = w1_row + dw1_dx * step;
        let w2 = w2_row + dw2_dx * step;
        let mut z = w0 * a.z + w1 * b.z + w2 * c.z;
        let mut r = w0 * a.color[0] + w1 * b.color[0] + w2 * c.color[0];
        let mut g = w0 * a.color[1] + w1 * b.color[1] + w2 * c.color[1];
        let mut bl = w0 * a.color[2] + w1 * b.color[2] + w2 * c.color[2];
        let row = (y - band_start) as usize * width;
        for x in lo..=hi {
            let idx = row + x as usize;
            // Slightly conservative epsilon avoids z-fighting on shared edges.
            if z < depth[idx] + 1e-6 {
                depth[idx] = z;
                color[idx] = rgb(r, g, bl);
            }
            z += dz_dx;
            r += dr_dx;
            g += dg_dx;
            bl += db_dx;
        }
        w0_row += dw0_dy;
        w1_row += dw1_dy;
        w2_row += dw2_dy;
    }
}

/// Flat, single-color z-buffered triangle.
///
/// Each point is `[x, y, z]`: screen pixels plus NDC depth (near = -1,
/// far = +1; smaller wins).
pub fn draw_triangle_z(target: &mut Target, a: [f32; 3], b: [f32; 3], c: [f32; 3], color: u32) {
    rasterize(target, a, b, c, |_, _, _, _| color);
}

/// Coverage-only triangle: writes `color` wherever the triangle covers a pixel,
/// with no depth test and no shading.
///
/// This is for binary masks, such as the planar shadow mask, where overlapping
/// triangles are all the same value. Skipping the depth buffer saves a read, a
/// compare, and a write per pixel, and the caller does not have to clear or
/// allocate a depth buffer at all.
#[allow(clippy::too_many_arguments)] // band geometry plus the triangle
pub(crate) fn draw_triangle_cover(
    color: &mut [u32],
    width: usize,
    y_offset: usize,
    rows: usize,
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    value: u32,
) {
    let band_start = y_offset as i32;
    let band_end = (y_offset + rows) as i32 - 1;
    let min_x = (a[0].min(b[0]).min(c[0]).floor() as i32).max(0);
    let max_x = (a[0].max(b[0]).max(c[0]).ceil() as i32).min(width as i32 - 1);
    // Step from the triangle's own top for every band, so all bands follow the
    // same rounding sequence and coverage matches a full-screen pass exactly.
    let bbox_min_y = (a[1].min(b[1]).min(c[1]).floor() as i32).max(0);
    let bbox_max_y = (a[1].max(b[1]).max(c[1]).ceil() as i32).min(band_end);
    let min_y = bbox_min_y.max(band_start);
    if min_x > max_x || min_y > bbox_max_y {
        return;
    }

    let area = edge(a[0], a[1], b[0], b[1], c[0], c[1]);
    if area.abs() < f32::EPSILON {
        return;
    }
    let inv_area = 1.0 / area;

    let px = min_x as f32 + 0.5;
    let py = bbox_min_y as f32 + 0.5;
    let dw0_dx = (b[1] - c[1]) * inv_area;
    let dw0_dy = (c[0] - b[0]) * inv_area;
    let dw1_dx = (c[1] - a[1]) * inv_area;
    let dw1_dy = (a[0] - c[0]) * inv_area;
    let dw2_dx = (a[1] - b[1]) * inv_area;
    let dw2_dy = (b[0] - a[0]) * inv_area;
    let mut w0_row = edge(b[0], b[1], c[0], c[1], px, py) * inv_area;
    let mut w1_row = edge(c[0], c[1], a[0], a[1], px, py) * inv_area;
    let mut w2_row = edge(a[0], a[1], b[0], b[1], px, py) * inv_area;

    for y in bbox_min_y..=bbox_max_y {
        // The inside region of a triangle on one scanline is a single interval.
        // Solve each edge for the x range where its weight is >= -epsilon, then
        // fill the whole span in one go. This skips the per-pixel edge tests
        // and stores, which is most of the cost for large shadow triangles.
        if y >= band_start
            && let Some((lo, hi)) = row_span(
                [w0_row, w1_row, w2_row],
                [dw0_dx, dw1_dx, dw2_dx],
                min_x,
                max_x,
            )
        {
            let row = (y - band_start) as usize * width;
            let start = row + lo as usize;
            let end = row + hi as usize;
            color[start..=end].fill(value);
        }
        w0_row += dw0_dy;
        w1_row += dw1_dy;
        w2_row += dw2_dy;
    }
}

/// Multiplicative shadow triangle: darkens whatever is already in the frame
/// buffer instead of painting opaque black, which reads as a soft shadow and
/// lets the ground grid show through.
///
/// Points are `[x, y, z]` like [`draw_triangle_z`]. Note that blending straight
/// into the frame means overlapping triangles darken repeatedly; for shadow
/// volumes use the mask pass in [`crate::pipeline`] instead.
pub fn draw_triangle_shadow(
    target: &mut Target,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    tint: [f32; 3],
) {
    rasterize(target, a, b, c, |existing, _, _, _| {
        let [r, g, b] = crate::draw::pixel::unpack(existing);
        rgb(r * tint[0], g * tint[1], b * tint[2])
    });
}

/// Finds the inclusive x range on one scanline where all three barycentric
/// weights are inside the triangle. The weights are linear in x, so each edge
/// gives a lower or upper bound and the inside region is their intersection.
#[inline]
fn row_span(w: [f32; 3], dw: [f32; 3], min_x: i32, max_x: i32) -> Option<(i32, i32)> {
    const EPS: f32 = -1e-4;
    let mut lo = min_x;
    let mut hi = max_x;
    for (w, dw) in w.into_iter().zip(dw) {
        if dw > 0.0 {
            let x = min_x as f32 + (EPS - w) / dw;
            lo = lo.max(x.ceil() as i32);
        } else if dw < 0.0 {
            let x = min_x as f32 + (EPS - w) / dw;
            hi = hi.min(x.floor() as i32);
        } else if w < EPS {
            return None;
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// Shared barycentric/z-test core. `shade` receives the pixel already in the
/// frame buffer so modes can replace it (flat/Gouraud) or modulate it (shadow).
///
/// The edge functions and barycentric weights are stepped incrementally across
/// each scanline instead of being recomputed with three cross products per
/// pixel. That matters a lot at supersampled resolutions, where the same
/// triangle covers 4x or 9x as many pixels.
fn rasterize<F>(target: &mut Target, a: [f32; 3], b: [f32; 3], c: [f32; 3], shade: F)
where
    F: Fn(u32, f32, f32, f32) -> u32,
{
    // Clamp the bounding box to the screen and then to this band. A vertex
    // clipped at the near plane can project millions of pixels off-screen;
    // without this the per-pixel loop would walk that whole range.
    let band_start = target.band_offset() as i32;
    let band_end = target.band_end() as i32 - 1;
    let min_x = (a[0].min(b[0]).min(c[0]).floor() as i32).max(0);
    let max_x = (a[0].max(b[0]).max(c[0]).ceil() as i32).min(target.width as i32 - 1);
    // Step weights from the triangle's own top, not from the band's top. Every
    // band then walks the same sequence of rounding steps, so a banded render
    // is pixel-identical to a single-threaded one. Rows outside the band are
    // skipped but still stepped, which is a few adds per row.
    let bbox_min_y = (a[1].min(b[1]).min(c[1]).floor() as i32).max(0);
    let bbox_max_y = (a[1].max(b[1]).max(c[1]).ceil() as i32).min(target.height as i32 - 1);
    let min_y = bbox_min_y.max(band_start);
    let max_y = bbox_max_y.min(band_end);
    if min_x > max_x || min_y > max_y {
        return; // completely outside this band
    }

    // Twice the signed area. This is also the barycentric denominator.
    let area = edge(a[0], a[1], b[0], b[1], c[0], c[1]);
    if area.abs() < f32::EPSILON {
        return; // degenerate
    }
    let inv_area = 1.0 / area;

    // Edge functions at the first pixel center, plus their per-step deltas.
    // w0 is the weight of a, w1 of b, w2 of c.
    let px = min_x as f32 + 0.5;
    let py = bbox_min_y as f32 + 0.5;
    let dw0_dx = (b[1] - c[1]) * inv_area;
    let dw0_dy = (c[0] - b[0]) * inv_area;
    let dw1_dx = (c[1] - a[1]) * inv_area;
    let dw1_dy = (a[0] - c[0]) * inv_area;
    let dw2_dx = (a[1] - b[1]) * inv_area;
    let dw2_dy = (b[0] - a[0]) * inv_area;
    let mut w0_row = edge(b[0], b[1], c[0], c[1], px, py) * inv_area;
    let mut w1_row = edge(c[0], c[1], a[0], a[1], px, py) * inv_area;
    let mut w2_row = edge(a[0], a[1], b[0], b[1], px, py) * inv_area;
    // Depth is linear too, so it can be stepped instead of computed from the
    // weights on every pixel.
    let dz_dx = dw0_dx * a[2] + dw1_dx * b[2] + dw2_dx * c[2];

    let width = target.width;
    let color = &mut target.color[..];
    let depth = &mut target.depth[..];

    for y in bbox_min_y..=bbox_max_y {
        if !(band_start..=band_end).contains(&y) {
            w0_row += dw0_dy;
            w1_row += dw1_dy;
            w2_row += dw2_dy;
            continue;
        }
        let Some((lo, hi)) = row_span(
            [w0_row, w1_row, w2_row],
            [dw0_dx, dw1_dx, dw2_dx],
            min_x,
            max_x,
        ) else {
            w0_row += dw0_dy;
            w1_row += dw1_dy;
            w2_row += dw2_dy;
            continue;
        };

        let step = (lo - min_x) as f32;
        let mut w0 = w0_row + dw0_dx * step;
        let mut w1 = w1_row + dw1_dx * step;
        let mut w2 = w2_row + dw2_dx * step;
        let mut z = w0 * a[2] + w1 * b[2] + w2 * c[2];
        let row = (y - band_start) as usize * width;
        for x in lo..=hi {
            let idx = row + x as usize;
            // Slightly conservative epsilon avoids z-fighting on shared edges.
            if z < depth[idx] + 1e-6 {
                depth[idx] = z;
                color[idx] = shade(color[idx], w0, w1, w2);
            }
            w0 += dw0_dx;
            w1 += dw1_dx;
            w2 += dw2_dx;
            z += dz_dx;
        }
        w0_row += dw0_dy;
        w1_row += dw1_dy;
        w2_row += dw2_dy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framebuffer::Framebuffer;

    const W: usize = 800;

    fn tri(target: &mut Target, z: f32, color: u32) {
        draw_triangle_z(
            target,
            [390.0, 390.0, z],
            [410.0, 390.0, z],
            [400.0, 410.0, z],
            color,
        );
    }

    /// Regression: the nearer of two overlapping triangles must win.
    #[test]
    fn depth_test_keeps_nearer_triangle() {
        let mut fb = Framebuffer::new(W, W);
        {
            let mut t = fb.target();
            tri(&mut t, 0.8, 1); // far first
            tri(&mut t, 0.1, 2); // near second
            assert_eq!(t.color[400 * W + 400], 2, "nearer triangle wins");
            assert!((t.depth[400 * W + 400] - 0.1).abs() < 1e-5);
        }
        // Reverse order: near first, far must not overwrite.
        let mut fb = Framebuffer::new(W, W);
        let mut t = fb.target();
        tri(&mut t, 0.1, 2);
        tri(&mut t, 0.8, 1);
        assert_eq!(t.color[400 * W + 400], 2, "far triangle rejected");
    }

    #[test]
    fn degenerate_triangle_draws_nothing() {
        let mut fb = Framebuffer::new(64, 64);
        let mut t = fb.target();
        draw_triangle_z(
            &mut t,
            [10.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            5,
        );
        assert!(t.color.iter().all(|&p| p == 0));
    }

    #[test]
    fn offscreen_triangle_is_cheap() {
        let mut fb = Framebuffer::new(64, 64);
        let mut t = fb.target();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            draw_triangle_z(
                &mut t,
                [1e7, 1e7, 0.0],
                [1e7 + 10.0, 1e7, 0.0],
                [1e7, 1e7 + 10.0, 0.0],
                1,
            );
        }
        assert!(start.elapsed().as_secs_f64() < 0.5, "bbox was not clamped");
    }

    #[test]
    fn gouraud_interpolates_vertex_colors() {
        let mut fb = Framebuffer::new(W, W);
        let mut t = fb.target();
        let mk = |x: f32, y: f32, color: [f32; 3]| ShadedVertex {
            x,
            y,
            z: 0.0,
            color,
        };
        draw_triangle_gouraud(
            &mut t,
            mk(380.0, 380.0, [1.0, 0.0, 0.0]),
            mk(420.0, 380.0, [0.0, 1.0, 0.0]),
            mk(400.0, 420.0, [0.0, 0.0, 1.0]),
        );
        let ch = |p: u32| ((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF);
        let (r, g, b) = ch(t.color[382 * W + 382]);
        assert!(r > 220 && g < 60 && b < 60, "red corner: {r},{g},{b}");
        let (r, g, b) = ch(t.color[382 * W + 417]);
        assert!(g > 220 && r < 60 && b < 60, "green corner: {r},{g},{b}");
        let (r, g, b) = ch(t.color[417 * W + 400]);
        assert!(b > 220 && r < 60 && g < 60, "blue corner: {r},{g},{b}");
    }

    #[test]
    fn shadow_triangle_darkens_existing_pixels() {
        let mut fb = Framebuffer::new(64, 64);
        fb.clear(0x808080);
        let mut t = fb.target();
        draw_triangle_shadow(
            &mut t,
            [10.0, 10.0, 0.0],
            [50.0, 10.0, 0.0],
            [30.0, 50.0, 0.0],
            [0.5, 0.5, 0.5],
        );
        let p = t.color[30 * 64 + 30];
        let [r, _, _] = crate::draw::pixel::unpack(p);
        assert!((r - 0.25).abs() < 0.02, "half tint over half gray: {r}");
    }
}
