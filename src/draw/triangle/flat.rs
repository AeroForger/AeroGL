//! Flat, z-buffered triangles and the shared barycentric rasterizer.

use crate::draw::pixel::{edge, rgb};
use crate::framebuffer::Target;

/// Flat, single-color z-buffered triangle.
///
/// Each point is `[x, y, z]`: screen pixels plus NDC depth (near = -1,
/// far = +1; smaller wins).
pub fn draw_triangle_z(target: &mut Target, a: [f32; 3], b: [f32; 3], c: [f32; 3], color: u32) {
    rasterize(target, a, b, c, |_, _, _, _| color);
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
pub(super) fn row_span(w: [f32; 3], dw: [f32; 3], min_x: i32, max_x: i32) -> Option<(i32, i32)> {
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
    let depth_write = target.depth_write;
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
                if depth_write {
                    depth[idx] = z;
                }
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
