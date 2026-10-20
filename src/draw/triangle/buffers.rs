//! Raw-buffer triangle passes for masks and depth-only shadow maps.

use crate::draw::pixel::edge;

use super::flat::row_span;

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

/// Depth-only triangle for a shadow map. `depth` is a full-screen NDC depth
/// buffer and smaller values win. Colors are not touched.
pub(crate) fn draw_triangle_depth(
    depth: &mut [f32],
    width: usize,
    height: usize,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) {
    let min_x = (a[0].min(b[0]).min(c[0]).floor() as i32).max(0);
    let max_x = (a[0].max(b[0]).max(c[0]).ceil() as i32).min(width as i32 - 1);
    let min_y = (a[1].min(b[1]).min(c[1]).floor() as i32).max(0);
    let max_y = (a[1].max(b[1]).max(c[1]).ceil() as i32).min(height as i32 - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge(a[0], a[1], b[0], b[1], c[0], c[1]);
    if area.abs() < f32::EPSILON {
        return;
    }
    let inv_area = 1.0 / area;

    let px = min_x as f32 + 0.5;
    let py = min_y as f32 + 0.5;
    let dw0_dx = (b[1] - c[1]) * inv_area;
    let dw0_dy = (c[0] - b[0]) * inv_area;
    let dw1_dx = (c[1] - a[1]) * inv_area;
    let dw1_dy = (a[0] - c[0]) * inv_area;
    let dw2_dx = (a[1] - b[1]) * inv_area;
    let dw2_dy = (b[0] - a[0]) * inv_area;
    let mut w0_row = edge(b[0], b[1], c[0], c[1], px, py) * inv_area;
    let mut w1_row = edge(c[0], c[1], a[0], a[1], px, py) * inv_area;
    let mut w2_row = edge(a[0], a[1], b[0], b[1], px, py) * inv_area;
    let dz_dx = dw0_dx * a[2] + dw1_dx * b[2] + dw2_dx * c[2];

    for y in min_y..=max_y {
        if let Some((lo, hi)) = row_span(
            [w0_row, w1_row, w2_row],
            [dw0_dx, dw1_dx, dw2_dx],
            min_x,
            max_x,
        ) {
            let step = (lo - min_x) as f32;
            let mut z = (w0_row + dw0_dx * step) * a[2]
                + (w1_row + dw1_dx * step) * b[2]
                + (w2_row + dw2_dx * step) * c[2];
            let row = y as usize * width;
            for x in lo..=hi {
                let idx = row + x as usize;
                if z < depth[idx] {
                    depth[idx] = z;
                }
                z += dz_dx;
            }
        }
        w0_row += dw0_dy;
        w1_row += dw1_dy;
        w2_row += dw2_dy;
    }
}
