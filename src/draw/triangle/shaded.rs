//! Perspective-correct shaded triangles with optional texture mapping.

use crate::draw::pixel::{edge, rgb, unpack};
use crate::framebuffer::Target;
use crate::texture::Texture;

use super::flat::row_span;

/// A projected vertex: screen position, NDC depth, lit color, texture
/// coordinate, reciprocal clip-space `w`, and opacity.
///
/// `inv_w` is what makes interpolation perspective correct. Use
/// [`ShadedVertex::new`] (which sets `inv_w = 1.0`) when you only have
/// screen-space vertices; values of `1.0` make the textured path reduce to
/// plain linear interpolation.
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
    /// Texture coordinate.
    pub uv: [f32; 2],
    /// Reciprocal of clip-space `w` (`1/w`).
    pub inv_w: f32,
    /// Opacity in `[0,1]`. Below `1.0` the pixel is alpha-blended.
    pub alpha: f32,
}

impl ShadedVertex {
    /// A screen-space vertex with no texture coordinate (`inv_w = 1`).
    pub fn new(x: f32, y: f32, z: f32, color: [f32; 3]) -> Self {
        Self {
            x,
            y,
            z,
            color,
            uv: [0.0, 0.0],
            inv_w: 1.0,
            alpha: 1.0,
        }
    }

    /// Sets the texture coordinate (builder style).
    pub fn with_uv(mut self, uv: [f32; 2]) -> Self {
        self.uv = uv;
        self
    }

    /// Sets the reciprocal clip-space `w` (builder style).
    pub fn with_inv_w(mut self, inv_w: f32) -> Self {
        self.inv_w = inv_w;
        self
    }

    /// Sets the opacity (builder style).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Gouraud-shaded, z-buffered triangle: interpolates per-vertex colors and
/// depth across the face.
///
/// This is the untextured convenience wrapper around
/// [`draw_triangle_shaded`].
pub fn draw_triangle_gouraud(
    target: &mut Target,
    a: ShadedVertex,
    b: ShadedVertex,
    c: ShadedVertex,
) {
    draw_triangle_shaded(target, a, b, c, None);
}

/// Perspective-correct, z-buffered triangle with optional texture mapping and
/// per-vertex alpha.
///
/// Colors and UVs are interpolated as `attribute / w` and divided by the
/// interpolated `1 / w`, which keeps textures stable on large, steeply angled
/// faces. When `texture` is `Some`, the sampled albedo is multiplied with the
/// lit vertex color. When the vertex alpha is below `1.0`, the result is
/// blended over the existing pixel (draw transparent geometry back to front
/// and turn off depth writes yourself).
pub fn draw_triangle_shaded(
    target: &mut Target,
    a: ShadedVertex,
    b: ShadedVertex,
    c: ShadedVertex,
    texture: Option<&Texture>,
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

    // Perspective-correct attributes, pre-divided by w.
    let iw = [a.inv_w, b.inv_w, c.inv_w];
    let uvw = [
        [a.uv[0] * iw[0], a.uv[1] * iw[0]],
        [b.uv[0] * iw[1], b.uv[1] * iw[1]],
        [c.uv[0] * iw[2], c.uv[1] * iw[2]],
    ];
    let cw = [
        [a.color[0] * iw[0], a.color[1] * iw[0], a.color[2] * iw[0]],
        [b.color[0] * iw[1], b.color[1] * iw[1], b.color[2] * iw[1]],
        [c.color[0] * iw[2], c.color[1] * iw[2], c.color[2] * iw[2]],
    ];

    let mut d_iw_dx = 0.0;
    let mut d_u_dx = 0.0;
    let mut d_v_dx = 0.0;
    let mut d_r_dx = 0.0;
    let mut d_g_dx = 0.0;
    let mut d_b_dx = 0.0;
    let mut d_iw_dy = 0.0;
    let mut d_u_dy = 0.0;
    let mut d_v_dy = 0.0;
    let mut d_r_dy = 0.0;
    let mut d_g_dy = 0.0;
    let mut d_b_dy = 0.0;
    for i in 0..3 {
        let (dx, dy) = [(dw0_dx, dw0_dy), (dw1_dx, dw1_dy), (dw2_dx, dw2_dy)][i];
        d_iw_dx += dx * iw[i];
        d_iw_dy += dy * iw[i];
        d_u_dx += dx * uvw[i][0];
        d_u_dy += dy * uvw[i][0];
        d_v_dx += dx * uvw[i][1];
        d_v_dy += dy * uvw[i][1];
        d_r_dx += dx * cw[i][0];
        d_r_dy += dy * cw[i][0];
        d_g_dx += dx * cw[i][1];
        d_g_dy += dy * cw[i][1];
        d_b_dx += dx * cw[i][2];
        d_b_dy += dy * cw[i][2];
    }

    let dz_dx = dw0_dx * a.z + dw1_dx * b.z + dw2_dx * c.z;
    let mut iw_row = w0_row * iw[0] + w1_row * iw[1] + w2_row * iw[2];
    let mut u_row = w0_row * uvw[0][0] + w1_row * uvw[1][0] + w2_row * uvw[2][0];
    let mut v_row = w0_row * uvw[0][1] + w1_row * uvw[1][1] + w2_row * uvw[2][1];
    let mut r_row = w0_row * cw[0][0] + w1_row * cw[1][0] + w2_row * cw[2][0];
    let mut g_row = w0_row * cw[0][1] + w1_row * cw[1][1] + w2_row * cw[2][1];
    let mut b_row = w0_row * cw[0][2] + w1_row * cw[1][2] + w2_row * cw[2][2];

    let alpha = a.alpha.clamp(0.0, 1.0);
    let blend = alpha < 0.999;
    let depth_write = target.depth_write;
    let width = target.width;
    let color = &mut target.color[..];
    let depth = &mut target.depth[..];

    for y in bbox_min_y..=bbox_max_y {
        if !(band_start..=band_end).contains(&y) {
            step_shaded_row(
                [&mut w0_row, &mut w1_row, &mut w2_row],
                [dw0_dy, dw1_dy, dw2_dy],
                [
                    &mut iw_row,
                    &mut u_row,
                    &mut v_row,
                    &mut r_row,
                    &mut g_row,
                    &mut b_row,
                ],
                [d_iw_dy, d_u_dy, d_v_dy, d_r_dy, d_g_dy, d_b_dy],
            );
            continue;
        }
        let Some((lo, hi)) = row_span(
            [w0_row, w1_row, w2_row],
            [dw0_dx, dw1_dx, dw2_dx],
            min_x,
            max_x,
        ) else {
            step_shaded_row(
                [&mut w0_row, &mut w1_row, &mut w2_row],
                [dw0_dy, dw1_dy, dw2_dy],
                [
                    &mut iw_row,
                    &mut u_row,
                    &mut v_row,
                    &mut r_row,
                    &mut g_row,
                    &mut b_row,
                ],
                [d_iw_dy, d_u_dy, d_v_dy, d_r_dy, d_g_dy, d_b_dy],
            );
            continue;
        };

        let step = (lo - min_x) as f32;
        let mut z = (w0_row + dw0_dx * step) * a.z
            + (w1_row + dw1_dx * step) * b.z
            + (w2_row + dw2_dx * step) * c.z;
        let mut iw = iw_row + d_iw_dx * step;
        let mut u = u_row + d_u_dx * step;
        let mut v = v_row + d_v_dx * step;
        let mut r = r_row + d_r_dx * step;
        let mut g = g_row + d_g_dx * step;
        let mut b = b_row + d_b_dx * step;

        let row = (y - band_start) as usize * width;
        for x in lo..=hi {
            let idx = row + x as usize;
            // Slightly conservative epsilon avoids z-fighting on shared edges.
            if z < depth[idx] + 1e-6 {
                let w = if iw.abs() > 1e-9 { iw } else { 1.0 };
                let mut out = [r / w, g / w, b / w];
                if let Some(texture) = texture {
                    let texel = texture.sample(u / w, v / w);
                    out = [out[0] * texel[0], out[1] * texel[1], out[2] * texel[2]];
                }
                color[idx] = if blend {
                    blend_over(color[idx], out, alpha)
                } else {
                    rgb(out[0], out[1], out[2])
                };
                if depth_write {
                    depth[idx] = z;
                }
            }
            z += dz_dx;
            iw += d_iw_dx;
            u += d_u_dx;
            v += d_v_dx;
            r += d_r_dx;
            g += d_g_dx;
            b += d_b_dx;
        }
        step_shaded_row(
            [&mut w0_row, &mut w1_row, &mut w2_row],
            [dw0_dy, dw1_dy, dw2_dy],
            [
                &mut iw_row,
                &mut u_row,
                &mut v_row,
                &mut r_row,
                &mut g_row,
                &mut b_row,
            ],
            [d_iw_dy, d_u_dy, d_v_dy, d_r_dy, d_g_dy, d_b_dy],
        );
    }
}

/// Advances the barycentric weights and every perspective-corrected attribute
/// by one scanline.
#[inline]
fn step_shaded_row(
    [w0, w1, w2]: [&mut f32; 3],
    [dw0, dw1, dw2]: [f32; 3],
    attrs: [&mut f32; 6],
    deltas: [f32; 6],
) {
    *w0 += dw0;
    *w1 += dw1;
    *w2 += dw2;
    for (value, delta) in attrs.into_iter().zip(deltas) {
        *value += delta;
    }
}

/// Source-over alpha blend in the packed framebuffer format.
fn blend_over(dst: u32, src: [f32; 3], alpha: f32) -> u32 {
    let dst = unpack(dst);
    let inv = 1.0 - alpha;
    rgb(
        src[0] * alpha + dst[0] * inv,
        src[1] * alpha + dst[1] * inv,
        src[2] * alpha + dst[2] * inv,
    )
}
