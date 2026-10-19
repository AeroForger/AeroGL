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
/// depth across the face with barycentric weights.
pub fn draw_triangle_gouraud(
    target: &mut Target,
    a: ShadedVertex,
    b: ShadedVertex,
    c: ShadedVertex,
) {
    rasterize(
        target,
        [a.x, a.y, a.z],
        [b.x, b.y, b.z],
        [c.x, c.y, c.z],
        &|_, w0, w1, w2| {
            rgb(
                a.color[0] * w0 + b.color[0] * w1 + c.color[0] * w2,
                a.color[1] * w0 + b.color[1] * w1 + c.color[1] * w2,
                a.color[2] * w0 + b.color[2] * w1 + c.color[2] * w2,
            )
        },
    );
}

/// Flat, single-color z-buffered triangle.
///
/// Each point is `[x, y, z]`: screen pixels plus NDC depth (near = -1,
/// far = +1; smaller wins).
pub fn draw_triangle_z(target: &mut Target, a: [f32; 3], b: [f32; 3], c: [f32; 3], color: u32) {
    rasterize(target, a, b, c, &|_, _, _, _| color);
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
    rasterize(target, a, b, c, &|existing, _, _, _| {
        let [r, g, b] = crate::draw::pixel::unpack(existing);
        rgb(r * tint[0], g * tint[1], b * tint[2])
    });
}

/// Shared barycentric/z-test core. `shade` receives the pixel already in the
/// frame buffer so modes can replace it (flat/Gouraud) or modulate it (shadow).
fn rasterize(
    target: &mut Target,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    shade: &dyn Fn(u32, f32, f32, f32) -> u32,
) {
    // Clamp the bounding box to the screen *before* looping. A vertex clipped
    // at the near plane can project millions of pixels off-screen; without
    // this the per-pixel loop would walk that whole range.
    let min_x = (a[0].min(b[0]).min(c[0]).floor() as i32).max(0);
    let max_x = (a[0].max(b[0]).max(c[0]).ceil() as i32).min(target.width as i32 - 1);
    let min_y = (a[1].min(b[1]).min(c[1]).floor() as i32).max(0);
    let max_y = (a[1].max(b[1]).max(c[1]).ceil() as i32).min(target.height as i32 - 1);
    if min_x > max_x || min_y > max_y {
        return; // completely off-screen
    }

    // Barycentric denominator (twice the signed area).
    let area = edge(a[0], a[1], b[0], b[1], c[0], c[1]);
    if area.abs() < f32::EPSILON {
        return; // degenerate
    }
    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            // Normalized barycentric weights, so w0 + w1 + w2 = 1.
            let w0 = edge(b[0], b[1], c[0], c[1], px, py) * inv_area;
            let w1 = edge(c[0], c[1], a[0], a[1], px, py) * inv_area;
            let w2 = edge(a[0], a[1], b[0], b[1], px, py) * inv_area;
            if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                continue;
            }

            let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
            let idx = y as usize * target.width + x as usize;
            // Slightly conservative epsilon avoids z-fighting on shared edges.
            if z >= target.depth[idx] + 1e-6 {
                continue;
            }
            target.depth[idx] = z;
            target.color[idx] = shade(target.color[idx], w0, w1, w2);
        }
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
