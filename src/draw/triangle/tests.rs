use super::*;
use crate::draw::{rgb, unpack};
use crate::framebuffer::{Framebuffer, Target};
use crate::texture::Texture;

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
    let mk = |x: f32, y: f32, color: [f32; 3]| ShadedVertex::new(x, y, 0.0, color);
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
fn texture_is_sampled_across_the_face() {
    let mut fb = Framebuffer::new(64, 64);
    let tex = Texture::from_fn(8, 1, |u, _| {
        if u < 0.5 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        }
    });
    {
        let mut t = fb.target();
        let white = [1.0, 1.0, 1.0];
        let v = |x, y, uv| ShadedVertex::new(x, y, 0.0, white).with_uv(uv);
        // Two triangles covering the full 64x64 target.
        draw_triangle_shaded(
            &mut t,
            v(0.0, 0.0, [0.0, 0.0]),
            v(64.0, 0.0, [1.0, 0.0]),
            v(64.0, 64.0, [1.0, 1.0]),
            Some(&tex),
        );
        draw_triangle_shaded(
            &mut t,
            v(0.0, 0.0, [0.0, 0.0]),
            v(64.0, 64.0, [1.0, 1.0]),
            v(0.0, 64.0, [0.0, 1.0]),
            Some(&tex),
        );
    }
    // Near the left edge the texel is red, near the right edge it is blue.
    let left = fb.color[32 * 64 + 4];
    let right = fb.color[32 * 64 + 60];
    assert!((left >> 16) & 0xFF > 200, "left should be red: {left:06x}");
    assert!(right & 0xFF > 200, "right should be blue: {right:06x}");
}

#[test]
fn alpha_blends_with_the_existing_pixel() {
    let mut fb = Framebuffer::new(32, 32);
    fb.clear(rgb(0.0, 0.0, 1.0));
    {
        let mut t = fb.target();
        let v = |x, y| ShadedVertex::new(x, y, 0.0, [1.0, 0.0, 0.0]).with_alpha(0.5);
        draw_triangle_shaded(&mut t, v(0.0, 0.0), v(32.0, 0.0), v(0.0, 32.0), None);
    }
    let [r, _, b] = unpack(fb.color[16 * 32 + 8]);
    assert!((r - 0.5).abs() < 0.02 && (b - 0.5).abs() < 0.02, "{r} {b}");
}

#[test]
fn alpha_is_interpolated_per_vertex() {
    let mut fb = Framebuffer::new(32, 32);
    fb.clear(rgb(0.0, 0.0, 1.0));
    {
        let mut t = fb.target();
        let v = |x, y, alpha| ShadedVertex::new(x, y, 0.0, [1.0, 0.0, 0.0]).with_alpha(alpha);
        draw_triangle_shaded(
            &mut t,
            v(0.0, 0.0, 0.0),
            v(32.0, 0.0, 1.0),
            v(0.0, 32.0, 1.0),
            None,
        );
    }
    let [left_r, _, left_b] = unpack(fb.color[2 * 32 + 2]);
    let [right_r, _, right_b] = unpack(fb.color[2 * 32 + 25]);
    assert!(left_r < 0.2 && left_b > 0.8, "left {left_r} {left_b}");
    assert!(
        right_r > 0.75 && right_b < 0.25,
        "right {right_r} {right_b}"
    );
}

#[test]
fn depth_write_can_be_disabled() {
    let mut fb = Framebuffer::new(32, 32);
    {
        let mut t = fb.target();
        t.set_depth_write(false);
        draw_triangle_z(
            &mut t,
            [0.0, 0.0, 0.0],
            [32.0, 0.0, 0.0],
            [0.0, 32.0, 0.0],
            1,
        );
    }
    assert!(fb.depth.iter().all(|d| d.is_infinite()));
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
