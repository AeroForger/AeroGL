//! Ground grid projection and drawing.

use crate::draw::{draw_line_z, rgb};
use crate::framebuffer::Target;
use crate::math::{Mat4, Vec3};

use super::clip::{lerp_clip, ndc_to_screen};

/// One ground-grid segment, already projected to screen space.
#[derive(Clone, Copy)]
pub(super) struct GridSegment {
    x0: f32,
    y0: f32,
    z0: f32,
    x1: f32,
    y1: f32,
    z1: f32,
    color: u32,
}

/// Projects the ground grid once per frame.
pub(super) fn prepare_grid(
    view_proj: Mat4,
    near: f32,
    width: usize,
    height: usize,
    y: f32,
    extent: i32,
) -> Vec<GridSegment> {
    let minor = rgb(0.13, 0.19, 0.25);
    let major = rgb(0.20, 0.30, 0.38);
    let mut out = Vec::with_capacity((extent as usize * 2 + 1) * 2);

    let mut segment = |a: Vec3, b: Vec3, color: u32| {
        let ca = view_proj.transform_h(a);
        let cb = view_proj.transform_h(b);
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
        let sa = ndc_to_screen(Vec3::new(ca[0] / ca[3], ca[1] / ca[3], za), width, height);
        let sb = ndc_to_screen(Vec3::new(cb[0] / cb[3], cb[1] / cb[3], zb), width, height);
        out.push(GridSegment {
            x0: sa.0,
            y0: sa.1,
            z0: za,
            x1: sb.0,
            y1: sb.1,
            z1: zb,
            color,
        });
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
    out
}

/// Depth-tested ground grid: minor lines every 1 unit, major every 5.
pub fn draw_grid(target: &mut Target, view: Mat4, proj: Mat4, near: f32, y: f32, extent: i32) {
    let view_proj = proj.multiply(&view);
    let segments = prepare_grid(view_proj, near, target.width, target.height, y, extent);
    for s in &segments {
        draw_line_z(target, s.x0, s.y0, s.z0, s.x1, s.y1, s.z1, s.color);
    }
}

/// Draws precomputed grid segments into a target (or band).
pub(super) fn draw_grid_segments(target: &mut Target, segments: &[GridSegment]) {
    for s in segments {
        draw_line_z(target, s.x0, s.y0, s.z0, s.x1, s.y1, s.z1, s.color);
    }
}
