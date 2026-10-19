//! Line drawing: Bresenham, float stepping, depth testing, screen clipping.
//! Made by AeroForger

use crate::draw::pixel::lerp;
use crate::framebuffer::Target;

/// A segment clipped to the screen rectangle. `t0`/`t1` are parameters along
/// the original segment, so callers can interpolate depth.
pub struct ScreenSeg {
    /// Start x on screen.
    pub x0: f32,
    /// Start y on screen.
    pub y0: f32,
    /// Parameter of the start point along the original segment.
    pub t0: f32,
    /// End x on screen.
    pub x1: f32,
    /// End y on screen.
    pub y1: f32,
    /// Parameter of the end point along the original segment.
    pub t1: f32,
}

/// Liang-Barsky clip of a segment against a `width` x `height` screen.
///
/// This matters more than it looks: a vertex clipped at the near plane can land
/// at screen coordinates in the millions, and stepping a line pixel by pixel
/// across that distance costs millions of iterations per line.
pub fn clip_segment_screen(
    width: usize,
    height: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> Option<ScreenSeg> {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let max_x = width as f32 - 1.0;
    let max_y = height as f32 - 1.0;

    let p = [-dx, dx, -dy, dy];
    let q = [x0, max_x - x0, y0, max_y - y0];

    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for i in 0..4 {
        if p[i].abs() < f32::EPSILON {
            if q[i] < 0.0 {
                return None; // parallel and outside this edge
            }
        } else {
            let r = q[i] / p[i];
            if p[i] < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return None;
            }
        }
    }
    Some(ScreenSeg {
        x0: x0 + dx * t0,
        y0: y0 + dy * t0,
        t0,
        x1: x0 + dx * t1,
        y1: y0 + dy * t1,
        t1,
    })
}

/// Steps for a clipped segment, capped so a bad endpoint can never explode.
fn step_count(seg: &ScreenSeg, width: usize, height: usize) -> i32 {
    let steps = (seg.x1 - seg.x0).abs().max((seg.y1 - seg.y0).abs()).round() as i32;
    steps.clamp(0, 2 * (width + height) as i32)
}

/// Draws a 1-pixel line with no depth test (overwrites).
pub fn draw_line(target: &mut Target, x0: f32, y0: f32, x1: f32, y1: f32, color: u32) {
    let Some(seg) = clip_segment_screen(target.width, target.height, x0, y0, x1, y1) else {
        return;
    };
    let steps = step_count(&seg, target.width, target.height);
    if steps == 0 {
        target.set_pixel(seg.x0.round() as i32, seg.y0.round() as i32, color);
        return;
    }
    let (dx, dy) = (seg.x1 - seg.x0, seg.y1 - seg.y0);
    let (x_inc, y_inc) = (dx / steps as f32, dy / steps as f32);
    let (mut x, mut y) = (seg.x0, seg.y0);
    for _ in 0..=steps {
        target.set_pixel(x.round() as i32, y.round() as i32, color);
        x += x_inc;
        y += y_inc;
    }
}

/// Draws a depth-tested 3D line: interpolates NDC depth and writes only pixels
/// that pass the z-test (smaller is closer).
#[allow(clippy::too_many_arguments)] // x0,y0,z0,x1,y1,z1 grouped in threes
pub fn draw_line_z(
    target: &mut Target,
    x0: f32,
    y0: f32,
    z0: f32,
    x1: f32,
    y1: f32,
    z1: f32,
    color: u32,
) {
    let Some(seg) = clip_segment_screen(target.width, target.height, x0, y0, x1, y1) else {
        return;
    };
    let steps = step_count(&seg, target.width, target.height);
    let inv = if steps == 0 { 0.0 } else { 1.0 / steps as f32 };
    let (dx, dy) = (seg.x1 - seg.x0, seg.y1 - seg.y0);
    let dt = seg.t1 - seg.t0;

    for i in 0..=steps {
        let s = i as f32 * inv;
        let t = seg.t0 + dt * s;
        let x = (seg.x0 + dx * s).round() as i32;
        let y = (seg.y0 + dy * s).round() as i32;
        if !(0..target.width as i32).contains(&x) || !(0..target.height as i32).contains(&y) {
            continue;
        }
        let idx = y as usize * target.width + x as usize;
        let z = lerp(z0, z1, t);
        if z <= target.depth[idx] {
            target.depth[idx] = z;
            target.color[idx] = color;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framebuffer::Framebuffer;

    #[test]
    fn fully_offscreen_segment_is_rejected() {
        assert!(clip_segment_screen(800, 800, -100.0, -100.0, -50.0, -80.0).is_none());
        assert!(clip_segment_screen(800, 800, 900.0, 10.0, 1200.0, 40.0).is_none());
    }

    #[test]
    fn onscreen_segment_passes_through_unchanged() {
        let s = clip_segment_screen(800, 800, 10.0, 20.0, 100.0, 200.0).expect("visible");
        assert_eq!((s.t0, s.t1), (0.0, 1.0));
        assert_eq!((s.x0, s.y0), (10.0, 20.0));
    }

    /// The regression that cost ~1.7 s/frame before clipping existed: an
    /// endpoint far off-screen (near-plane clipping can produce coordinates in
    /// the millions).
    #[test]
    fn huge_offscreen_endpoint_is_clipped_to_screen() {
        let s = clip_segment_screen(800, 800, -4_000_000.0, 400.0, 400.0, 400.0).expect("visible");
        assert!(s.x0 >= 0.0 && s.x0 <= 800.0, "clipped x0 in screen");
        assert!(step_count(&s, 800, 800) <= 2 * 1600);
    }

    #[test]
    fn huge_line_terminates_fast() {
        let mut fb = Framebuffer::new(800, 800);
        let mut target = fb.target();
        let start = std::time::Instant::now();
        for _ in 0..100 {
            draw_line_z(&mut target, -9_000_000.0, 400.0, 0.0, 400.0, 400.0, 0.0, 7);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        assert!(ms < 100.0, "100 huge lines took {ms:.1} ms");
    }

    #[test]
    fn nearer_line_wins_crossing() {
        let mut fb = Framebuffer::new(800, 800);
        let mut t = fb.target();
        draw_line_z(&mut t, 100.0, 300.0, 0.8, 700.0, 300.0, 0.8, 1);
        draw_line_z(&mut t, 400.0, 100.0, 0.2, 400.0, 700.0, 0.2, 2);
        assert_eq!(t.color[300 * 800 + 400], 2, "nearer vertical line wins");
    }

    #[test]
    fn line_connects_endpoints() {
        let mut fb = Framebuffer::new(20, 20);
        let mut t = fb.target();
        draw_line(&mut t, 0.0, 0.0, 10.0, 10.0, 1);
        for i in 0..=10 {
            assert_eq!(t.color[i * 20 + i], 1, "diagonal pixel {i}");
        }
    }
}
