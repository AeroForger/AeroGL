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
    clip_segment_rect(
        0.0,
        width as f32 - 1.0,
        0.0,
        height as f32 - 1.0,
        x0,
        y0,
        x1,
        y1,
    )
}

/// Same clip, but against an arbitrary rectangle. Bands use a y range inside
/// the full screen so a line only steps through rows that band owns.
#[allow(clippy::too_many_arguments)] // clip rectangle plus the segment
fn clip_segment_rect(
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> Option<ScreenSeg> {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let p = [-dx, dx, -dy, dy];
    let q = [x0 - x_min, x_max - x0, y0 - y_min, y_max - y0];

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

/// Step indices whose sampled points can land in `[band_start, band_end]`.
///
/// Sampling stays based on the full segment, so the chosen pixels are exactly
/// the ones a full-screen draw would produce. The band only narrows the range,
/// which keeps a banded grid from walking the whole line again on every thread.
fn step_range_for_band(
    seg: &ScreenSeg,
    steps: i32,
    band_start: f32,
    band_end: f32,
) -> Option<(i32, i32)> {
    let dy = seg.y1 - seg.y0;
    if dy.abs() < 1e-6 {
        let y = seg.y0.round();
        return (y >= band_start && y <= band_end).then_some((0, steps));
    }
    let inv = 1.0 / dy;
    let a = (band_start - 0.5 - seg.y0) * inv;
    let b = (band_end + 0.5 - seg.y0) * inv;
    let (t0, t1) = if a <= b { (a, b) } else { (b, a) };
    // A nearly horizontal segment plus a band far from it makes these products
    // huge (up to ~1e13). Clamp in float first: a raw `as i32` saturates at
    // i32::MIN/MAX, and the trailing margin would then overflow on the add or
    // subtract. The margin is only a safety pad, so clamping it is harmless.
    let limit = steps as f32;
    let lo = (t0 * limit).floor() - 1.0;
    let hi = (t1 * limit).ceil() + 1.0;
    let i0 = lo.max(0.0).min(limit) as i32;
    let i1 = hi.max(0.0).min(limit) as i32;
    (i0 <= i1).then_some((i0, i1))
}

/// Draws a 1-pixel line with no depth test (overwrites).
///
/// Lines clip to the full screen, not to the target's band. A band iterates
/// only the step indices that can reach its rows, while sampling with the full
/// segment's step count, so it lands on exactly the same pixels as a
/// single-threaded full-screen draw.
pub fn draw_line(target: &mut Target, x0: f32, y0: f32, x1: f32, y1: f32, color: u32) {
    let Some(seg) = clip_segment_screen(target.width, target.height, x0, y0, x1, y1) else {
        return;
    };
    let steps = step_count(&seg, target.width, target.height);
    if steps == 0 {
        target.set_pixel(seg.x0.round() as i32, seg.y0.round() as i32, color);
        return;
    }
    let Some((i0, i1)) = step_range_for_band(
        &seg,
        steps,
        target.band_offset() as f32,
        target.band_end() as f32 - 1.0,
    ) else {
        return;
    };
    let (dx, dy) = (seg.x1 - seg.x0, seg.y1 - seg.y0);
    let inv = 1.0 / steps as f32;
    for i in i0..=i1 {
        let s = i as f32 * inv;
        let x = seg.x0 + dx * s;
        let y = seg.y0 + dy * s;
        target.set_pixel(x.round() as i32, y.round() as i32, color);
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
    let band_start = target.band_offset();
    let depth_write = target.depth_write;
    let Some((i0, i1)) = step_range_for_band(
        &seg,
        steps,
        band_start as f32,
        target.band_end() as f32 - 1.0,
    ) else {
        return;
    };

    for i in i0..=i1 {
        let s = i as f32 * inv;
        let t = seg.t0 + dt * s;
        let x = (seg.x0 + dx * s).round() as i32;
        let y = (seg.y0 + dy * s).round() as i32;
        if !(0..target.width as i32).contains(&x) || !target.contains_row(y) {
            continue;
        }
        let idx = (y as usize - band_start) * target.width + x as usize;
        let z = lerp(z0, z1, t);
        if z <= target.depth[idx] {
            if depth_write {
                target.depth[idx] = z;
            }
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

    /// A nearly horizontal segment (dy just above the flat-line cutoff) makes
    /// the band math divide by a tiny slope, so the step index lands in the
    /// billions. It must clamp, not overflow i32 on the +/-1 pad.
    #[test]
    fn near_horizontal_segment_with_far_bands_does_not_overflow() {
        let seg = ScreenSeg {
            x0: 0.0,
            y0: 40.0,
            t0: 0.0,
            x1: 1000.0,
            y1: 40.0 + 1e-5,
            t1: 1.0,
        };
        let steps = 2000;
        for (band_start, band_end) in [(0.0, 39.0), (0.0, 200.0), (1500.0, 1600.0)] {
            if let Some((i0, i1)) = step_range_for_band(&seg, steps, band_start, band_end) {
                assert!(i0 >= 0 && i1 <= steps && i0 <= i1, "band gave {i0}..{i1}");
            }
        }
    }

    /// End to end: a near-horizontal grid line drawn into every band must still
    /// light up the row it crosses, exactly like a full-screen draw would.
    #[test]
    fn near_horizontal_line_draws_in_the_band_that_owns_its_row() {
        let mut fb = Framebuffer::new(400, 300);
        {
            let mut target = fb.target();
            let mut bands = target.split_bands(7);
            for band in &mut bands {
                draw_line(band, 0.0, 150.0, 399.0, 150.0 + 1e-4, 9);
            }
        }
        let t = fb.target();
        assert_eq!(t.pixel(200, 150), 9, "row 150 is drawn by its owning band");
    }
}
