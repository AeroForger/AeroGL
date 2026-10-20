//! Near and far plane clipping in homogeneous clip space.

use crate::draw::{ShadedVertex, lerp};
use crate::math::Vec3;

/// One vertex in clip space, carrying its precomputed vertex color, texture
/// coordinate, and opacity.
#[derive(Clone, Copy)]
pub(super) struct ClipVertex {
    pub(super) pos: [f32; 4], // x, y, z, w, with no perspective divide yet
    pub(super) color: [f32; 3],
    pub(super) uv: [f32; 2],
    pub(super) alpha: f32,
}

/// Fixed-capacity polygon from clipping a triangle against the near and far
/// planes. A triangle gains at most one vertex per clip plane, so five is the
/// true maximum; eight leaves room and keeps the path allocation-free.
#[derive(Clone, Copy)]
pub(super) struct Poly<V: Copy> {
    verts: [V; 8],
    len: usize,
}

impl<V: Copy> Poly<V> {
    /// An empty polygon seeded with `fill` so the array is initialized without
    /// requiring `V: Default`.
    pub(super) fn new(fill: V) -> Self {
        Self {
            verts: [fill; 8],
            len: 0,
        }
    }

    pub(super) fn push(&mut self, v: V) {
        debug_assert!(self.len < self.verts.len(), "clip polygon overflow");
        self.verts[self.len] = v;
        self.len += 1;
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn as_slice(&self) -> &[V] {
        &self.verts[..self.len]
    }
}

impl ClipVertex {
    pub(super) fn ndc(&self) -> Vec3 {
        let inv_w = 1.0 / self.pos[3];
        Vec3::new(
            self.pos[0] * inv_w,
            self.pos[1] * inv_w,
            self.pos[2] * inv_w,
        )
    }

    pub(super) fn shaded(self, ndc: Vec3, width: usize, height: usize) -> ShadedVertex {
        let inv_w = if self.pos[3].abs() > 1e-9 {
            1.0 / self.pos[3]
        } else {
            1.0
        };
        ShadedVertex {
            x: (ndc.x * 0.5 + 0.5) * width as f32,
            y: (1.0 - (ndc.y * 0.5 + 0.5)) * height as f32,
            z: ndc.z,
            color: self.color,
            uv: self.uv,
            inv_w,
            alpha: self.alpha,
        }
    }

    pub(super) fn lerp(a: ClipVertex, b: ClipVertex, t: f32) -> ClipVertex {
        ClipVertex {
            pos: [
                lerp(a.pos[0], b.pos[0], t),
                lerp(a.pos[1], b.pos[1], t),
                lerp(a.pos[2], b.pos[2], t),
                lerp(a.pos[3], b.pos[3], t),
            ],
            color: [
                lerp(a.color[0], b.color[0], t),
                lerp(a.color[1], b.color[1], t),
                lerp(a.color[2], b.color[2], t),
            ],
            uv: [lerp(a.uv[0], b.uv[0], t), lerp(a.uv[1], b.uv[1], t)],
            alpha: lerp(a.alpha, b.alpha, t),
        }
    }
}

/// Sutherland-Hodgman clip of one triangle against the near plane
/// (`w >= near`). Vertices behind the camera would otherwise divide by a
/// negative `w` and produce garbage triangles wrapping the screen.
pub(super) fn clip_near(tri: [ClipVertex; 3], near: f32) -> Poly<ClipVertex> {
    let mut out = Poly::new(tri[0]);
    for i in 0..3 {
        let (a, b) = (tri[i], tri[(i + 1) % 3]);
        let da = a.pos[3] - near;
        let db = b.pos[3] - near;
        if da >= 0.0 {
            out.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            out.push(ClipVertex::lerp(a, b, t));
        }
    }
    out
}

/// Sutherland-Hodgman clip against the far plane (`z <= w`, i.e. NDC z <= 1).
pub(super) fn clip_far(poly: Poly<ClipVertex>) -> Poly<ClipVertex> {
    let verts = poly.as_slice();
    let mut out = Poly::new(verts[0]);
    for i in 0..verts.len() {
        let (a, b) = (verts[i], verts[(i + 1) % verts.len()]);
        let da = a.pos[3] - a.pos[2];
        let db = b.pos[3] - b.pos[2];
        if da >= 0.0 {
            out.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            out.push(ClipVertex::lerp(a, b, t));
        }
    }
    out
}
pub(super) fn ndc_to_screen(ndc: Vec3, width: usize, height: usize) -> (f32, f32) {
    (
        (ndc.x * 0.5 + 0.5) * width as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * height as f32,
    )
}

/// Linear interpolation between two clip-space points.
pub(super) fn lerp_clip(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cv(w: f32) -> ClipVertex {
        cv_z(w, 0.0)
    }

    fn cv_z(w: f32, z: f32) -> ClipVertex {
        ClipVertex {
            pos: [0.0, 0.0, z, w],
            color: [0.0, 0.0, 0.0],
            uv: [0.0, 0.0],
            alpha: 1.0,
        }
    }

    #[test]
    fn clip_far_drops_geometry_past_the_far_plane() {
        let mut outside = Poly::new(cv_z(1.0, 2.0));
        outside.push(cv_z(1.0, 2.0));
        outside.push(cv_z(1.0, 2.0));
        outside.push(cv_z(1.0, 2.0));
        assert_eq!(clip_far(outside).len(), 0);

        let mut inside = Poly::new(cv_z(1.0, -0.5));
        inside.push(cv_z(1.0, -0.5));
        inside.push(cv_z(1.0, -0.5));
        inside.push(cv_z(1.0, -0.5));
        assert_eq!(clip_far(inside).len(), 3);

        let mut straddle = Poly::new(cv_z(1.0, -0.5));
        straddle.push(cv_z(1.0, -0.5));
        straddle.push(cv_z(1.0, 2.0));
        straddle.push(cv_z(1.0, -0.5));
        assert_eq!(clip_far(straddle).len(), 4, "one vertex beyond far");
    }

    #[test]
    fn clip_near_splits_straddling_triangle() {
        let out = clip_near([cv(5.0), cv(2.0), cv(6.0)], 4.0);
        assert_eq!(out.len(), 4, "one vertex behind -> quad");
        assert!(out.as_slice().iter().all(|v| v.pos[3] >= 4.0 - 1e-4));
    }

    #[test]
    fn clip_near_drops_fully_behind_triangle() {
        assert_eq!(clip_near([cv(1.0), cv(2.0), cv(3.0)], 4.0).len(), 0);
    }

    #[test]
    fn clip_near_keeps_fully_in_front_triangle() {
        assert_eq!(clip_near([cv(5.0), cv(6.0), cv(7.0)], 4.0).len(), 3);
    }
}
