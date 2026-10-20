//! Orbit camera: spherical coordinates around a target, with projection.
//! Made by AeroForger

use crate::math::{Mat4, Vec3};

/// An orbit camera looking at [`Camera::target`].
///
/// Yaw/pitch are radians; distance is in world units.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Horizontal angle around the target.
    pub yaw: f32,
    /// Vertical angle above the target (clamped when orbiting).
    pub pitch: f32,
    /// Distance from the target.
    pub distance: f32,
    /// Point the camera orbits and looks at.
    pub target: Vec3,
    /// Vertical field of view in degrees.
    pub fov_y_deg: f32,
    /// Near clip distance.
    pub near: f32,
    /// Far clip distance.
    pub far: f32,
}

impl Camera {
    /// Creates a camera orbiting the origin.
    pub fn new(yaw: f32, pitch: f32, distance: f32) -> Self {
        Self {
            yaw,
            pitch,
            distance,
            target: Vec3::ZERO,
            fov_y_deg: 65.0,
            near: 0.1,
            far: 100.0,
        }
    }

    /// World-space eye position.
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        Vec3::new(
            self.target.x + self.distance * cp * self.yaw.cos(),
            self.target.y + self.distance * self.pitch.sin(),
            self.target.z + self.distance * cp * self.yaw.sin(),
        )
    }

    /// View matrix (world to view).
    pub fn view(&self) -> Mat4 {
        Mat4::look_at(self.eye(), self.target, Vec3::new(0.0, 1.0, 0.0))
    }

    /// Projection matrix for the given aspect ratio (`width / height`).
    ///
    /// Degenerate fields are sanitized instead of producing NaNs: a non-finite
    /// or non-positive aspect becomes `1.0`, the FOV is clamped to
    /// `(1, 179)` degrees, and near/far fall back to sane values.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let aspect = if aspect.is_finite() && aspect > 1e-6 {
            aspect
        } else {
            1.0
        };
        let near = if self.near.is_finite() && self.near > 1e-4 {
            self.near
        } else {
            0.1
        };
        let far = if self.far.is_finite() && self.far > near + 1e-3 {
            self.far
        } else {
            near + 100.0
        };
        let fov_y_deg = if self.fov_y_deg.is_finite() {
            self.fov_y_deg.clamp(1.0, 179.0)
        } else {
            65.0
        };
        Mat4::perspective(fov_y_deg.to_radians(), aspect, near, far)
    }

    /// Applies a mouse drag delta in pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.4, 1.4);
    }

    /// Applies scroll-wheel ticks, clamped to a sane zoom range.
    pub fn zoom(&mut self, wheel_y: f32) {
        self.distance = (self.distance * (1.0 - wheel_y * 0.05)).clamp(1.5, 40.0);
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(0.6, 0.35, 8.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_is_distance_away_from_target() {
        let cam = Camera::new(0.3, 0.2, 5.0);
        let d = cam.eye().sub(cam.target).length();
        assert!((d - 5.0).abs() < 1e-4);
    }

    #[test]
    fn orbiting_stays_level_and_zooms_in_range() {
        let mut cam = Camera::default();
        cam.orbit(1000.0, -1000.0);
        assert!(cam.pitch <= 1.4 && cam.pitch >= -1.4);
        for _ in 0..100 {
            cam.zoom(1.0);
        }
        assert!(
            (cam.distance - 1.5).abs() < 1e-4,
            "zoom clamps at near limit"
        );
    }

    #[test]
    fn projection_sanitizes_degenerate_fields() {
        let cam = Camera {
            fov_y_deg: 0.0,
            near: 0.0,
            far: 0.0,
            ..Camera::default()
        };
        let m = cam.projection(0.0);
        assert!(m.m.iter().flatten().all(|v| v.is_finite()), "{:?}", m.m);
    }
}
