//! Orbit camera: spherical coordinates around a target, with projection.
//! Made by AeroForger

use crate::math::{Mat4, Vec3, finite_or, finite_vec_or};
use std::fmt;

/// A problem with the values in a [`Camera`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CameraError {
    /// An angle, target, or clip distance is not finite.
    NonFinite,
    /// Distance is zero or negative.
    InvalidDistance,
    /// Field of view is outside `(0, 180)` degrees.
    InvalidFieldOfView,
    /// Near is not positive, or far is not greater than near.
    InvalidClipPlanes,
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "camera fields must be finite"),
            Self::InvalidDistance => write!(f, "camera distance must be positive"),
            Self::InvalidFieldOfView => {
                write!(f, "camera field of view must be between 0 and 180 degrees")
            }
            Self::InvalidClipPlanes => {
                write!(
                    f,
                    "camera near must be positive and far must be greater than near"
                )
            }
        }
    }
}

impl std::error::Error for CameraError {}

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
        let target = finite_vec_or(self.target, Vec3::ZERO);
        let yaw = finite_or(self.yaw, 0.0);
        let pitch = finite_or(self.pitch, 0.35);
        let distance = finite_or(self.distance, 8.0).abs().max(1e-4);
        let cp = pitch.cos();
        Vec3::new(
            target.x + distance * cp * yaw.cos(),
            target.y + distance * pitch.sin(),
            target.z + distance * cp * yaw.sin(),
        )
    }

    /// View matrix (world to view).
    pub fn view(&self) -> Mat4 {
        let target = finite_vec_or(self.target, Vec3::ZERO);
        Mat4::look_at(self.eye(), target, Vec3::new(0.0, 1.0, 0.0))
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
        let near = finite_or(self.near, 0.1);
        let near = if near > 1e-4 { near } else { 0.1 };
        let far = finite_or(self.far, near + 100.0);
        let far = if far > near + 1e-3 { far } else { near + 100.0 };
        let fov_y_deg = if self.fov_y_deg.is_finite() {
            self.fov_y_deg.clamp(1.0, 179.0)
        } else {
            65.0
        };
        Mat4::perspective(fov_y_deg.to_radians(), aspect, near, far)
    }

    /// Checks that every field describes a usable camera.
    ///
    /// Rendering sanitizes bad values, but this reports them at setup time.
    pub fn validate(&self) -> Result<(), CameraError> {
        if !self.yaw.is_finite()
            || !self.pitch.is_finite()
            || !self.distance.is_finite()
            || !self.target.x.is_finite()
            || !self.target.y.is_finite()
            || !self.target.z.is_finite()
            || !self.fov_y_deg.is_finite()
            || !self.near.is_finite()
            || !self.far.is_finite()
        {
            return Err(CameraError::NonFinite);
        }
        if self.distance <= 0.0 {
            return Err(CameraError::InvalidDistance);
        }
        if !(0.0 < self.fov_y_deg && self.fov_y_deg < 180.0) {
            return Err(CameraError::InvalidFieldOfView);
        }
        if self.near <= 0.0 || self.far <= self.near {
            return Err(CameraError::InvalidClipPlanes);
        }
        Ok(())
    }

    /// Applies a mouse drag delta in pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        let dx = finite_or(dx, 0.0);
        let dy = finite_or(dy, 0.0);
        self.yaw = finite_or(self.yaw, 0.0) + dx * 0.01;
        self.pitch = (finite_or(self.pitch, 0.35) + dy * 0.01).clamp(-1.4, 1.4);
    }

    /// Applies scroll-wheel ticks, clamped to a sane zoom range.
    pub fn zoom(&mut self, wheel_y: f32) {
        let wheel_y = finite_or(wheel_y, 0.0);
        let distance = finite_or(self.distance, 8.0);
        self.distance = (distance * (1.0 - wheel_y * 0.05)).clamp(1.5, 40.0);
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

    #[test]
    fn eye_and_view_stay_finite_for_bad_public_fields() {
        let cam = Camera {
            yaw: f32::NAN,
            pitch: f32::INFINITY,
            distance: f32::NAN,
            target: Vec3::new(f32::NAN, f32::INFINITY, f32::NEG_INFINITY),
            ..Camera::default()
        };
        assert!(cam.eye().x.is_finite());
        assert!(cam.eye().y.is_finite());
        assert!(cam.eye().z.is_finite());
        assert!(cam.view().m.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn navigation_ignores_non_finite_deltas() {
        let mut cam = Camera::default();
        cam.orbit(f32::NAN, f32::INFINITY);
        cam.zoom(f32::NAN);
        assert!(cam.yaw.is_finite() && cam.pitch.is_finite() && cam.distance.is_finite());
        assert!(cam.validate().is_ok(), "{:?}", cam.validate());
    }

    #[test]
    fn validate_reports_bad_setup_values() {
        assert!(Camera::default().validate().is_ok());

        let cam = Camera {
            distance: 0.0,
            ..Camera::default()
        };
        assert_eq!(cam.validate(), Err(CameraError::InvalidDistance));

        let cam = Camera {
            fov_y_deg: 180.0,
            ..Camera::default()
        };
        assert_eq!(cam.validate(), Err(CameraError::InvalidFieldOfView));

        let default = Camera::default();
        let cam = Camera {
            far: default.near,
            ..default
        };
        assert_eq!(cam.validate(), Err(CameraError::InvalidClipPlanes));
    }
}
