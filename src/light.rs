//! Lighting: the primary directional sun plus point and spot lights.
//! Made by AeroForger

use crate::math::{Vec3, finite_or, finite_vec_or, normalized_or};

/// Direction used when a light direction is zero or non-finite.
pub(crate) const DEFAULT_LIGHT_DIR: Vec3 = Vec3::new(0.0, -1.0, 0.0);

/// A directional (sun-like) light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Direction the light travels, pointing toward the scene. Normalized
    /// automatically at render time.
    pub direction: Vec3,
    /// Light color, linear RGB.
    pub color: [f32; 3],
    /// Ambient fill, added everywhere regardless of facing.
    pub ambient: f32,
}

impl DirectionalLight {
    /// A sun pointing down and to the left, slightly toward the viewer.
    pub fn new(direction: Vec3, color: [f32; 3]) -> Self {
        Self {
            direction,
            color,
            ambient: 0.15,
        }
    }

    /// Overrides the ambient term (builder style).
    pub fn with_ambient(mut self, ambient: f32) -> Self {
        self.ambient = finite_or(ambient, 0.15).clamp(0.0, 1.0);
        self
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self::new(Vec3::new(-0.45, -1.0, -0.35), [1.0, 1.0, 1.0])
    }
}

/// A point light: falls off with distance and reaches nothing past `range`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// World-space position.
    pub position: Vec3,
    /// Light color, linear RGB.
    pub color: [f32; 3],
    /// Multiplier on `color`.
    pub intensity: f32,
    /// Distance at which the light has fully faded out. `0` disables falloff.
    pub range: f32,
}

impl PointLight {
    /// A white light at `position`.
    pub fn new(position: Vec3, color: [f32; 3]) -> Self {
        Self {
            position,
            color,
            intensity: 1.0,
            range: 8.0,
        }
    }

    /// Sets the color multiplier (builder style).
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = finite_or(intensity, 0.0).max(0.0);
        self
    }

    /// Sets the falloff range; `0` disables attenuation (builder style).
    pub fn with_range(mut self, range: f32) -> Self {
        self.range = finite_or(range, 0.0).max(0.0);
        self
    }

    /// Smooth quadratic falloff, `1` at the light and `0` at `range`.
    pub fn attenuation(&self, distance: f32) -> f32 {
        falloff(distance, self.range)
    }
}

/// A spot light: a point light restricted to a cone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotLight {
    /// World-space position.
    pub position: Vec3,
    /// Direction the cone points (toward the scene).
    pub direction: Vec3,
    /// Light color, linear RGB.
    pub color: [f32; 3],
    /// Multiplier on `color`.
    pub intensity: f32,
    /// Distance at which the light has fully faded out.
    pub range: f32,
    /// Full-intensity cone half-angle in degrees.
    pub inner_angle_deg: f32,
    /// Zero-intensity cone half-angle in degrees.
    pub outer_angle_deg: f32,
}

impl SpotLight {
    /// A white spot pointing down `direction`.
    pub fn new(position: Vec3, direction: Vec3, color: [f32; 3]) -> Self {
        Self {
            position,
            direction,
            color,
            intensity: 2.0,
            range: 12.0,
            inner_angle_deg: 15.0,
            outer_angle_deg: 30.0,
        }
    }

    /// Sets the cone angles in degrees (builder style).
    pub fn with_cone(mut self, inner_deg: f32, outer_deg: f32) -> Self {
        let (inner, outer) = cone_angles(inner_deg, outer_deg);
        self.inner_angle_deg = inner;
        self.outer_angle_deg = outer;
        self
    }

    /// Sets the color multiplier (builder style).
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = finite_or(intensity, 0.0).max(0.0);
        self
    }

    /// Sets the falloff range (builder style).
    pub fn with_range(mut self, range: f32) -> Self {
        self.range = finite_or(range, 0.0).max(0.0);
        self
    }

    /// Distance falloff, same curve as [`PointLight::attenuation`].
    pub fn attenuation(&self, distance: f32) -> f32 {
        falloff(distance, self.range)
    }

    /// Cone falloff for a unit vector pointing from the surface toward the
    /// light (`to_light`). Smooth between the inner and outer angle.
    pub fn cone_factor(&self, to_light: Vec3) -> f32 {
        let axis = normalized_or(self.direction, DEFAULT_LIGHT_DIR);
        // `to_light` points toward the light, so the ray from the light toward
        // the surface is `-to_light`.
        let to_light = finite_vec_or(to_light, Vec3::ZERO);
        let cos = (-to_light).dot(axis).clamp(-1.0, 1.0);
        let (inner, outer) = cone_cosines(self.inner_angle_deg, self.outer_angle_deg);
        if cos <= outer {
            return 0.0;
        }
        if cos >= inner {
            return 1.0;
        }
        let t = (cos - outer) / (inner - outer).max(1e-6);
        t * t * (3.0 - 2.0 * t)
    }
}

/// An additional local light in a [`crate::Scene`].
///
/// The primary sun lives in [`crate::Scene::light`]; entries in
/// [`crate::Scene::lights`] are added on top of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    /// Omnidirectional point light.
    Point(PointLight),
    /// Cone-shaped spot light.
    Spot(SpotLight),
}

/// Distance falloff shared by point and spot lights.
pub(crate) fn falloff(distance: f32, range: f32) -> f32 {
    let range = finite_or(range, 0.0).max(0.0);
    if range <= 0.0 {
        return 1.0;
    }
    let distance = finite_or(distance, 0.0).max(0.0);
    let t = (distance / range).clamp(0.0, 1.0);
    (1.0 - t * t) * (1.0 - t * t)
}

/// Clamps cone angles so `outer` is always slightly wider than `inner`.
pub(crate) fn cone_angles(inner_deg: f32, outer_deg: f32) -> (f32, f32) {
    let inner = finite_or(inner_deg, 15.0).clamp(0.0, 179.0);
    let outer = finite_or(outer_deg, 30.0).clamp(inner + 0.01, 180.0);
    (inner, outer)
}

/// Cosine values for a cone, with the angles sanitized first.
pub(crate) fn cone_cosines(inner_deg: f32, outer_deg: f32) -> (f32, f32) {
    let (inner, outer) = cone_angles(inner_deg, outer_deg);
    (inner.to_radians().cos(), outer.to_radians().cos())
}

/// Repairs cosine values that were set directly on a public [`SpotLight`] or
/// [`crate::pipeline::ResolvedLight`].
pub(crate) fn sanitize_cone_cosines(inner_cos: f32, outer_cos: f32) -> (f32, f32) {
    let inner = finite_or(inner_cos, 1.0).clamp(-1.0, 1.0);
    let outer = finite_or(outer_cos, -1.0).clamp(-1.0, 1.0);
    if inner > outer {
        (inner, outer)
    } else {
        // Inverted or equal values would divide by zero. Keep a narrow band
        // around the requested inner edge instead of calling acos per vertex.
        let center = inner.clamp(-1.0, 1.0 - 2e-4);
        ((center + 1e-4).min(1.0), (center - 1e-4).max(-1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_light_fades_to_zero_at_range() {
        let light = PointLight::new(Vec3::ZERO, [1.0; 3]).with_range(10.0);
        assert!((light.attenuation(0.0) - 1.0).abs() < 1e-6);
        assert!((light.attenuation(5.0) - 0.5625).abs() < 1e-4);
        assert!(light.attenuation(10.0) <= 1e-6);
    }

    #[test]
    fn spot_cone_is_full_inside_and_dark_outside() {
        let spot = SpotLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            [1.0; 3],
        )
        .with_cone(10.0, 20.0);
        // Straight down: inside the inner cone.
        assert!((spot.cone_factor(Vec3::new(0.0, 1.0, 0.0)) - 1.0).abs() < 1e-6);
        // Far sideways: outside the outer cone.
        assert_eq!(spot.cone_factor(Vec3::new(5.0, 0.0, 0.0)), 0.0);
    }

    #[test]
    fn inverted_or_non_finite_cone_angles_are_repaired() {
        let mut spot = SpotLight::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), [1.0; 3]);
        spot.inner_angle_deg = 70.0;
        spot.outer_angle_deg = 20.0;
        let inside = spot.cone_factor(Vec3::new(0.0, 1.0, 0.0));
        assert!((0.0..=1.0).contains(&inside), "cone factor {inside}");

        spot.inner_angle_deg = f32::NAN;
        spot.outer_angle_deg = f32::INFINITY;
        let factor = spot.cone_factor(Vec3::new(0.0, 1.0, 0.0));
        assert!(factor.is_finite() && (0.0..=1.0).contains(&factor));
    }

    #[test]
    fn light_builders_sanitize_negative_and_non_finite_values() {
        let point = PointLight::new(Vec3::ZERO, [1.0; 3])
            .with_intensity(-2.0)
            .with_range(f32::NAN);
        assert_eq!(point.intensity, 0.0);
        assert_eq!(point.range, 0.0);
        assert_eq!(point.attenuation(f32::NAN), 1.0);

        let spot = SpotLight::new(Vec3::ZERO, Vec3::ZERO, [1.0; 3])
            .with_cone(f32::NAN, f32::NAN)
            .with_intensity(f32::NAN)
            .with_range(-4.0);
        assert!(spot.inner_angle_deg.is_finite());
        assert!(spot.outer_angle_deg > spot.inner_angle_deg);
        assert_eq!(spot.intensity, 0.0);
        assert_eq!(spot.range, 0.0);

        let (inner, outer) = sanitize_cone_cosines(f32::NAN, f32::NAN);
        assert!(inner.is_finite() && outer.is_finite());
        assert!(inner > outer);
    }
}
