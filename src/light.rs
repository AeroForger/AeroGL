//! Lighting: the primary directional sun plus point and spot lights.
//! Made by AeroForger

use crate::math::Vec3;

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
        self.ambient = ambient;
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
        self.intensity = intensity;
        self
    }

    /// Sets the falloff range; `0` disables attenuation (builder style).
    pub fn with_range(mut self, range: f32) -> Self {
        self.range = range;
        self
    }

    /// Smooth quadratic falloff, `1` at the light and `0` at `range`.
    pub fn attenuation(&self, distance: f32) -> f32 {
        if self.range <= 0.0 {
            return 1.0;
        }
        let t = (distance / self.range).clamp(0.0, 1.0);
        (1.0 - t * t) * (1.0 - t * t)
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
        self.inner_angle_deg = inner_deg;
        self.outer_angle_deg = outer_deg.max(inner_deg + 0.01);
        self
    }

    /// Sets the color multiplier (builder style).
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// Sets the falloff range (builder style).
    pub fn with_range(mut self, range: f32) -> Self {
        self.range = range;
        self
    }

    /// Distance falloff, same curve as [`PointLight::attenuation`].
    pub fn attenuation(&self, distance: f32) -> f32 {
        if self.range <= 0.0 {
            return 1.0;
        }
        let t = (distance / self.range).clamp(0.0, 1.0);
        (1.0 - t * t) * (1.0 - t * t)
    }

    /// Cone falloff for a unit vector pointing from the surface toward the
    /// light (`to_light`). Smooth between the inner and outer angle.
    pub fn cone_factor(&self, to_light: Vec3) -> f32 {
        let axis = self.direction.normalize();
        // `to_light` points toward the light, so the ray from the light toward
        // the surface is `-to_light`.
        let cos = (-to_light).dot(axis).clamp(-1.0, 1.0);
        let inner = self.inner_angle_deg.to_radians().cos();
        let outer = self.outer_angle_deg.to_radians().cos();
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
}
