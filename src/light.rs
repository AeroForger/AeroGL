//! Lighting description.
//! Made by AeroForger

use crate::math::Vec3;

/// A single directional (sun-like) light.
#[derive(Clone, Copy, Debug)]
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
