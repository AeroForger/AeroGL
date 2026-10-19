//! Surface material description.
//! Made by AeroForger

/// An opaque, non-textured surface material with Blinn-Phong parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Base albedo color, linear RGB in `[0,1]`.
    pub color: [f32; 3],
    /// Specular highlight strength (`0.0` = matte).
    pub specular: f32,
    /// Specular exponent; higher is a tighter highlight.
    pub shininess: f32,
    /// How much of the rim light is added (`0.0` = none).
    pub rim: f32,
}

impl Material {
    /// Matte material with the given base color.
    pub fn new(color: [f32; 3]) -> Self {
        Self {
            color,
            specular: 0.35,
            shininess: 48.0,
            rim: 0.10,
        }
    }

    /// Sets the specular strength and exponent (builder style).
    pub fn with_specular(mut self, strength: f32, shininess: f32) -> Self {
        self.specular = strength;
        self.shininess = shininess;
        self
    }

    /// Sets the rim-light amount (builder style).
    pub fn with_rim(mut self, rim: f32) -> Self {
        self.rim = rim;
        self
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([0.8, 0.8, 0.8])
    }
}
