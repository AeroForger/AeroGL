//! Surface material description: albedo, specular, textures, alpha, emission.
//! Made by AeroForger

use crate::texture::Texture;
use std::sync::Arc;

/// A Blinn-Phong surface material, optionally textured and transparent.
///
/// Cloning a material is cheap: textures are shared through an [`Arc`].
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    /// Base albedo color, linear RGB in `[0,1]`. Multiplied with the texture
    /// (if any) before lighting.
    pub color: [f32; 3],
    /// Specular highlight strength (`0.0` = matte).
    pub specular: f32,
    /// Specular exponent; higher is a tighter highlight.
    pub shininess: f32,
    /// How much of the rim light is added (`0.0` = none).
    pub rim: f32,
    /// Optional albedo texture, sampled with the vertex UVs.
    pub texture: Option<Arc<Texture>>,
    /// Opacity in `[0,1]`. Values below `1.0` draw the object in the sorted
    /// transparent pass, blended over what is already in the frame.
    pub alpha: f32,
    /// Light emitted by the surface itself, added after shading.
    pub emissive: [f32; 3],
}

impl Material {
    /// Matte material with the given base color.
    pub fn new(color: [f32; 3]) -> Self {
        Self {
            color,
            specular: 0.35,
            shininess: 48.0,
            rim: 0.10,
            texture: None,
            alpha: 1.0,
            emissive: [0.0, 0.0, 0.0],
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

    /// Attaches an albedo texture (builder style). The material color is
    /// ignored for textured surfaces and treated as white.
    pub fn with_texture(mut self, texture: Texture) -> Self {
        self.texture = Some(Arc::new(texture));
        self
    }

    /// Attaches a texture that is already shared with other materials.
    pub fn with_shared_texture(mut self, texture: Arc<Texture>) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Sets the opacity (builder style).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Sets the emitted light color (builder style).
    pub fn with_emissive(mut self, emissive: [f32; 3]) -> Self {
        self.emissive = emissive;
        self
    }

    /// Whether this material needs the blended transparent pass.
    pub fn is_transparent(&self) -> bool {
        self.alpha < 0.999
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([0.8, 0.8, 0.8])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_fields() {
        let m = Material::new([1.0, 0.5, 0.25])
            .with_specular(0.7, 32.0)
            .with_rim(0.2)
            .with_alpha(0.4)
            .with_texture(Texture::solid(4, [1.0, 1.0, 1.0]));
        assert_eq!(m.specular, 0.7);
        assert_eq!(m.rim, 0.2);
        assert!((m.alpha - 0.4).abs() < 1e-6);
        assert!(m.texture.is_some());
        assert!(m.is_transparent());
    }

    #[test]
    fn alpha_clamps_to_unit_range() {
        assert_eq!(Material::new([1.0; 3]).with_alpha(3.0).alpha, 1.0);
        assert_eq!(Material::new([1.0; 3]).with_alpha(-1.0).alpha, 0.0);
    }
}
