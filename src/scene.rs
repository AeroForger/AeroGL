//! Scene graph: objects, ground, and lighting.
//! Made by AeroForger

use crate::light::DirectionalLight;
use crate::material::Material;
use crate::math::Mat4;
use crate::mesh::Mesh;

/// How an object is rasterized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DrawMode {
    /// Solid, Gouraud-shaded triangles.
    #[default]
    Shaded,
    /// Triangle edges only.
    Wireframe,
    /// Shaded surface plus triangle edges.
    ShadedWireframe,
}

/// A mesh instance in a [`Scene`].
#[derive(Clone, Debug)]
pub struct Object {
    /// The geometry.
    pub mesh: Mesh,
    /// Model matrix (model space to world space).
    pub transform: Mat4,
    /// Surface appearance.
    pub material: Material,
    /// Render mode for this object.
    pub mode: DrawMode,
    /// Whether this object casts a ground shadow.
    pub casts_shadow: bool,
}

impl Object {
    /// A shaded object with an identity transform that casts shadows.
    pub fn new(mesh: Mesh, material: Material) -> Self {
        Self {
            mesh,
            transform: Mat4::identity(),
            material,
            mode: DrawMode::Shaded,
            casts_shadow: true,
        }
    }

    /// Sets the model matrix (builder style).
    pub fn with_transform(mut self, transform: Mat4) -> Self {
        self.transform = transform;
        self
    }

    /// Sets the render mode (builder style).
    pub fn with_mode(mut self, mode: DrawMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enables or disables shadow casting (builder style).
    pub fn with_shadow(mut self, casts_shadow: bool) -> Self {
        self.casts_shadow = casts_shadow;
        self
    }
}

/// Ground plane behavior: grid overlay and planar shadows.
#[derive(Clone, Copy, Debug)]
pub struct Ground {
    /// Height of the ground plane in world units.
    pub y: f32,
    /// Draw the depth-tested grid.
    pub grid: bool,
    /// Draw hard planar shadows.
    pub shadows: bool,
    /// Per-channel multiplier applied where shadows land (0 = black shadow).
    pub shadow_tint: [f32; 3],
}

impl Default for Ground {
    fn default() -> Self {
        Self {
            y: -1.35,
            grid: true,
            shadows: true,
            shadow_tint: [0.42, 0.48, 0.60],
        }
    }
}

/// A collection of objects, a light, and ground settings.
///
/// ```
/// use aerogl::{Material, Mesh, Object, Scene};
///
/// let mut scene = Scene::new();
/// scene.add(Object::new(Mesh::cube(), Material::new([0.9, 0.4, 0.2])));
/// assert_eq!(scene.len(), 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Scene {
    /// All objects, drawn in insertion order.
    pub objects: Vec<Object>,
    /// The single directional light.
    pub light: DirectionalLight,
    /// Ground plane settings.
    pub ground: Ground,
}

impl Scene {
    /// An empty scene with default light and ground.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            light: DirectionalLight::default(),
            ground: Ground::default(),
        }
    }

    /// Adds an object and returns its index.
    pub fn add(&mut self, object: Object) -> usize {
        self.objects.push(object);
        self.objects.len() - 1
    }

    /// Convenience: adds a shaded object from a mesh and material.
    pub fn add_mesh(&mut self, mesh: Mesh, material: Material) -> usize {
        self.add(Object::new(mesh, material))
    }

    /// Mutable access to an object by index.
    pub fn object_mut(&mut self, index: usize) -> Option<&mut Object> {
        self.objects.get_mut(index)
    }

    /// Number of objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the scene has no objects.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn add_returns_usable_index() {
        let mut scene = Scene::new();
        let i = scene.add_mesh(Mesh::cube(), Material::new([1.0, 0.0, 0.0]));
        assert_eq!(i, 0);
        scene.object_mut(i).unwrap().transform = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(scene.len(), 1);
        assert!(!scene.is_empty());
        assert!(scene.object_mut(9).is_none());
    }

    #[test]
    fn builder_methods_set_fields() {
        let o = Object::new(Mesh::cube(), Material::default())
            .with_mode(DrawMode::Wireframe)
            .with_shadow(false);
        assert_eq!(o.mode, DrawMode::Wireframe);
        assert!(!o.casts_shadow);
    }
}
