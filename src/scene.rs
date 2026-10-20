//! Scene graph: objects, ground, fog, and lighting.
//! Made by AeroForger

use crate::light::{DirectionalLight, Light};
use crate::material::Material;
use crate::math::{Mat4, Vec3};
use crate::mesh::Mesh;
use crate::shadow_map::ShadowSettings;

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
    /// Whether this object casts a shadow (planar ground shadows and the
    /// directional shadow map).
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

    /// Center of the mesh's bounding box in world space.
    pub fn world_center(&self) -> Vec3 {
        self.transform.transform_point(self.mesh.center())
    }

    /// Whether this object is drawn in the blended transparent pass.
    pub fn is_transparent(&self) -> bool {
        self.material.is_transparent()
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

/// Distance fog, mixed toward `color` between `start` and `end`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    /// Fog color, linear RGB.
    pub color: [f32; 3],
    /// Distance from the camera where fog starts.
    pub start: f32,
    /// Distance where the surface is fully fogged.
    pub end: f32,
}

impl Fog {
    /// Fog that starts at `start` and fully covers geometry at `end`.
    pub fn new(color: [f32; 3], start: f32, end: f32) -> Self {
        Self { color, start, end }
    }

    /// The mix factor for a point `distance` away from the camera.
    pub fn factor(&self, distance: f32) -> f32 {
        if self.end <= self.start {
            return if distance >= self.end { 1.0 } else { 0.0 };
        }
        ((distance - self.start) / (self.end - self.start)).clamp(0.0, 1.0)
    }
}

/// A collection of objects, a primary sun, extra lights, fog, and ground
/// settings.
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
    /// All objects. Opaque objects draw in insertion order; transparent objects
    /// are sorted back to front and drawn last.
    pub objects: Vec<Object>,
    /// The primary directional light (also used for planar ground shadows and
    /// the directional shadow map).
    pub light: DirectionalLight,
    /// Additional point and spot lights, added on top of the sun.
    pub lights: Vec<Light>,
    /// Ground plane settings.
    pub ground: Ground,
    /// Optional distance fog.
    pub fog: Option<Fog>,
    /// Optional directional shadow mapping.
    pub shadow_map: Option<ShadowSettings>,
}

impl Scene {
    /// An empty scene with default light and ground.
    pub fn new() -> Self {
        Self::default()
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

    /// Adds a point or spot light and returns its index.
    pub fn add_light(&mut self, light: Light) -> usize {
        self.lights.push(light);
        self.lights.len() - 1
    }

    /// Enables distance fog (builder style).
    pub fn with_fog(mut self, fog: Fog) -> Self {
        self.fog = Some(fog);
        self
    }

    /// Enables shadow mapping with the given settings (builder style).
    pub fn with_shadow_map(mut self, settings: ShadowSettings) -> Self {
        self.shadow_map = Some(settings);
        self
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

    /// Removes every object, light, and effect, keeping the default sun and
    /// ground.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::PointLight;

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

    #[test]
    fn scene_accepts_extra_lights_and_effects() {
        let mut scene = Scene::new();
        let i = scene.add_light(Light::Point(PointLight::new(Vec3::ZERO, [1.0; 3])));
        assert_eq!(i, 0);
        scene = scene
            .with_fog(Fog::new([0.1, 0.1, 0.2], 5.0, 20.0))
            .with_shadow_map(ShadowSettings::default());
        assert!(scene.fog.is_some());
        assert!(scene.shadow_map.is_some());
    }

    #[test]
    fn fog_factor_clamps() {
        let fog = Fog::new([0.0; 3], 5.0, 10.0);
        assert_eq!(fog.factor(0.0), 0.0);
        assert!((fog.factor(7.5) - 0.5).abs() < 1e-6);
        assert_eq!(fog.factor(99.0), 1.0);
    }

    #[test]
    fn world_center_follows_transform() {
        let o = Object::new(Mesh::cube(), Material::default())
            .with_transform(Mat4::translation(Vec3::new(1.0, 2.0, 3.0)));
        let c = o.world_center();
        assert!((c.x - 1.0).abs() < 1e-5 && (c.y - 2.0).abs() < 1e-5);
    }
}
