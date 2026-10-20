//! Frame rendering: shared preparation, banded passes, and transparency.

use crate::camera::Camera;
use crate::framebuffer::Target;
use crate::math::{Mat4, Vec3, finite_or};
use crate::scene::Scene;
use crate::shadow_map::ShadowMap;

use super::clip::ClipVertex;
use super::grid::{GridSegment, draw_grid_segments, prepare_grid};
use super::object::{prepare_object, render_prepared_object};
use super::shade::RenderContext;
use super::shadow::{
    ScreenBounds, ShadowBand, ShadowMask, ShadowTri, composite_shadow, prepare_shadows,
};

/// Extra state for [`render_scene_with`].
///
/// All fields are optional. Planar shadows allocate a temporary mask when the
/// scene enables them and no mask is supplied, and a shadow map is allocated
/// only when [`Scene::shadow_map`] is set.
#[derive(Default)]
pub struct SceneRender<'a> {
    background: Option<u32>,
    shadow_mask: Option<&'a mut ShadowMask>,
    shadow_map: Option<&'a mut ShadowMap>,
}

impl<'a> SceneRender<'a> {
    /// No background clear, no caller-owned buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears color to `background` before drawing.
    pub fn background(mut self, background: u32) -> Self {
        self.background = Some(background);
        self
    }

    /// Reuses a caller-owned planar shadow mask.
    pub fn shadow_mask(mut self, mask: &'a mut ShadowMask) -> Self {
        self.shadow_mask = Some(mask);
        self
    }

    /// Reuses a caller-owned directional shadow map.
    pub fn shadow_map(mut self, map: &'a mut ShadowMap) -> Self {
        self.shadow_map = Some(map);
        self
    }
}

/// Everything that is the same for every band: lit vertices, projected
/// shadows and projected grid segments.
struct PreparedFrame {
    objects: Vec<Vec<ClipVertex>>,
    shadow_tris: Vec<ShadowTri>,
    shadow_bounds: Option<ScreenBounds>,
    grid: Vec<GridSegment>,
}

/// Builds the per-frame data shared by all render bands.
fn prepare_frame(
    scene: &Scene,
    view_proj: Mat4,
    near: f32,
    width: usize,
    height: usize,
    ctx: &RenderContext<'_>,
) -> PreparedFrame {
    let ground_y = scene.ground.sanitized_y();
    let objects = scene
        .objects
        .iter()
        .map(|object| prepare_object(object, view_proj, ctx))
        .collect();
    let (shadow_tris, shadow_bounds) = if scene.ground.shadows {
        prepare_shadows(
            scene,
            view_proj,
            near,
            width,
            height,
            ground_y,
            ctx.light_dir,
        )
    } else {
        (Vec::new(), None)
    };
    let grid = if scene.ground.grid {
        prepare_grid(view_proj, near, width, height, ground_y, 10)
    } else {
        Vec::new()
    };
    PreparedFrame {
        objects,
        shadow_tris,
        shadow_bounds,
        grid,
    }
}

/// Draws transparent objects back to front, with depth writes disabled.
fn render_transparent(
    target: &mut Target,
    scene: &Scene,
    prep: &PreparedFrame,
    near: f32,
    camera_pos: Vec3,
) {
    let mut indices: Vec<usize> = (0..scene.objects.len())
        .filter(|&i| scene.objects[i].is_transparent())
        .collect();
    if indices.is_empty() {
        return;
    }
    indices.sort_by(|&a, &b| {
        let da = scene.objects[a].world_center().sub(camera_pos);
        let db = scene.objects[b].world_center().sub(camera_pos);
        // Farthest first.
        db.dot(db).total_cmp(&da.dot(da))
    });
    let previous_depth_write = target.depth_write;
    target.set_depth_write(false);
    for i in indices {
        render_prepared_object(target, &scene.objects[i], &prep.objects[i], near);
    }
    target.set_depth_write(previous_depth_write);
}

/// Renders an entire scene: ground grid, planar shadows, opaque objects, then
/// sorted transparent objects.
///
/// When the target is a full screen and large enough, the grid, shadow mask,
/// shadow composite and opaque object passes all run band by band on worker
/// threads. Every band owns distinct color, depth and mask rows, so no pass
/// needs a lock. Transparent objects are drawn afterwards, back to front.
pub fn render_scene(target: &mut Target, scene: &Scene, camera: &Camera, shadow: &mut ShadowMask) {
    render_scene_with(
        target,
        scene,
        camera,
        SceneRender::new().shadow_mask(shadow),
    );
}

/// Like [`render_scene`], but also clears color and depth per band before
/// drawing.
///
/// Clearing inside the band pass keeps each band's rows hot in cache and avoids
/// a separate full-frame clear. This is what [`crate::Renderer`] uses.
pub fn render_scene_clear(
    target: &mut Target,
    scene: &Scene,
    camera: &Camera,
    shadow: &mut ShadowMask,
    background: u32,
) {
    render_scene_with(
        target,
        scene,
        camera,
        SceneRender::new()
            .background(background)
            .shadow_mask(shadow),
    );
}

/// Like [`render_scene_clear`], but reusing a caller-owned [`ShadowMap`].
///
/// Pass a persistent map when rendering frames in a loop so the depth buffer
/// is not reallocated every frame. The map is only touched when
/// [`Scene::shadow_map`] is set.
pub fn render_scene_clear_with_shadow_map(
    target: &mut Target,
    scene: &Scene,
    camera: &Camera,
    shadow: &mut ShadowMask,
    background: u32,
    shadow_map: &mut ShadowMap,
) {
    render_scene_with(
        target,
        scene,
        camera,
        SceneRender::new()
            .background(background)
            .shadow_mask(shadow)
            .shadow_map(shadow_map),
    );
}

/// Renders a scene with explicit options and optional caller-owned buffers.
///
/// Use this instead of [`render_scene`] when you want to skip the planar shadow
/// mask, supply a persistent [`ShadowMap`], or combine both.
pub fn render_scene_with(
    target: &mut Target,
    scene: &Scene,
    camera: &Camera,
    settings: SceneRender<'_>,
) {
    if !target.is_valid() {
        return;
    }
    let SceneRender {
        background,
        shadow_mask,
        shadow_map,
    } = settings;
    let mut shadow_map_scratch = shadow_map;
    let mut owned_mask;
    let mut shadow = match shadow_mask {
        Some(mask) => Some(mask),
        None if scene.ground.shadows => {
            owned_mask = ShadowMask::new();
            Some(&mut owned_mask)
        }
        None => None,
    };
    let view = camera.view();
    let proj = camera.projection(target.width as f32 / target.height as f32);
    let view_proj = proj.multiply(&view);
    let near = finite_or(camera.near, 0.1).max(1e-4);

    // Render the light's depth pass before any shading samples it.
    let mut owned_shadow: ShadowMap;
    let shadow_ref = if let Some(settings) = scene.shadow_map {
        let map: &mut ShadowMap = match shadow_map_scratch.as_mut() {
            Some(map) => map,
            None => {
                owned_shadow = ShadowMap::new(settings);
                &mut owned_shadow
            }
        };
        map.render(scene);
        Some(&*map)
    } else {
        None
    };

    let mut ctx = RenderContext::from_scene(camera.eye(), scene);
    ctx.shadow = shadow_ref;
    let prep = prepare_frame(scene, view_proj, near, target.width, target.height, &ctx);
    let threads = if target.band_rows() == target.height {
        crate::parallel::threads_for(target.width * target.height, 200_000)
    } else {
        1
    };
    let camera_pos = camera.eye();
    let opaque: Vec<usize> = (0..scene.objects.len())
        .filter(|&i| !scene.objects[i].is_transparent())
        .collect();

    if threads <= 1 {
        if let Some(bg) = background {
            target.clear(bg);
        }
        draw_grid_segments(target, &prep.grid);
        if scene.ground.shadows
            && let Some(shadow) = shadow.as_deref_mut()
        {
            shadow.prepare(target.width, target.height);
            shadow.cover_tris(&prep.shadow_tris);
            shadow.adopt_bounds_f32(prep.shadow_bounds);
            composite_shadow(target, shadow, scene.ground.shadow_tint);
        }
        for &i in &opaque {
            render_prepared_object(target, &scene.objects[i], &prep.objects[i], near);
        }
        render_transparent(target, scene, &prep, near, camera_pos);
        return;
    }

    if scene.ground.shadows
        && let Some(shadow) = shadow.as_deref_mut()
    {
        shadow.prepare(target.width, target.height);
    }
    let tint = scene.ground.shadow_tint;
    let bands = target.split_bands(threads);
    let mut shadow_bands: Vec<Option<ShadowBand<'_>>> = match shadow.as_deref_mut() {
        Some(shadow) if scene.ground.shadows => {
            shadow.split_bands(threads).into_iter().map(Some).collect()
        }
        _ => (0..bands.len()).map(|_| None).collect(),
    };
    let PreparedFrame {
        objects,
        shadow_tris,
        shadow_bounds,
        grid,
    } = &prep;
    let opaque = &opaque;

    std::thread::scope(|scope| {
        for (mut band, shadow_band) in bands.into_iter().zip(shadow_bands.drain(..)) {
            scope.spawn(move || {
                if let Some(bg) = background {
                    band.clear(bg);
                }
                draw_grid_segments(&mut band, grid);
                if let Some(mut shadow_band) = shadow_band {
                    shadow_band.cover(shadow_tris);
                    shadow_band.set_bounds(*shadow_bounds);
                    shadow_band.composite_into(band.color, tint);
                }
                for &i in opaque {
                    render_prepared_object(&mut band, &scene.objects[i], &objects[i], near);
                }
            });
        }
    });

    if scene.ground.shadows
        && let Some(shadow) = shadow
    {
        shadow.adopt_bounds_f32(*shadow_bounds);
    }

    render_transparent(target, scene, &prep, near, camera_pos);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::framebuffer::{Framebuffer, Target};
    use crate::material::Material;
    use crate::math::{Mat4, Vec3};
    use crate::scene::{Ground, Object, Scene};

    #[test]
    fn render_scene_on_bands_matches_the_full_frame() {
        // Target::band is public API. A band must compose shadows and draw the
        // grid against the full screen, not against its own row range.
        let (width, height) = (64, 48);
        let background = 0x102030;
        let camera = Camera::new(0.6, 0.35, 8.0);
        let mut scene = Scene::new();
        scene.add(
            Object::new(crate::mesh::Mesh::cube(), Material::new([0.9, 0.4, 0.2]))
                .with_transform(Mat4::translation(Vec3::new(0.0, 0.8, 0.0))),
        );

        let mut full = Framebuffer::new(width, height);
        let mut mask = ShadowMask::new();
        {
            let mut target = full.target();
            render_scene_clear(&mut target, &scene, &camera, &mut mask, background);
        }

        let split = height / 2;
        let mut banded = Framebuffer::new(width, height);
        let (top_color, bottom_color) = banded.color.split_at_mut(split * width);
        let (top_depth, bottom_depth) = banded.depth.split_at_mut(split * width);
        {
            let mut target = Target::band(width, height, 0, split, top_color, top_depth);
            render_scene_clear(&mut target, &scene, &camera, &mut mask, background);
        }
        {
            let mut target = Target::band(
                width,
                height,
                split,
                height - split,
                bottom_color,
                bottom_depth,
            );
            render_scene_clear(&mut target, &scene, &camera, &mut mask, background);
        }

        assert_eq!(full.color, banded.color, "banded frame differs");
    }

    #[test]
    fn shadow_tint_defaults_come_from_ground() {
        let g = Ground::default();
        assert!(g.shadows, "shadows on by default");
        assert!(g.grid, "grid on by default");
    }

    #[test]
    fn transparent_pass_preserves_depth_write_state() {
        let mut scene = Scene::new();
        scene.ground.shadows = false;
        scene.ground.grid = false;
        scene.add(Object::new(
            crate::mesh::Mesh::cube(),
            Material::new([1.0, 0.0, 0.0]).with_alpha(0.4),
        ));
        let camera = Camera::default();
        let mut fb = Framebuffer::new(32, 32);
        let mut mask = ShadowMask::new();
        let mut target = fb.target();
        target.set_depth_write(false);
        render_scene_clear(&mut target, &scene, &camera, &mut mask, 0);
        assert!(
            !target.depth_write,
            "the transparent pass must restore the caller's state"
        );
    }

    #[test]
    fn render_scene_with_can_allocate_a_temporary_shadow_mask() {
        let (width, height) = (48, 40);
        let background = 0x102030;
        let camera = Camera::default();
        let mut scene = Scene::new();
        scene.add(Object::new(
            crate::mesh::Mesh::cube(),
            Material::new([0.9, 0.4, 0.2]),
        ));

        let mut with_mask = Framebuffer::new(width, height);
        let mut mask = ShadowMask::new();
        {
            let mut target = with_mask.target();
            render_scene_clear(&mut target, &scene, &camera, &mut mask, background);
        }

        let mut with_temporary = Framebuffer::new(width, height);
        {
            let mut target = with_temporary.target();
            render_scene_with(
                &mut target,
                &scene,
                &camera,
                SceneRender::new().background(background),
            );
        }

        assert_eq!(with_mask.color, with_temporary.color);
    }
}
