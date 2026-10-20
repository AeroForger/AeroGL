// Integration tests that exercise AeroGL exactly as a downstream user would:
// through the public API only, with no window.

use aerogl::{
    AntiAliasing, Camera, DirectionalLight, DrawMode, Fog, Light, Mat4, Material, Mesh, Object,
    PointLight, PostProcess, Renderer, Scene, ShadowSettings, Texture, ToneMapping, Vec3, Vertex,
};

fn scene_with_cube() -> Scene {
    let mut scene = Scene::new();
    scene.add(Object::new(
        Mesh::cube(),
        Material::new([0.9, 0.35, 0.2]).with_specular(0.5, 48.0),
    ));
    scene
}

#[test]
fn renders_a_full_frame_of_pixels() {
    let mut renderer = Renderer::new(200, 150);
    let background = renderer.options.background;
    let pixels = renderer.render(&scene_with_cube(), &Camera::default());

    assert_eq!(pixels.len(), 200 * 150);
    let lit = pixels.iter().filter(|&&p| p != background).count();
    assert!(lit > 1000, "expected a visible cube, got {lit} lit pixels");
}

#[test]
fn custom_geometry_from_vertices_and_indices() {
    // A single upright triangle, written out by hand.
    let mesh = Mesh::new(
        vec![
            Vertex::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            Vertex::new(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            Vertex::new(Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        ],
        vec![0, 1, 2],
    );
    assert_eq!(mesh.tri_count(), 1);

    let mut scene = Scene::new();
    scene.add(
        Object::new(mesh, Material::new([0.2, 1.0, 0.4]))
            .with_mode(DrawMode::ShadedWireframe)
            .with_transform(Mat4::translation(Vec3::new(0.0, 0.2, 0.0))),
    );

    let mut renderer = Renderer::new(96, 96);
    let background = renderer.options.background;
    let pixels = renderer.render(&scene, &Camera::new(0.0, 0.0, 4.0));
    assert!(pixels.iter().any(|&p| p != background), "triangle drawn");
}

#[test]
fn anti_aliasing_modes_are_selectable_at_runtime() {
    let scene = scene_with_cube();
    let camera = Camera::default();

    let mut renderer = Renderer::new(120, 90);
    let plain = renderer.render(&scene, &camera).to_vec();

    for aa in [
        AntiAliasing::Msaa { factor: 2 },
        AntiAliasing::Msaa { factor: 3 },
        AntiAliasing::Smaa,
    ] {
        renderer.options.aa = aa;
        let smoothed = renderer.render(&scene, &camera);
        assert_eq!(smoothed.len(), 120 * 90);
        assert_ne!(plain, smoothed, "{aa:?} should change the image");
    }
}

#[test]
fn shadows_composite_once_even_with_overlapping_meshes() {
    // Two identical cubes occupy the same space, so their shadow triangles
    // overlap exactly. If the shadow pass blended straight into the frame, the
    // overlap would darken twice. The mask keeps a single composite, so
    // enabling the second caster must not change a single pixel.
    let cube = || {
        Object::new(Mesh::cube(), Material::new([0.9, 0.35, 0.2]))
            .with_transform(Mat4::translation(Vec3::new(0.0, -0.3, 0.0)))
    };
    let mut one_caster = Scene::new();
    one_caster.add(cube());
    one_caster.add(cube().with_shadow(false));

    let mut two_casters = Scene::new();
    two_casters.add(cube());
    two_casters.add(cube());

    let camera = Camera::default();
    let mut r = Renderer::new(160, 120);
    r.options.aa = AntiAliasing::None;
    let one = r.render(&one_caster, &camera).to_vec();
    let two = r.render(&two_casters, &camera).to_vec();

    assert_eq!(one, two, "overlapping shadow geometry must not stack");
    let background = r.options.background;
    assert!(one.iter().any(|&p| p != background), "scene was empty");
}

#[test]
fn front_faces_are_shaded_not_inside_out() {
    // With a light pointing straight at the camera and a red cube facing it,
    // the center pixel must be the lit front face. Rendering the back faces
    // instead (an inverted winding test) leaves the cube dark, because the
    // inside of the far face receives no diffuse light.
    let mut scene = Scene::new();
    scene.light =
        DirectionalLight::new(Vec3::new(0.0, 0.0, -1.0), [1.0, 1.0, 1.0]).with_ambient(0.1);
    scene.ground.shadows = false;
    scene.ground.grid = false;
    scene.add(Object::new(
        Mesh::cube(),
        Material::new([0.9, 0.2, 0.15])
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    ));

    let mut renderer = Renderer::new(64, 64);
    renderer.options.aa = AntiAliasing::None;
    let camera = Camera::new(std::f32::consts::FRAC_PI_2, 0.0, 5.0);
    let pixels = renderer.render(&scene, &camera);

    let center = pixels[32 * 64 + 32];
    let red = (center >> 16) & 0xFF;
    assert!(
        red > 180,
        "front face should be fully lit, got red {red} (inside-out rendering shows the unlit back faces)"
    );
}

#[test]
fn renderer_can_be_resized() {
    let mut renderer = Renderer::new(64, 64);
    let _ = renderer.render(&scene_with_cube(), &Camera::default());
    renderer.resize(128, 48);
    let pixels = renderer.render(&scene_with_cube(), &Camera::default());
    assert_eq!(pixels.len(), 128 * 48);
    assert_eq!((renderer.width(), renderer.height()), (128, 48));
}

#[test]
fn ground_grid_stays_stable_across_camera_angles() {
    // Regression for the "attempt to add/subtract with overflow" panic in the
    // banded line rasterizer. A near-horizontal grid line makes the per-band
    // step range divide by a tiny slope, so the step index can land in the
    // billions. That has to clamp instead of overflowing i32 inside a worker
    // thread, which surfaced as "a scoped thread panicked".
    let scene = scene_with_cube();
    assert!(
        scene.ground.grid,
        "the grid must be on for this test to bite"
    );
    let mut renderer = Renderer::new(160, 160);
    for step in 0..180 {
        let yaw = step as f32 * std::f32::consts::TAU / 180.0;
        for pitch in [0.2f32, 0.45] {
            let camera = Camera::new(yaw, pitch, 7.0);
            let pixels = renderer.render(&scene, &camera);
            assert_eq!(pixels.len(), 160 * 160);
        }
    }
}

#[test]
fn triangle_soup_builds_a_renderable_mesh() {
    // from_triangles derives normals from winding, so this also checks that the
    // derived normal points the same way as the supplied vertex order.
    let mesh = Mesh::from_triangles(&[(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
    )]);
    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.tri_count(), 1);
    for v in &mesh.vertices {
        // Face normal points toward +Z for this winding.
        assert!(
            v.normal.z > 0.99,
            "normal should face the camera: {:?}",
            v.normal
        );
    }
}

fn luminance_sum(pixels: &[u32]) -> u64 {
    pixels
        .iter()
        .map(|&p| ((p >> 16) & 0xFF) as u64 + ((p >> 8) & 0xFF) as u64 + (p & 0xFF) as u64)
        .sum()
}

fn bare_scene(material: Material) -> Scene {
    let mut scene = Scene::new();
    scene.ground.grid = false;
    scene.ground.shadows = false;
    scene.add(Object::new(Mesh::cube(), material));
    scene
}

#[test]
fn textures_change_the_surface_appearance() {
    let plain = bare_scene(
        Material::new([1.0, 1.0, 1.0])
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    );
    let textured = bare_scene(
        Material::new([1.0, 1.0, 1.0])
            .with_texture(Texture::checker(64, 4, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    );

    let mut renderer = Renderer::new(96, 96);
    renderer.options.aa = AntiAliasing::None;
    let camera = Camera::default();
    let plain_pixels = renderer.render(&plain, &camera).to_vec();
    let textured_pixels = renderer.render(&textured, &camera).to_vec();

    assert_ne!(plain_pixels, textured_pixels, "texture must be sampled");
    let colors: std::collections::HashSet<u32> = textured_pixels.iter().copied().collect();
    assert!(colors.len() > 10, "checker should produce many shades");
    assert_eq!(
        Texture::checker(8, 2, [1.0; 3], [0.0; 3]).width(),
        8,
        "public texture accessors stay usable"
    );
}

#[test]
fn point_lights_add_light_on_top_of_the_sun() {
    let mut scene = bare_scene(
        Material::new([1.0, 1.0, 1.0])
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    );
    // Kill the sun so only the added point light can illuminate the cube.
    scene.light.color = [0.0, 0.0, 0.0];
    scene.light.ambient = 0.0;

    let mut renderer = Renderer::new(96, 96);
    renderer.options.aa = AntiAliasing::None;
    let camera = Camera::default();
    let unlit = luminance_sum(renderer.render(&scene, &camera));

    scene.add_light(Light::Point(
        PointLight::new(Vec3::new(0.0, 0.0, 4.0), [1.0, 1.0, 1.0])
            .with_intensity(2.0)
            .with_range(10.0),
    ));
    let lit = luminance_sum(renderer.render(&scene, &camera));
    assert!(lit > unlit + 10_000, "point light should brighten the cube");
}

#[test]
fn alpha_blending_shows_the_background_through_glass() {
    let opaque = bare_scene(
        Material::new([1.0, 0.0, 0.0])
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    );
    let glass = bare_scene(
        Material::new([1.0, 0.0, 0.0])
            .with_alpha(0.4)
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    );

    let mut renderer = Renderer::new(96, 96);
    renderer.options.aa = AntiAliasing::None;
    let background = renderer.options.background;
    let camera = Camera::default();
    let opaque_pixels = renderer.render(&opaque, &camera).to_vec();
    let glass_pixels = renderer.render(&glass, &camera).to_vec();

    let center = 48 * 96 + 48;
    let solid = opaque_pixels[center];
    let blended = glass_pixels[center];
    assert_ne!(solid, blended, "alpha must blend with the background");
    let red_solid = (solid >> 16) & 0xFF;
    let red_blended = (blended >> 16) & 0xFF;
    let blue_background = background & 0xFF;
    let blue_blended = blended & 0xFF;
    assert!(red_blended < red_solid, "glass should not be fully opaque");
    assert!(
        blue_blended > blue_background / 2,
        "background should show through the glass"
    );
}

#[test]
fn shadow_mapping_darkens_the_floor_under_a_cube() {
    let build = |shadows: bool| {
        let mut scene = Scene::new();
        scene.ground.grid = false;
        scene.ground.shadows = false;
        if shadows {
            scene.shadow_map = Some(ShadowSettings::new(256).with_darkness(0.7));
        }
        scene.add(
            Object::new(
                Mesh::plane(12.0, 20),
                Material::new([0.8, 0.8, 0.8])
                    .with_specular(0.0, 1.0)
                    .with_rim(0.0),
            )
            .with_transform(Mat4::translation(Vec3::new(0.0, -1.35, 0.0))),
        );
        scene.add(
            Object::new(
                Mesh::cube(),
                Material::new([0.9, 0.5, 0.2])
                    .with_specular(0.0, 1.0)
                    .with_rim(0.0),
            )
            .with_transform(Mat4::translation(Vec3::new(0.0, 0.0, 0.0))),
        );
        scene
    };

    let mut renderer = Renderer::new(160, 120);
    renderer.options.aa = AntiAliasing::None;
    let camera = Camera::new(0.35, 0.75, 5.5);
    let lit = renderer.render(&build(false), &camera).to_vec();
    let shadowed = renderer.render(&build(true), &camera).to_vec();

    let changed = lit.iter().zip(&shadowed).filter(|(a, b)| a != b).count();
    assert!(
        changed > 200,
        "shadow map should alter the floor: {changed}"
    );
    assert!(
        luminance_sum(&shadowed) < luminance_sum(&lit),
        "shadows must remove light, not add it"
    );
}

#[test]
fn obj_files_load_and_render() {
    let obj = "\
# a single triangle facing +Z
v -1 -1 0
v  1 -1 0
v  0  1 0
vt 0 0
vt 1 0
vt 0.5 1
f 1/1 2/2 3/3
";
    let mesh = Mesh::from_obj(obj).expect("valid OBJ");
    assert_eq!(mesh.tri_count(), 1);
    assert_eq!(mesh.vertices[1].uv, [1.0, 0.0]);

    let mut scene = Scene::new();
    scene.ground.grid = false;
    scene.ground.shadows = false;
    scene.add(Object::new(
        mesh,
        Material::new([0.2, 1.0, 0.4])
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    ));
    let mut renderer = Renderer::new(96, 96);
    renderer.options.aa = AntiAliasing::None;
    let background = renderer.options.background;
    let pixels = renderer.render(&scene, &Camera::default());
    assert!(pixels.iter().any(|&p| p != background), "OBJ should render");
    assert!(Mesh::from_obj("v 0 0 0\nf 1 2 3\n").is_err());
}

#[test]
fn fog_fades_distant_surfaces_toward_the_fog_color() {
    let mut scene = bare_scene(
        Material::new([1.0, 1.0, 1.0])
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    );
    scene.fog = None;
    let mut renderer = Renderer::new(96, 96);
    renderer.options.aa = AntiAliasing::None;
    // Push the cube away from the camera, then make the fog dense enough to
    // reach it.
    scene.object_mut(0).unwrap().transform = Mat4::translation(Vec3::new(0.0, 0.0, -4.0));
    let clear = renderer.render(&scene, &Camera::default()).to_vec();

    scene.fog = Some(Fog::new([0.9, 0.1, 0.1], 2.0, 9.0));
    let foggy = renderer.render(&scene, &Camera::default()).to_vec();
    assert_ne!(clear, foggy, "fog should change the image");
    let clear_luma = luminance_sum(&clear);
    let foggy_luma = luminance_sum(&foggy);
    assert!(foggy_luma != clear_luma);
}

#[test]
fn post_process_grades_the_final_image() {
    let scene = scene_with_cube();
    let camera = Camera::default();
    let mut renderer = Renderer::new(96, 96);
    renderer.options.aa = AntiAliasing::None;
    let plain = renderer.render(&scene, &camera).to_vec();

    renderer.options.post = PostProcess::default()
        .with_exposure(1.4)
        .with_tone_mapping(ToneMapping::Aces)
        .with_vignette(0.4);
    let graded = renderer.render(&scene, &camera).to_vec();
    assert_ne!(plain, graded, "post-processing should change the output");

    renderer.options.post = PostProcess::default();
    let identity = renderer.render(&scene, &camera).to_vec();
    assert_eq!(
        plain, identity,
        "the default post process must be an identity transform"
    );
}

#[test]
fn geometry_beyond_the_far_plane_is_clipped() {
    let mut scene = bare_scene(
        Material::new([1.0, 0.0, 0.0])
            .with_specular(0.0, 1.0)
            .with_rim(0.0),
    );
    scene.object_mut(0).unwrap().transform = Mat4::translation(Vec3::new(0.0, 0.0, -200.0));

    let mut renderer = Renderer::new(64, 64);
    renderer.options.aa = AntiAliasing::None;
    let background = renderer.options.background;
    let pixels = renderer.render(&scene, &Camera::default());
    assert!(
        pixels.iter().all(|&p| p == background),
        "geometry past the far plane must not render"
    );
}
