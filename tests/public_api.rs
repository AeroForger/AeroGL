// Integration tests that exercise AeroGL exactly as a downstream user would:
// through the public API only, with no window.

use aerogl::{
    AntiAliasing, Camera, DirectionalLight, DrawMode, Mat4, Material, Mesh, Object, Renderer,
    Scene, Vec3, Vertex,
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
