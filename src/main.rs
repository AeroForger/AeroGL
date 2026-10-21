// AeroGL demo: textured surfaces, multiple lights, real shadow mapping, fog,
// tone mapping, and sorted alpha blending.
//
// Controls:
//   C     cycle object render mode (shaded / wireframe / both)
//   A     cycle anti-aliasing (off / SMAA / MSAA 2x, 4 samples)
//   G     toggle ground grid
//   S     toggle directional shadow mapping
//   F     toggle distance fog
//   T     toggle the graded post-process (ACES + vignette)
//   Space pause / resume animation
//   drag  orbit camera, scroll to zoom, Esc to quit
//
// The same scene can be rendered headless. See examples/offscreen.rs and
// examples/pipeline.rs.
//
// Made by AeroForger

use aerogl::{
    AeroglWindow, AntiAliasing, Camera, DrawMode, Fog, Light, Mat4, Material, Mesh, Object,
    PointLight, PostProcess, Scene, ShadowSettings, SpotLight, Texture, ToneMapping, Vec3,
};
use minifb::{Key, MouseButton, MouseMode};
use std::sync::Arc;
use std::time::Instant;

const GROUND_Y: f32 = -1.35;

const RENDER_MODES: [DrawMode; 3] = [
    DrawMode::Shaded,
    DrawMode::Wireframe,
    DrawMode::ShadedWireframe,
];

fn mode_label(mode: DrawMode) -> &'static str {
    match mode {
        DrawMode::Shaded => "shaded",
        DrawMode::Wireframe => "wireframe",
        DrawMode::ShadedWireframe => "shaded + wireframe",
    }
}

fn aa_label(aa: AntiAliasing) -> String {
    match aa {
        AntiAliasing::None => "AA off".to_string(),
        AntiAliasing::Msaa { factor } => format!("MSAA {factor}x ({} samples)", factor * factor),
        AntiAliasing::Smaa => "SMAA".to_string(),
    }
}

fn next_aa(aa: AntiAliasing) -> AntiAliasing {
    match aa {
        // Start with the cheap option: SMAA is about half the cost of 4x MSAA.
        AntiAliasing::None => AntiAliasing::Smaa,
        AntiAliasing::Smaa => AntiAliasing::Msaa { factor: 2 },
        AntiAliasing::Msaa { .. } => AntiAliasing::None,
    }
}

fn graded_post() -> PostProcess {
    PostProcess::default()
        .with_exposure(1.25)
        .with_tone_mapping(ToneMapping::Aces)
        .with_saturation(1.08)
        .with_vignette(0.22)
}

fn main() {
    let mut app = AeroglWindow::new("AeroGL", 1000, 720).expect("could not open a window");
    let mut scene = Scene::new();
    scene.ground.y = GROUND_Y;
    scene.ground.grid = false;
    // The shadow map replaces the old planar shadows in this demo.
    scene.ground.shadows = false;
    scene.shadow_map = Some(
        ShadowSettings::new(1024)
            .with_bias(0.0035)
            .with_darkness(0.62),
    );

    // A checkerboard floor: the mesh is subdivided so per-vertex lighting and
    // the shadow map can vary across it.
    let floor_tex = Arc::new(
        Texture::checker(256, 16, [0.20, 0.24, 0.31], [0.10, 0.12, 0.16])
            .with_wrap(aerogl::WrapMode::Repeat),
    );
    scene.add(
        Object::new(
            Mesh::plane(18.0, 24),
            Material::new([1.0, 1.0, 1.0])
                .with_shared_texture(floor_tex)
                .with_specular(0.08, 24.0)
                .with_rim(0.0),
        )
        .with_transform(Mat4::translation(Vec3::new(0.0, GROUND_Y - 0.002, 0.0)))
        .with_shadow(false),
    );

    let cube_tex = Arc::new(Texture::checker(
        128,
        4,
        [0.95, 0.48, 0.22],
        [0.55, 0.20, 0.08],
    ));
    let cube = scene.add(
        Object::new(
            Mesh::cube(),
            Material::new([1.0, 1.0, 1.0])
                .with_shared_texture(cube_tex)
                .with_specular(0.45, 64.0),
        )
        .with_transform(Mat4::translation(Vec3::new(-2.6, -0.35, 0.0))),
    );
    let torus = scene.add(
        Object::new(
            Mesh::torus(32, 16, 0.9, 0.35),
            Material::new([0.30, 0.72, 1.0]).with_specular(0.85, 96.0),
        )
        .with_transform(Mat4::translation(Vec3::new(2.1, 0.9, 0.0))),
    );
    // Transparent glass sphere: drawn back to front after the opaque pass.
    let glass = scene.add(
        Object::new(
            Mesh::sphere(20, 30),
            Material::new([0.55, 0.82, 1.0])
                .with_alpha(0.38)
                .with_specular(1.0, 160.0)
                .with_rim(0.35),
        )
        .with_transform(Mat4::translation(Vec3::new(0.2, 0.55, 2.6))),
    );

    // Two local lights on top of the sun: a warm orbiting point light and a
    // cool spot from above.
    let point_index = scene.add_light(Light::Point(
        PointLight::new(Vec3::new(3.5, 2.0, 0.0), [1.0, 0.52, 0.22])
            .with_intensity(2.6)
            .with_range(9.0),
    ));
    let spot_index = scene.add_light(Light::Spot(
        SpotLight::new(
            Vec3::new(0.0, 5.5, 0.5),
            Vec3::new(0.0, -1.0, -0.1),
            [0.40, 0.70, 1.0],
        )
        .with_intensity(2.4)
        .with_range(12.0)
        .with_cone(16.0, 30.0),
    ));
    scene.fog = Some(Fog::new([0.03, 0.05, 0.08], 11.0, 30.0));

    let mut camera = Camera::new(0.6, 0.30, 9.0);
    camera.target = Vec3::new(0.0, 0.4, 0.0);
    app.renderer_mut().options.post = graded_post();

    let mut last_mouse: Option<(f32, f32)> = None;
    let mut elapsed = 0.0f32;
    let mut paused = false;
    let mut mode = 0usize;
    let mut graded = true;

    let mut last_time = Instant::now();
    let mut fps = 0.0f32;
    let mut scene_ms = 0.0f32;
    let mut post_ms = 0.0f32;
    let mut present_ms = 0.0f32;
    let mut title_timer = 0.0f32;
    let show_timings = std::env::var_os("AEROGL_TIMINGS").is_some();

    while app.is_open() {
        let now = Instant::now();
        let dt = (now - last_time).as_secs_f32().min(0.1); // clamp hitches
        last_time = now;
        fps += (1.0 / dt.max(1e-3) - fps) * 0.1; // smoothed

        // --- input (one mutable borrow of the window for the whole block) ---
        let mut cycle_aa = false;
        let mut toggle_post = false;
        {
            let window = app.window_mut();
            if let Some((_, y)) = window.get_scroll_wheel() {
                camera.zoom(y);
            }
            let pressed = |k: Key| window.is_key_pressed(k, minifb::KeyRepeat::No);
            if pressed(Key::C) {
                mode = (mode + 1) % RENDER_MODES.len();
                let m = RENDER_MODES[mode];
                if let Some(o) = scene.object_mut(cube) {
                    o.mode = m;
                }
                if let Some(o) = scene.object_mut(torus) {
                    o.mode = m;
                }
            }
            if pressed(Key::A) {
                cycle_aa = true;
            }
            if pressed(Key::G) {
                scene.ground.grid = !scene.ground.grid;
            }
            if pressed(Key::S) {
                scene.shadow_map = if scene.shadow_map.is_some() {
                    None
                } else {
                    Some(
                        ShadowSettings::new(1024)
                            .with_bias(0.0035)
                            .with_darkness(0.62),
                    )
                };
            }
            if pressed(Key::F) {
                scene.fog = if scene.fog.is_some() {
                    None
                } else {
                    Some(Fog::new([0.03, 0.05, 0.08], 11.0, 30.0))
                };
            }
            if pressed(Key::T) {
                graded = !graded;
                toggle_post = true;
            }
            if pressed(Key::Space) {
                paused = !paused;
            }

            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
                if window.get_mouse_down(MouseButton::Left) {
                    if let Some((lx, ly)) = last_mouse {
                        camera.orbit(mx - lx, my - ly);
                    }
                    last_mouse = Some((mx, my));
                } else {
                    last_mouse = None;
                }
            }
        }

        if cycle_aa {
            let next = next_aa(app.renderer().options.aa);
            app.renderer_mut().options.aa = next;
        }
        if toggle_post {
            app.renderer_mut().options.post = if graded {
                graded_post()
            } else {
                PostProcess::default()
            };
        }

        // --- animate (fixed step so motion speed is frame-rate independent) ---
        if !paused {
            elapsed += dt.min(0.05);
        }
        if let Some(o) = scene.object_mut(cube) {
            // Y-only spin: the cube keeps its bottom face on the ground instead
            // of tilting through it.
            o.transform = Mat4::translation(Vec3::new(-2.6, -0.35, 0.0))
                .multiply(&Mat4::rotation_y(elapsed * 0.7));
        }
        if let Some(o) = scene.object_mut(torus) {
            o.transform = Mat4::translation(Vec3::new(2.1, 0.9, 0.0))
                .multiply(&Mat4::rotation_x(1.0 + elapsed * 0.6))
                .multiply(&Mat4::rotation_y(elapsed * 0.4))
                .multiply(&Mat4::scale(Vec3::new(0.85, 0.85, 0.85)));
        }
        if let Some(o) = scene.object_mut(glass) {
            o.transform =
                Mat4::translation(Vec3::new(0.2, 0.65 + (elapsed * 1.4).sin() * 0.15, 2.6))
                    .multiply(&Mat4::rotation_y(-elapsed * 0.5));
        }

        // Orbit the warm point light around the scene.
        let angle = elapsed * 0.8;
        scene.lights[point_index] = Light::Point(
            PointLight::new(
                Vec3::new(angle.cos() * 3.8, 2.1, angle.sin() * 3.8),
                [1.0, 0.52, 0.22],
            )
            .with_intensity(2.6)
            .with_range(9.0),
        );
        // Sweep the cool spot slowly across the floor.
        scene.lights[spot_index] = Light::Spot(
            SpotLight::new(
                Vec3::new(angle.sin() * 2.0, 5.5, angle.cos() * 1.5),
                Vec3::new(-angle.sin() * 0.35, -1.0, -0.2),
                [0.40, 0.70, 1.0],
            )
            .with_intensity(2.4)
            .with_range(12.0)
            .with_cone(16.0, 30.0),
        );

        // --- present ---
        app.draw(&scene, &camera).expect("present failed");

        let timings = app.last_timings();
        scene_ms += (timings.scene_ms - scene_ms) * 0.1;
        post_ms += (timings.post_ms - post_ms) * 0.1;
        present_ms += (timings.present_ms - present_ms) * 0.1;

        // Rebuilding the window title is a platform round-trip: throttle it.
        title_timer += dt;
        if title_timer >= 0.25 {
            title_timer = 0.0;
            let selected = app.renderer().options.aa;
            let active = app.renderer().active_aa();
            let aa_text = if active == selected {
                aa_label(selected)
            } else {
                format!("{} (capped from {})", aa_label(active), aa_label(selected))
            };
            let status = format!(
                "AeroGL - {}x{} | {} | {} | {} tris | shadows {} | fog {} | {:.0} fps | scene {:.1} post {:.1} present {:.1} ms | C A G S F T",
                app.renderer().width(),
                app.renderer().height(),
                mode_label(RENDER_MODES[mode]),
                aa_text,
                scene
                    .objects
                    .iter()
                    .map(|o| o.mesh.tri_count())
                    .sum::<usize>(),
                if scene.shadow_map.is_some() {
                    "on"
                } else {
                    "off"
                },
                if scene.fog.is_some() { "on" } else { "off" },
                fps,
                scene_ms,
                post_ms,
                present_ms
            );
            app.set_title(&status);
            if show_timings {
                eprintln!("{status}");
            }
        }
    }
}
