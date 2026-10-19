// AeroGL demo: a spinning cube and torus (Blinn-Phong Gouraud shading) plus a
// wireframe sphere, over a depth-tested ground grid with planar shadows.
//
// Controls:
//   C     cycle object render mode (shaded / wireframe / both)
//   A     cycle anti-aliasing (off / MSAA x4 / SMAA)
//   G     toggle ground grid
//   S     toggle shadows
//   Space pause / resume animation
//   drag  orbit camera, scroll to zoom, Esc to quit
//
// The same scene can be rendered headless. See examples/offscreen.rs.
//
// Made by AeroForger

use aerogl::{
    AeroglWindow, AntiAliasing, Camera, DrawMode, Mat4, Material, Mesh, Object, Scene, Vec3,
};
use minifb::{Key, MouseButton, MouseMode};
use std::time::Instant;

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
        AntiAliasing::None => AntiAliasing::Msaa { factor: 2 },
        AntiAliasing::Msaa { .. } => AntiAliasing::Smaa,
        AntiAliasing::Smaa => AntiAliasing::None,
    }
}

fn main() {
    let mut app = AeroglWindow::new("AeroGL", 800, 800).expect("could not open a window");
    let mut scene = Scene::new();

    let cube = scene.add(
        Object::new(
            Mesh::cube(),
            Material::new([0.95, 0.42, 0.22]).with_specular(0.55, 64.0),
        )
        .with_transform(Mat4::translation(Vec3::new(-2.5, -0.35, 0.0))),
    );
    let torus = scene.add(
        Object::new(
            Mesh::torus(32, 16, 0.9, 0.35),
            Material::new([0.30, 0.72, 1.0]).with_specular(0.7, 96.0),
        )
        .with_transform(Mat4::translation(Vec3::new(2.0, 0.9, 0.0))),
    );
    let sphere = scene.add(
        Object::new(
            Mesh::sphere(18, 28),
            Material::new([0.78, 0.84, 0.92]).with_specular(0.2, 24.0),
        )
        .with_transform(Mat4::translation(Vec3::new(0.1, 0.55, 2.4)))
        .with_mode(DrawMode::Wireframe),
    );

    let mut camera = Camera::new(0.6, 0.35, 8.0);
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut elapsed = 0.0f32;
    let mut paused = false;
    let mut mode = 0usize;

    let mut last_time = Instant::now();
    let mut fps = 0.0f32;
    let mut title_timer = 0.0f32;

    while app.is_open() {
        let now = Instant::now();
        let dt = (now - last_time).as_secs_f32().min(0.1); // clamp hitches
        last_time = now;
        fps += (1.0 / dt.max(1e-3) - fps) * 0.1; // smoothed

        // --- input (one mutable borrow of the window for the whole block) ---
        let mut cycle_aa = false;
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
                scene.ground.shadows = !scene.ground.shadows;
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

        // --- animate (fixed step so motion speed is frame-rate independent) ---
        if !paused {
            elapsed += dt.min(0.05);
        }
        if let Some(o) = scene.object_mut(cube) {
            // Y-only spin: the cube keeps its bottom face on the ground instead
            // of tilting through it.
            o.transform = Mat4::translation(Vec3::new(-2.5, -0.35, 0.0))
                .multiply(&Mat4::rotation_y(elapsed * 0.7));
        }
        if let Some(o) = scene.object_mut(torus) {
            o.transform = Mat4::translation(Vec3::new(2.0, 0.9, 0.0))
                .multiply(&Mat4::rotation_x(1.0 + elapsed * 0.6))
                .multiply(&Mat4::rotation_y(elapsed * 0.4))
                .multiply(&Mat4::scale(Vec3::new(0.85, 0.85, 0.85)));
        }
        if let Some(o) = scene.object_mut(sphere) {
            o.transform = Mat4::translation(Vec3::new(0.1, 0.55, 2.4))
                .multiply(&Mat4::rotation_y(-elapsed * 0.5));
        }

        // --- present ---
        app.draw(&scene, &camera).expect("present failed");

        // Rebuilding the window title is a platform round-trip: throttle it.
        title_timer += dt;
        if title_timer >= 0.25 {
            title_timer = 0.0;
            let aa = app.renderer().options.aa;
            app.set_title(&format!(
                "AeroGL - {}, {} | {} tris | {:.0} fps | C mode, A AA, G grid, S shadows",
                mode_label(RENDER_MODES[mode]),
                aa_label(aa),
                scene
                    .objects
                    .iter()
                    .map(|o| o.mesh.tri_count())
                    .sum::<usize>(),
                fps
            ));
        }
    }
}
