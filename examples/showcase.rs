//! Render frames for the AeroGL demo video.
//!
//! This is a video-capture helper, not part of the library API. It renders
//! deterministic frames at 1080x1920 (9:16) and writes raw RGB to stdout so
//! ffmpeg can encode without a giant intermediate file.
//!
//! Examples:
//!   cargo run --release --example showcase -- --still hero --out video/assets/hero.ppm
//!   cargo run --release --example showcase -- --aa off --out video/assets/aa_off.ppm
//!   cargo run --release --example showcase -- --bench
//!   cargo run --release --example showcase -- --shot 01 | ffmpeg -f rawvideo \
//!       -pix_fmt rgb24 -s 1080x1920 -r 30 -i - video/assets/shot01.mp4

use aerogl::pipeline::{self, ShadowMask};
use aerogl::{
    AntiAliasing, Camera, DrawMode, Framebuffer, Mat4, Material, Mesh, Object, Renderer, Scene,
    Vec3,
};
use std::io::{self, BufWriter, Write};
use std::time::Instant;

const W: usize = 1080;
const H: usize = 1920;
const FPS: usize = 30;
const BACKGROUND: u32 = 0x0008_0d12;
const MASK_TINT: u32 = 0x0058_c4dd;

const LAND_W: usize = 1920;
const LAND_H: usize = 1080;

const CUBE_X: f32 = -1.55;
const TORUS_X: f32 = 1.55;
const GROUND_Y: f32 = -1.35;

// ---------------------------------------------------------------------------
// Materials and transforms
// ---------------------------------------------------------------------------

fn material_cube() -> Material {
    Material::new([0.95, 0.42, 0.22]).with_specular(0.55, 64.0)
}

fn material_torus() -> Material {
    Material::new([0.30, 0.72, 1.0]).with_specular(0.7, 96.0)
}

fn material_sphere() -> Material {
    Material::new([0.78, 0.84, 0.92]).with_specular(0.2, 24.0)
}

fn cube_transform(t: f32) -> Mat4 {
    Mat4::translation(Vec3::new(CUBE_X, -0.35, 0.0)).multiply(&Mat4::rotation_y(t * 0.7))
}

fn torus_transform(t: f32) -> Mat4 {
    Mat4::translation(Vec3::new(TORUS_X, 0.2, 0.0))
        .multiply(&Mat4::rotation_x(1.0 + t * 0.6))
        .multiply(&Mat4::rotation_y(t * 0.4))
        .multiply(&Mat4::scale(Vec3::new(0.85, 0.85, 0.85)))
}

fn sphere_transform(t: f32) -> Mat4 {
    Mat4::translation(Vec3::new(-0.2, 2.9, 0.2)).multiply(&Mat4::rotation_y(-t * 0.5))
}

// ---------------------------------------------------------------------------
// Cameras
// ---------------------------------------------------------------------------

fn hero_camera() -> Camera {
    let mut camera = Camera::new(0.55, 0.20, 7.8);
    camera.target = Vec3::new(0.0, 1.25, 0.0);
    camera
}

fn hero_camera_orbit(yaw: f32) -> Camera {
    let mut camera = hero_camera();
    camera.yaw = yaw;
    camera
}

fn close_camera(distance: f32) -> Camera {
    let mut camera = Camera::new(0.75, 0.18, distance);
    camera.target = Vec3::new(CUBE_X + 1.0, -0.2, 1.0);
    camera
}

fn ground_camera() -> Camera {
    let mut camera = Camera::new(0.35, 0.06, 9.2);
    camera.target = Vec3::new(0.0, -0.75, 0.0);
    camera
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

fn hero_scene(
    t: f32,
    cube_mode: DrawMode,
    torus_mode: DrawMode,
    sphere_mode: DrawMode,
    shadows: bool,
) -> Scene {
    let mut scene = Scene::new();
    scene.ground.y = GROUND_Y;
    scene.ground.grid = true;
    scene.ground.shadows = shadows;
    scene.add(
        Object::new(Mesh::cube(), material_cube())
            .with_transform(cube_transform(t))
            .with_mode(cube_mode),
    );
    scene.add(
        Object::new(Mesh::torus(32, 16, 0.9, 0.35), material_torus())
            .with_transform(torus_transform(t))
            .with_mode(torus_mode),
    );
    scene.add(
        Object::new(Mesh::sphere(18, 28), material_sphere())
            .with_transform(sphere_transform(t))
            .with_mode(sphere_mode),
    );
    scene
}

fn cube_only_scene(t: f32, mode: DrawMode, shadows: bool) -> Scene {
    let mut scene = Scene::new();
    scene.ground.y = GROUND_Y;
    scene.ground.grid = false;
    scene.ground.shadows = shadows;
    scene.add(
        Object::new(Mesh::cube(), material_cube())
            .with_transform(cube_transform(t))
            .with_mode(mode),
    );
    scene
}

/// The landscape scene used for the AA benchmark and the AA lab panel.
/// This matches `tests/perf.rs` so the video numbers line up with the repo.
fn landscape_scene() -> Scene {
    let mut scene = Scene::new();
    scene.ground.shadows = true;
    scene.add(Object::new(Mesh::cube(), material_cube()).with_transform(
        Mat4::translation(Vec3::new(-2.5, -0.35, 0.0)).multiply(&Mat4::rotation_y(0.9)),
    ));
    scene.add(
        Object::new(Mesh::torus(32, 16, 0.9, 0.35), material_torus()).with_transform(
            Mat4::translation(Vec3::new(2.0, 0.9, 0.0))
                .multiply(&Mat4::rotation_x(1.6))
                .multiply(&Mat4::scale(Vec3::new(0.85, 0.85, 0.85))),
        ),
    );
    scene.add(
        Object::new(Mesh::sphere(18, 28), material_sphere())
            .with_transform(Mat4::translation(Vec3::new(0.1, 0.55, 2.4)))
            .with_mode(DrawMode::Wireframe),
    );
    scene
}

fn landscape_camera() -> Camera {
    Camera::new(0.6, 0.35, 8.0)
}

// ---------------------------------------------------------------------------
// Frame writing
// ---------------------------------------------------------------------------

fn pack_rgb(buf: &mut [u8], pixels: &[u32]) {
    for (i, &p) in pixels.iter().enumerate() {
        buf[i * 3] = (p >> 16) as u8;
        buf[i * 3 + 1] = (p >> 8) as u8;
        buf[i * 3 + 2] = p as u8;
    }
}

fn write_ppm(path: &str, width: usize, height: usize, pixels: &[u32]) -> io::Result<()> {
    let mut out = BufWriter::new(std::fs::File::create(path)?);
    write!(out, "P6\n{width} {height}\n255\n")?;
    for &p in pixels {
        out.write_all(&[(p >> 16) as u8, (p >> 8) as u8, p as u8])?;
    }
    out.flush()
}

/// Renders the shadow mask over a scene as cyan pixels, for the mask flash.
fn mask_pixels(scene: &Scene, camera: &Camera) -> Vec<u32> {
    let mut fb = Framebuffer::new(W, H);
    let mut shadow = ShadowMask::new();
    {
        let mut target = fb.target();
        pipeline::render_scene_clear(&mut target, scene, camera, &mut shadow, BACKGROUND);
    }
    let mut pixels = fb.pixels().to_vec();
    for (idx, p) in pixels.iter_mut().enumerate() {
        if shadow.is_shadowed(idx) {
            *p = MASK_TINT;
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// Stills
// ---------------------------------------------------------------------------

fn still_scene(name: &str) -> (Scene, Camera, AntiAliasing) {
    match name {
        "hero" => (
            hero_scene(
                0.0,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                true,
            ),
            hero_camera(),
            AntiAliasing::Smaa,
        ),
        "wire" => (
            hero_scene(
                0.0,
                DrawMode::Wireframe,
                DrawMode::Wireframe,
                DrawMode::Wireframe,
                false,
            ),
            hero_camera(),
            AntiAliasing::None,
        ),
        "shaded" => (
            hero_scene(
                0.0,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                false,
            ),
            hero_camera(),
            AntiAliasing::None,
        ),
        "ground" => (
            hero_scene(
                0.0,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                true,
            ),
            ground_camera(),
            AntiAliasing::None,
        ),
        "close" => (
            cube_only_scene(0.0, DrawMode::Shaded, true),
            close_camera(3.4),
            AntiAliasing::None,
        ),
        "mask" => (
            hero_scene(
                0.0,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                true,
            ),
            ground_camera(),
            AntiAliasing::None,
        ),
        "final" => (
            hero_scene(
                0.0,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                true,
            ),
            hero_camera(),
            AntiAliasing::Smaa,
        ),
        _ => {
            eprintln!("unknown still: {name}");
            std::process::exit(2);
        }
    }
}

fn render_still(name: &str, out_path: &str) -> io::Result<()> {
    let (scene, camera, aa) = still_scene(name);
    if name == "mask" {
        let pixels = mask_pixels(&scene, &camera);
        return write_ppm(out_path, W, H, &pixels);
    }
    let mut renderer = Renderer::new(W, H);
    renderer.options.background = BACKGROUND;
    renderer.options.aa = aa;
    renderer.options.max_internal_pixels = Some(4_000_000);
    let pixels = renderer.render(&scene, &camera);
    write_ppm(out_path, W, H, pixels)
}

// ---------------------------------------------------------------------------
// AA stills (1920x1080)
// ---------------------------------------------------------------------------

fn render_aa(mode: &str, out_path: &str) -> io::Result<()> {
    let scene = landscape_scene();
    let camera = landscape_camera();
    let mut renderer = Renderer::new(LAND_W, LAND_H);
    renderer.options.background = BACKGROUND;
    match mode {
        "off" => {
            renderer.options.aa = AntiAliasing::None;
        }
        "smaa" => {
            renderer.options.aa = AntiAliasing::Smaa;
        }
        "msaa2" => {
            renderer.options.aa = AntiAliasing::Msaa { factor: 2 };
            renderer.options.max_internal_pixels = None;
        }
        "msaa2-default" => {
            renderer.options.aa = AntiAliasing::Msaa { factor: 2 };
            renderer.options.max_internal_pixels = Some(4_000_000);
        }
        "msaa4" => {
            renderer.options.aa = AntiAliasing::Msaa { factor: 4 };
            renderer.options.max_internal_pixels = None;
        }
        _ => {
            eprintln!("unknown aa mode: {mode}");
            std::process::exit(2);
        }
    }
    let active = renderer.active_aa();
    let pixels = renderer.render(&scene, &camera);
    write_ppm(out_path, LAND_W, LAND_H, pixels)?;
    eprintln!("wrote {out_path} ({LAND_W}x{LAND_H}, requested {mode}, active {active:?})");
    Ok(())
}

// ---------------------------------------------------------------------------
// Shot sequences
// ---------------------------------------------------------------------------

fn shot_length(shot: &str) -> Option<usize> {
    Some(match shot {
        "01" => 3 * FPS,
        "02" => 4 * FPS,
        "03" => 5 * FPS,
        "04" => 4 * FPS,
        "05" => 5 * FPS,
        "06" => 5 * FPS,
        "12" => 4 * FPS,
        "13" => 3 * FPS,
        _ => return None,
    })
}

/// Returns the scene, camera, AA mode, and whether this frame is a mask flash.
fn shot_state(shot: &str, i: usize, n: usize) -> (Scene, Camera, AntiAliasing, bool) {
    let p = if n > 1 {
        i as f32 / (n - 1) as f32
    } else {
        0.0
    };
    let t = i as f32 / FPS as f32;

    match shot {
        "01" => {
            let scene = cube_only_scene(t, DrawMode::Wireframe, false);
            let camera = close_camera(3.2);
            (scene, camera, AntiAliasing::None, false)
        }
        "02" => {
            let scene = hero_scene(
                t,
                DrawMode::Wireframe,
                DrawMode::Wireframe,
                DrawMode::Wireframe,
                false,
            );
            let camera = hero_camera_orbit(0.55 + 0.20 * p);
            (scene, camera, AntiAliasing::None, false)
        }
        "03" => {
            let scene = hero_scene(
                t,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                false,
            );
            let mut camera = hero_camera();
            camera.distance = 8.8 - 1.6 * p;
            (scene, camera, AntiAliasing::None, false)
        }
        "04" => {
            let scene = hero_scene(
                t,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                false,
            );
            let camera = hero_camera_orbit(0.55 + 0.40 * p);
            (scene, camera, AntiAliasing::None, false)
        }
        "05" => {
            let mut scene = hero_scene(
                t,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                true,
            );
            // Shadows off for the first 2 seconds, then on. A short mask flash
            // shows the offscreen mask pass before the composite.
            scene.ground.shadows = i >= 2 * FPS;
            let camera = ground_camera();
            let mask_flash = (2 * FPS..2 * FPS + 9).contains(&i);
            (scene, camera, AntiAliasing::None, mask_flash)
        }
        "06" => {
            let scene = cube_only_scene(t, DrawMode::Shaded, true);
            let camera = close_camera(3.4 - 0.4 * p);
            (scene, camera, AntiAliasing::None, false)
        }
        "12" => {
            let scene = hero_scene(
                t,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                true,
            );
            let camera = hero_camera_orbit(0.55 + 0.07 * p);
            (scene, camera, AntiAliasing::Smaa, false)
        }
        "13" => {
            let scene = hero_scene(
                t,
                DrawMode::Shaded,
                DrawMode::Shaded,
                DrawMode::Wireframe,
                true,
            );
            let camera = hero_camera_orbit(0.55 + 0.20 * p);
            (scene, camera, AntiAliasing::Smaa, false)
        }
        _ => {
            eprintln!("unknown shot: {shot}");
            std::process::exit(2);
        }
    }
}

fn render_shot(shot: &str) -> io::Result<()> {
    let n = shot_length(shot).unwrap_or_else(|| {
        eprintln!("unknown shot: {shot}");
        std::process::exit(2);
    });
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 22, stdout.lock());
    let mut bytes = vec![0u8; W * H * 3];
    let mut renderer = Renderer::new(W, H);
    renderer.options.background = BACKGROUND;

    for i in 0..n {
        let (scene, camera, aa, mask_flash) = shot_state(shot, i, n);
        renderer.options.aa = aa;
        renderer.options.max_internal_pixels = Some(4_000_000);
        if mask_flash {
            let pixels = mask_pixels(&scene, &camera);
            pack_rgb(&mut bytes, &pixels);
        } else {
            let pixels = renderer.render(&scene, &camera);
            pack_rgb(&mut bytes, pixels);
        }
        out.write_all(&bytes)?;
    }
    out.flush()?;
    eprintln!("shot {shot}: {n} frames at {W}x{H}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn bench_one(label: &str, width: usize, height: usize, aa: AntiAliasing, cap: Option<usize>) {
    let scene = if width == LAND_W {
        landscape_scene()
    } else {
        hero_scene(
            0.0,
            DrawMode::Shaded,
            DrawMode::Shaded,
            DrawMode::Wireframe,
            true,
        )
    };
    let camera = if width == LAND_W {
        landscape_camera()
    } else {
        hero_camera()
    };
    let mut r = Renderer::new(width, height);
    r.options.background = BACKGROUND;
    r.options.aa = aa;
    r.options.max_internal_pixels = cap;
    for _ in 0..10 {
        r.render(&scene, &camera);
    }
    let frames = 60;
    let start = Instant::now();
    for _ in 0..frames {
        r.render(&scene, &camera);
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    println!(
        "{label}: {ms:.2} ms/frame ({:.0} fps), active {:?}",
        1000.0 / ms,
        r.active_aa()
    );
}

fn run_bench() {
    println!("# 1920x1080 landscape (AA lab panel)");
    bench_one("off", LAND_W, LAND_H, AntiAliasing::None, None);
    bench_one("smaa", LAND_W, LAND_H, AntiAliasing::Smaa, None);
    bench_one(
        "msaa2-forced",
        LAND_W,
        LAND_H,
        AntiAliasing::Msaa { factor: 2 },
        None,
    );
    bench_one(
        "msaa2-default",
        LAND_W,
        LAND_H,
        AntiAliasing::Msaa { factor: 2 },
        Some(4_000_000),
    );
    bench_one(
        "msaa4-forced",
        LAND_W,
        LAND_H,
        AntiAliasing::Msaa { factor: 4 },
        None,
    );
    println!("# 1080x1920 vertical (hero plates)");
    bench_one("off", W, H, AntiAliasing::None, None);
    bench_one("smaa", W, H, AntiAliasing::Smaa, None);
    bench_one(
        "msaa2-default",
        W,
        H,
        AntiAliasing::Msaa { factor: 2 },
        Some(4_000_000),
    );
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!(
        "usage:\n  \
         --still <hero|wire|shaded|ground|close|mask|final> --out <path.ppm>\n  \
         --aa <off|smaa|msaa2|msaa2-default|msaa4> --out <path.ppm>\n  \
         --shot <01|02|03|04|05|06|12|13>  (raw RGB to stdout)\n  \
         --bench"
    );
    std::process::exit(2);
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    match args[1].as_str() {
        "--still" => {
            let name = args.get(2).unwrap_or_else(|| usage());
            let out = args
                .windows(2)
                .find(|w| w[0] == "--out")
                .map(|w| w[1].clone())
                .unwrap_or_else(|| usage());
            render_still(name, &out)?;
            eprintln!("wrote {out} ({W}x{H})");
        }
        "--aa" => {
            let mode = args.get(2).unwrap_or_else(|| usage());
            let out = args
                .windows(2)
                .find(|w| w[0] == "--out")
                .map(|w| w[1].clone())
                .unwrap_or_else(|| usage());
            render_aa(mode, &out)?;
        }
        "--shot" => {
            let shot = args.get(2).unwrap_or_else(|| usage());
            render_shot(shot)?;
        }
        "--bench" => run_bench(),
        _ => usage(),
    }
    Ok(())
}
