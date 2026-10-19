// Frame-time probes for the full demo scene. These are `#[ignore]`d: run them
// with `cargo test --release -- --ignored --nocapture` when changing the
// rasterizer, and record the numbers in the README.

use aerogl::{AntiAliasing, Camera, DrawMode, Mat4, Material, Mesh, Object, Renderer, Scene, Vec3};
use std::time::Instant;

fn demo_scene() -> Scene {
    let mut scene = Scene::new();
    scene.add(
        Object::new(
            Mesh::cube(),
            Material::new([0.95, 0.42, 0.22]).with_specular(0.55, 64.0),
        )
        .with_transform(
            Mat4::translation(Vec3::new(-2.5, -0.35, 0.0)).multiply(&Mat4::rotation_y(0.9)),
        ),
    );
    scene.add(
        Object::new(
            Mesh::torus(32, 16, 0.9, 0.35),
            Material::new([0.30, 0.72, 1.0]).with_specular(0.7, 96.0),
        )
        .with_transform(
            Mat4::translation(Vec3::new(2.0, 0.9, 0.0))
                .multiply(&Mat4::rotation_x(1.6))
                .multiply(&Mat4::scale(Vec3::new(0.85, 0.85, 0.85))),
        ),
    );
    scene.add(
        Object::new(
            Mesh::sphere(18, 28),
            Material::new([0.78, 0.84, 0.92]).with_specular(0.2, 24.0),
        )
        .with_transform(Mat4::translation(Vec3::new(0.1, 0.55, 2.4)))
        .with_mode(DrawMode::Wireframe),
    );
    scene
}

#[test]
#[ignore = "writes PPM screenshots to the temp dir"]
fn dump_frames() {
    let scene = demo_scene();
    let camera = Camera::new(0.6, 0.35, 8.0);
    let modes = [
        ("none", AntiAliasing::None),
        ("msaa2", AntiAliasing::Msaa { factor: 2 }),
        ("smaa", AntiAliasing::Smaa),
    ];
    for (name, aa) in modes {
        let mut r = Renderer::new(800, 800);
        r.options.aa = aa;
        let pixels = r.render(&scene, &camera);
        let path = std::env::temp_dir().join(format!("aerogl-{name}.ppm"));
        let mut out = Vec::new();
        out.extend_from_slice(b"P6\n800 800\n255\n");
        for &p in pixels {
            out.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
        }
        std::fs::write(&path, out).expect("write screenshot");
        println!("wrote {}", path.display());
    }
}

#[test]
#[ignore = "benchmark"]
fn bench_demo_scene() {
    let scene = demo_scene();
    let camera = Camera::new(0.6, 0.35, 8.0);
    let tris: usize = scene.objects.iter().map(|o| o.mesh.tri_count()).sum();
    println!("demo scene: {tris} triangles");

    for (label, width, height) in [
        ("800x800", 800, 800),
        ("1920x1080", 1920, 1080),
        ("2560x1440", 2560, 1440),
    ] {
        for aa in [
            AntiAliasing::None,
            AntiAliasing::Msaa { factor: 2 },
            AntiAliasing::Smaa,
        ] {
            let mut r = Renderer::new(width, height);
            r.options.aa = aa;
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
                "{label} {aa:?} (active {:?}): {ms:.2} ms/frame ({:.0} fps)",
                r.active_aa(),
                1000.0 / ms
            );
        }
    }
}
