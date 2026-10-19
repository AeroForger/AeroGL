//! Headless rendering: no window, no platform dependencies.
//!
//! Run with `cargo run --example offscreen --no-default-features` and open
//! `aerogl.ppm` in any image viewer. This is the pattern to use on a server, in
//! tests, in Wasm (copy the buffer into a canvas), or with your own window
//! library.
//!
//! Made by AeroForger

use aerogl::{AntiAliasing, Camera, Mat4, Material, Mesh, Object, Renderer, Scene, Vec3};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> std::io::Result<()> {
    let width = 640;
    let height = 480;

    let mut renderer = Renderer::new(width, height);
    renderer.options.aa = AntiAliasing::Msaa { factor: 2 };
    renderer.options.background = 0x0010_1820;

    let mut scene = Scene::new();

    scene.add(
        Object::new(
            Mesh::cube(),
            Material::new([0.95, 0.45, 0.20]).with_specular(0.6, 64.0),
        )
        .with_transform(Mat4::translation(Vec3::new(-1.8, -0.35, 0.0))),
    );

    scene.add(
        Object::new(
            Mesh::torus(40, 20, 1.0, 0.38),
            Material::new([0.25, 0.70, 1.0]).with_specular(0.8, 96.0),
        )
        .with_transform(
            Mat4::translation(Vec3::new(1.7, 0.1, 0.0))
                .multiply(&Mat4::rotation_x(1.1))
                .multiply(&Mat4::rotation_y(0.6)),
        ),
    );

    let camera = Camera::new(0.6, 0.32, 9.0);
    let pixels = renderer.render(&scene, &camera);

    // Write a binary PPM (P6): 3 bytes per pixel, no image crate needed.
    let mut out = BufWriter::new(File::create("aerogl.ppm")?);
    write!(out, "P6\n{width} {height}\n255\n")?;
    for &p in pixels {
        out.write_all(&[(p >> 16) as u8, (p >> 8) as u8, p as u8])?;
    }
    out.flush()?;

    println!("wrote aerogl.ppm ({width}x{height})");
    Ok(())
}
