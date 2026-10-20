//! Headless demo of the full pipeline: textures, multiple lights, shadow
//! mapping, sorted transparency, fog, and tone mapping.
//!
//! Usage:
//!   cargo run --release --example pipeline -- out.ppm
//!
//! Writes a binary PPM (P6) that any image viewer can open. PPM keeps the
//! example dependency-free; the same pixels can go straight into a window,
//! encoder, or GPU texture upload.
//!
//! Made by AeroForger

use aerogl::{
    AntiAliasing, Camera, Fog, Light, Mat4, Material, Mesh, Object, PointLight, PostProcess,
    Renderer, Scene, ShadowSettings, SpotLight, Texture, ToneMapping, Vec3, WrapMode,
};
use std::io::{self, BufWriter, Write};
use std::sync::Arc;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const GROUND_Y: f32 = -1.35;

fn main() -> io::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "pipeline.ppm".to_string());

    let mut scene = Scene::new();
    scene.ground.y = GROUND_Y;
    scene.ground.grid = false;
    scene.ground.shadows = false;
    scene.shadow_map = Some(
        ShadowSettings::new(1536)
            .with_bias(0.003)
            .with_darkness(0.52),
    );

    // Subdivided checker floor. UVs cover [0,1], so the texture's 16x16 cells
    // become 16 squares across the plane.
    let floor_texture = Arc::new(
        Texture::checker(512, 16, [0.22, 0.27, 0.35], [0.11, 0.14, 0.19])
            .with_wrap(WrapMode::Repeat),
    );
    scene.add(
        Object::new(
            Mesh::plane(20.0, 28),
            Material::new([1.0, 1.0, 1.0])
                .with_shared_texture(floor_texture)
                .with_specular(0.10, 32.0)
                .with_rim(0.0),
        )
        .with_transform(Mat4::translation(Vec3::new(0.0, GROUND_Y - 0.002, 0.0)))
        .with_shadow(false),
    );

    let cube_texture = Arc::new(Texture::checker(
        256,
        4,
        [0.96, 0.52, 0.24],
        [0.55, 0.18, 0.07],
    ));
    scene.add(
        Object::new(
            Mesh::cube(),
            Material::new([1.0, 1.0, 1.0])
                .with_shared_texture(cube_texture)
                .with_specular(0.55, 72.0),
        )
        .with_transform(
            Mat4::translation(Vec3::new(-2.8, -0.35, -0.4)).multiply(&Mat4::rotation_y(0.55)),
        ),
    );

    scene.add(
        Object::new(
            Mesh::torus(40, 20, 1.0, 0.38),
            Material::new([0.24, 0.66, 1.0])
                .with_specular(0.55, 80.0)
                .with_rim(0.15),
        )
        .with_transform(
            Mat4::translation(Vec3::new(2.3, 1.0, -0.6))
                .multiply(&Mat4::rotation_x(1.15))
                .multiply(&Mat4::rotation_y(-0.35)),
        ),
    );

    // Glass sphere: sorted into the transparent pass, blended over everything
    // opaque behind it.
    scene.add(
        Object::new(
            Mesh::sphere(24, 36),
            Material::new([0.58, 0.84, 1.0])
                .with_alpha(0.36)
                .with_specular(1.0, 180.0)
                .with_rim(0.4),
        )
        .with_transform(Mat4::translation(Vec3::new(0.35, 0.55, 2.8))),
    );

    scene.add_light(Light::Point(
        PointLight::new(Vec3::new(4.0, 2.4, 1.5), [1.0, 0.48, 0.20])
            .with_intensity(1.8)
            .with_range(8.0),
    ));
    scene.add_light(Light::Spot(
        SpotLight::new(
            Vec3::new(0.5, 5.8, 1.0),
            Vec3::new(-0.1, -1.0, -0.15),
            [0.42, 0.72, 1.0],
        )
        .with_intensity(2.0)
        .with_range(12.0)
        .with_cone(18.0, 34.0),
    ));
    scene.fog = Some(Fog::new([0.035, 0.055, 0.09], 12.0, 34.0));

    let mut camera = Camera::new(0.62, 0.28, 9.5);
    camera.target = Vec3::new(0.0, 0.35, 0.0);
    camera.fov_y_deg = 55.0;

    let mut renderer = Renderer::new(WIDTH, HEIGHT);
    renderer.options.aa = AntiAliasing::Smaa;
    renderer.options.background = 0x0008_0d14;
    renderer.options.post = PostProcess::default()
        .with_exposure(1.15)
        .with_tone_mapping(ToneMapping::Aces)
        .with_saturation(1.08)
        .with_vignette(0.20)
        .with_gamma(1.0);

    let start = std::time::Instant::now();
    let pixels = renderer.render(&scene, &camera);
    let elapsed = start.elapsed();

    write_ppm(&out, pixels, WIDTH, HEIGHT)?;
    let triangles: usize = scene.objects.iter().map(|o| o.mesh.tri_count()).sum();
    println!(
        "wrote {out} ({}x{}, {triangles} triangles, {} lights, shadow map on, {:.1} ms)",
        WIDTH,
        HEIGHT,
        1 + scene.lights.len(),
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Writes the packed `0x00RRGGBB` buffer as binary PPM.
fn write_ppm(path: &str, pixels: &[u32], width: usize, height: usize) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut out = BufWriter::new(file);
    write!(out, "P6\n{width} {height}\n255\n")?;
    let mut row = Vec::with_capacity(width * 3);
    for chunk in pixels.chunks(width) {
        row.clear();
        for &p in chunk {
            row.push(((p >> 16) & 0xFF) as u8);
            row.push(((p >> 8) & 0xFF) as u8);
            row.push((p & 0xFF) as u8);
        }
        out.write_all(&row)?;
    }
    out.flush()
}
