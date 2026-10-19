//! The reusable renderer: owns buffers, applies the AA strategy, returns pixels.
//! Made by AeroForger

use crate::aa::{AntiAliasing, SmaaScratch, apply_smaa_with_scratch, resolve_downsample};
use crate::camera::Camera;
use crate::framebuffer::Framebuffer;
use crate::pipeline::{ShadowMask, render_scene};
use crate::scene::Scene;

/// Options that affect how a frame is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderOptions {
    /// Anti-aliasing strategy.
    pub aa: AntiAliasing,
    /// Background clear color, packed `0x00RRGGBB`.
    pub background: u32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            aa: AntiAliasing::None,
            background: 0x001A_2B3C,
        }
    }
}

/// Renders scenes into an owned pixel buffer.
///
/// The renderer is backend-agnostic: call [`Renderer::render`] and hand the
/// resulting pixels to any windowing library, image encoder, or Wasm canvas.
/// With the `window` feature enabled, `AeroglWindow` is a ready-made `minifb`
/// frontend.
///
/// ```
/// use aerogl::{Camera, Material, Mesh, Renderer, Scene};
/// let mut renderer = Renderer::new(64, 64);
/// let mut scene = Scene::new();
/// scene.add_mesh(Mesh::cube(), Material::new([1.0, 0.5, 0.2]));
/// let pixels = renderer.render(&scene, &Camera::default());
/// assert_eq!(pixels.len(), 64 * 64);
/// ```
pub struct Renderer {
    width: usize,
    height: usize,
    /// Frame options; change these at any time.
    pub options: RenderOptions,
    internal: Framebuffer,
    output: Framebuffer,
    shadow: ShadowMask,
    smaa: SmaaScratch,
}

impl Renderer {
    /// Creates a renderer producing `width` x `height` pixels.
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0, "renderer needs a positive size");
        Self {
            width,
            height,
            options: RenderOptions::default(),
            internal: Framebuffer::new(width, height),
            output: Framebuffer::new(width, height),
            shadow: ShadowMask::new(),
            smaa: SmaaScratch::new(),
        }
    }

    /// Output width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Output height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Resizes the output (and internal buffers on the next frame).
    pub fn resize(&mut self, width: usize, height: usize) {
        assert!(width > 0 && height > 0);
        self.width = width;
        self.height = height;
        self.output.resize(width, height);
    }

    /// Renders one frame and returns the packed `0x00RRGGBB` pixels.
    ///
    /// The slice is valid until the next [`Renderer::render`] call.
    pub fn render(&mut self, scene: &Scene, camera: &Camera) -> &[u32] {
        let scale = self.options.aa.scale() as usize;
        if scale == 1 {
            // No supersampling: render directly into the output framebuffer.
            // This avoids copying the whole image through `internal` first.
            self.output.resize(self.width, self.height);
            self.output.clear(self.options.background);
            {
                let mut target = self.output.target();
                render_scene(&mut target, scene, camera, &mut self.shadow);
            }
            if self.options.aa == AntiAliasing::Smaa {
                apply_smaa_with_scratch(
                    &mut self.output.color,
                    self.width,
                    self.height,
                    &mut self.smaa,
                );
            }
        } else {
            let (iw, ih) = (self.width * scale, self.height * scale);
            self.internal.resize(iw, ih);
            self.internal.clear(self.options.background);
            {
                let mut target = self.internal.target();
                render_scene(&mut target, scene, camera, &mut self.shadow);
            }

            self.output.resize(self.width, self.height);
            resolve_downsample(
                &self.internal.color,
                &mut self.output.color,
                self.width,
                self.height,
                scale,
            );
        }
        &self.output.color
    }

    /// The last rendered pixels.
    pub fn pixels(&self) -> &[u32] {
        &self.output.color
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::mesh::Mesh;
    use crate::scene::Object;

    fn test_scene() -> Scene {
        let mut scene = Scene::new();
        scene.add(Object::new(Mesh::cube(), Material::new([0.9, 0.3, 0.2])));
        scene
    }

    fn non_background(pixels: &[u32], background: u32) -> usize {
        pixels.iter().filter(|&&p| p != background).count()
    }

    #[test]
    fn renders_something() {
        let mut r = Renderer::new(160, 120);
        let bg = r.options.background;
        let px = r.render(&test_scene(), &Camera::default());
        assert!(non_background(px, bg) > 100, "scene should cover pixels");
    }

    #[test]
    fn msaa_and_smaa_produce_different_pixels_than_none() {
        let scene = test_scene();
        let camera = Camera::default();

        let mut plain = Renderer::new(160, 120);
        let none = plain.render(&scene, &camera).to_vec();

        let mut msaa = Renderer::new(160, 120);
        msaa.options.aa = AntiAliasing::Msaa { factor: 2 };
        let mx = msaa.render(&scene, &camera).to_vec();

        let mut smaa = Renderer::new(160, 120);
        smaa.options.aa = AntiAliasing::Smaa;
        let sx = smaa.render(&scene, &camera).to_vec();

        assert_ne!(none, mx, "MSAA should change edge pixels");
        assert_ne!(none, sx, "SMAA should change edge pixels");
        assert_eq!(mx.len(), 160 * 120);
        assert_eq!(sx.len(), 160 * 120);
    }

    #[test]
    fn resolution_is_changeable_at_runtime() {
        let mut r = Renderer::new(32, 32);
        let _ = r.render(&test_scene(), &Camera::default());
        r.resize(64, 16);
        let px = r.render(&test_scene(), &Camera::default());
        assert_eq!(px.len(), 64 * 16);
    }

    /// Frame-time probe (not run by default):
    /// `cargo test --release bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_frames() {
        let scene = test_scene();
        let camera = Camera::default();
        for aa in [
            AntiAliasing::None,
            AntiAliasing::Msaa { factor: 2 },
            AntiAliasing::Smaa,
        ] {
            let mut r = Renderer::new(800, 800);
            r.options.aa = aa;
            for _ in 0..5 {
                r.render(&scene, &camera); // warm up
            }
            let start = std::time::Instant::now();
            let frames = 60;
            for _ in 0..frames {
                r.render(&scene, &camera);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / frames as f64;
            println!("{aa:?}: {ms:.2} ms/frame ({:.0} fps)", 1000.0 / ms);
        }
    }
}
