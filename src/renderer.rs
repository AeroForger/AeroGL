//! The reusable renderer: owns buffers, applies the AA strategy, returns pixels.
//! Made by AeroForger

use crate::aa::{AntiAliasing, SmaaScratch, apply_smaa_with_scratch, resolve_downsample};
use crate::camera::Camera;
use crate::framebuffer::Framebuffer;
use crate::pipeline::{ShadowMask, render_scene_clear_with_shadow_map};
use crate::post::PostProcess;
use crate::scene::Scene;
use crate::shadow_map::{ShadowMap, ShadowSettings};
use std::time::Instant;

/// Time spent in the last frame's render stages, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderTimings {
    /// Scene drawing and anti-aliasing.
    pub scene_ms: f32,
    /// Post-processing.
    pub post_ms: f32,
}

/// Options that affect how a frame is produced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptions {
    /// Anti-aliasing strategy.
    pub aa: AntiAliasing,
    /// Background clear color, packed `0x00RRGGBB`.
    pub background: u32,
    /// Largest supersampled frame, in pixels, before [`AntiAliasing::Msaa`]
    /// falls back to [`AntiAliasing::Smaa`].
    ///
    /// Supersampling costs scale with the square of the factor, so a 2x buffer
    /// at 1440p is four times the pixels of the window. That is fine for a
    /// still frame but can halve the frame rate. The default cap is a 4K
    /// buffer; set this to `None` to let MSAA use whatever the mode asks for.
    pub max_internal_pixels: Option<usize>,
    /// Exposure, tone mapping, gamma, and vignette applied to the final image.
    pub post: PostProcess,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            aa: AntiAliasing::None,
            background: 0x001A_2B3C,
            // One 4K frame. MSAA is still available at ordinary window sizes,
            // but a 1440p or 4K window falls back to the cheaper SMAA pass.
            max_internal_pixels: Some(4_000_000),
            post: PostProcess::default(),
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
    shadow_map: ShadowMap,
    smaa: SmaaScratch,
    last_timings: RenderTimings,
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
            // Grown on first use when a scene enables shadow mapping.
            shadow_map: ShadowMap::new(ShadowSettings::new(16)),
            smaa: SmaaScratch::new(),
            last_timings: RenderTimings::default(),
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
        let scene_start = Instant::now();
        let aa = self.active_aa();
        let scale = aa.scale() as usize;
        if scale == 1 {
            // No supersampling: render directly into the output framebuffer.
            // This avoids copying the whole image through `internal` first.
            self.output.resize(self.width, self.height);
            {
                let mut target = self.output.target();
                render_scene_clear_with_shadow_map(
                    &mut target,
                    scene,
                    camera,
                    &mut self.shadow,
                    self.options.background,
                    &mut self.shadow_map,
                );
            }
            if aa == AntiAliasing::Smaa {
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
            {
                let mut target = self.internal.target();
                render_scene_clear_with_shadow_map(
                    &mut target,
                    scene,
                    camera,
                    &mut self.shadow,
                    self.options.background,
                    &mut self.shadow_map,
                );
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
        let scene_ms = scene_start.elapsed().as_secs_f32() * 1000.0;
        let post_start = Instant::now();
        crate::post::apply(
            &mut self.output.color,
            self.width,
            self.height,
            &self.options.post,
        );
        self.last_timings = RenderTimings {
            scene_ms,
            post_ms: post_start.elapsed().as_secs_f32() * 1000.0,
        };
        &self.output.color
    }

    /// The anti-aliasing mode this renderer actually uses at its current size.
    ///
    /// This matches [`RenderOptions::aa`] except when a requested supersampled
    /// buffer would exceed [`RenderOptions::max_internal_pixels`]. In that case
    /// the renderer uses SMAA instead, so a large window cannot quietly turn a
    /// 2x request into a four-times-larger frame.
    pub fn active_aa(&self) -> AntiAliasing {
        let requested = self.options.aa;
        let Some(limit) = self.options.max_internal_pixels else {
            return requested;
        };
        let scale = requested.scale() as usize;
        if scale > 1 && self.width * scale * self.height * scale > limit {
            AntiAliasing::Smaa
        } else {
            requested
        }
    }

    /// The last rendered pixels.
    pub fn pixels(&self) -> &[u32] {
        &self.output.color
    }

    /// Timings from the last [`Renderer::render`] call.
    pub fn last_timings(&self) -> RenderTimings {
        self.last_timings
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

    #[test]
    fn oversized_msaa_falls_back_to_smaa() {
        // 1500x1500 at 2x is 9,000,000 internal pixels, above the 4M default.
        let mut big = Renderer::new(1500, 1500);
        big.options.aa = AntiAliasing::Msaa { factor: 2 };
        assert_eq!(big.active_aa(), AntiAliasing::Smaa);

        big.options.max_internal_pixels = None;
        assert_eq!(big.active_aa(), AntiAliasing::Msaa { factor: 2 });

        // 500x500 at 2x is 1,000,000 pixels, so MSAA stays selected.
        let mut small = Renderer::new(500, 500);
        small.options.aa = AntiAliasing::Msaa { factor: 2 };
        assert_eq!(small.active_aa(), AntiAliasing::Msaa { factor: 2 });
        // But a 4x request at the same size is 4,000,000 and still fits.
        small.options.aa = AntiAliasing::Msaa { factor: 4 };
        assert_eq!(small.active_aa(), AntiAliasing::Msaa { factor: 4 });
        // 501x501 at 4x is 4,016,016, just over the default.
        let mut just_over = Renderer::new(501, 501);
        just_over.options.aa = AntiAliasing::Msaa { factor: 4 };
        assert_eq!(just_over.active_aa(), AntiAliasing::Smaa);
    }

    /// The banded pass is an optimization, not a new look. A 520x520 frame is
    /// above the thread threshold, so this renders the same scene single- and
    /// multi-threaded and requires a pixel-for-pixel match.
    #[test]
    fn parallel_bands_match_single_threaded_output() {
        use crate::math::{Mat4, Vec3};
        use crate::parallel::set_thread_limit_for_tests;
        use crate::scene::DrawMode;

        let mut scene = Scene::new();
        scene.add(
            Object::new(Mesh::cube(), Material::new([0.9, 0.3, 0.2]))
                .with_transform(Mat4::translation(Vec3::new(-1.2, -0.3, 0.0))),
        );
        scene.add(
            Object::new(
                Mesh::torus(20, 10, 0.9, 0.35),
                Material::new([0.3, 0.7, 1.0]).with_specular(0.6, 64.0),
            )
            .with_transform(Mat4::translation(Vec3::new(1.1, 0.5, 0.0))),
        );
        scene.add(
            Object::new(Mesh::sphere(12, 18), Material::new([0.8, 0.8, 0.9]))
                .with_mode(DrawMode::Wireframe)
                .with_transform(Mat4::translation(Vec3::new(0.0, 0.6, 2.0))),
        );
        let camera = Camera::new(0.6, 0.35, 8.0);

        set_thread_limit_for_tests(1);
        let mut one = Renderer::new(520, 520);
        one.options.aa = AntiAliasing::None;
        let serial = one.render(&scene, &camera).to_vec();

        set_thread_limit_for_tests(8);
        let mut many = Renderer::new(520, 520);
        many.options.aa = AntiAliasing::None;
        let parallel = many.render(&scene, &camera).to_vec();
        set_thread_limit_for_tests(0);

        assert_eq!(
            serial, parallel,
            "banded render must match single-threaded render"
        );
    }

    /// The clear, downsample and SMAA passes also split across threads. Check
    /// that each AA mode gives the same pixels at one and many workers.
    #[test]
    fn parallel_aa_matches_single_threaded_output() {
        use crate::parallel::set_thread_limit_for_tests;

        let scene = test_scene();
        let camera = Camera::default();

        for aa in [
            AntiAliasing::Smaa,
            AntiAliasing::Msaa { factor: 2 },
            AntiAliasing::Msaa { factor: 4 },
        ] {
            set_thread_limit_for_tests(1);
            let mut one = Renderer::new(700, 700);
            one.options.aa = aa;
            one.options.max_internal_pixels = None;
            let serial = one.render(&scene, &camera).to_vec();

            set_thread_limit_for_tests(8);
            let mut many = Renderer::new(700, 700);
            many.options.aa = aa;
            many.options.max_internal_pixels = None;
            let parallel = many.render(&scene, &camera).to_vec();
            set_thread_limit_for_tests(0);

            assert_eq!(serial, parallel, "{aa:?} differs across thread counts");
        }
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
