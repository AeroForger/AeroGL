//! Optional window frontend built on `minifb`.
//!
//! Enabled by the default `window` feature. Works on Linux (X11 and Wayland
//! through the X11 bridge), Windows, and macOS. Disable the feature to use
//! AeroGL headless with your own windowing layer.
//!
//! ```no_run
//! use aerogl::{AeroglWindow, Camera, Material, Mesh, Object, Scene};
//!
//! let mut app = AeroglWindow::new("my app", 640, 480).expect("window");
//! let mut camera = Camera::default();
//! let mut scene = Scene::new();
//! scene.add_mesh(Mesh::torus(24, 12, 0.9, 0.35), Material::new([0.3, 0.7, 1.0]));
//!
//! while app.is_open() {
//!     app.draw(&scene, &camera).expect("present");
//! }
//! ```

use crate::camera::Camera;
use crate::renderer::Renderer;
use crate::scene::Scene;
use minifb::{Key, Window, WindowOptions};

/// A `minifb` window with a [`Renderer`] attached.
///
/// The window owns a renderer sized to the window; use [`Self::renderer_mut`]
/// to change resolution-independent settings such as anti-aliasing.
pub struct AeroglWindow {
    window: Window,
    renderer: Renderer,
    auto_resize: bool,
}

impl AeroglWindow {
    /// Opens a window. Fails if the platform can't create one.
    pub fn new(title: &str, width: usize, height: usize) -> Result<Self, minifb::Error> {
        let mut window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        )?;
        // Set once: re-applying every frame is a platform round-trip.
        window.set_target_fps(120);

        Ok(Self {
            window,
            renderer: Renderer::new(width, height),
            auto_resize: true,
        })
    }

    /// Whether the renderer follows window resizes (default `true`).
    ///
    /// Turn this off to render at a fixed internal resolution while the window
    /// scales the image, e.g. for chunky retro pixels.
    pub fn set_auto_resize(&mut self, auto_resize: bool) {
        self.auto_resize = auto_resize;
    }

    /// The renderer behind the window.
    pub fn renderer(&self) -> &Renderer {
        &self.renderer
    }

    /// The renderer behind the window, for changing options or resolution.
    pub fn renderer_mut(&mut self) -> &mut Renderer {
        &mut self.renderer
    }

    /// The underlying `minifb` window, for input queries.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// The underlying `minifb` window, for input and configuration.
    pub fn window_mut(&mut self) -> &mut Window {
        &mut self.window
    }

    /// Whether the window is still open and Escape has not been pressed.
    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    /// Renders and presents one frame.
    pub fn draw(&mut self, scene: &Scene, camera: &Camera) -> Result<(), minifb::Error> {
        if self.auto_resize {
            let (width, height) = self.window.get_size();
            // Ignore minimized or zero-sized windows, since a 0x0 framebuffer
            // is invalid.
            if width > 0
                && height > 0
                && (width != self.renderer.width() || height != self.renderer.height())
            {
                self.renderer.resize(width, height);
            }
        }
        let (width, height) = (self.renderer.width(), self.renderer.height());
        let pixels = self.renderer.render(scene, camera);
        self.window.update_with_buffer(pixels, width, height)
    }

    /// Sets the window title. This is a platform call, so don't do it every
    /// frame.
    pub fn set_title(&mut self, title: &str) {
        self.window.set_title(title);
    }
}
