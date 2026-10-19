//! # AeroGL
//!
//! A small software 3D rasterizer. No GPU and no shader compiler, just a
//! `Vec<u32>` framebuffer. Render into your own buffer or use the optional
//! `minifb` window backend.
//!
//! ## Quick start
//!
//! ```no_run
//! use aerogl::{Camera, Material, Mesh, Object, Renderer, Scene, Vec3};
//!
//! let mut renderer = Renderer::new(800, 800);
//! let camera = Camera::new(0.6, 0.4, 8.0);
//!
//! let mut scene = Scene::new();
//! scene.add(Object::new(Mesh::torus(32, 16, 0.9, 0.35), Material::new([0.3, 0.7, 1.0])));
//!
//! // Render one frame and get packed 0x00RRGGBB pixels back.
//! let pixels: &[u32] = renderer.render(&scene, &camera);
//! # let _ = pixels;
//! ```
//!
//! ## Feature flags
//!
//! * `window` (default): the `AeroglWindow` frontend built on `minifb`. Works
//!   on Linux (X11/Wayland), Windows, and macOS.
//! * No features: headless only. Render into a buffer and show it however you
//!   want (SDL, winit, a Wasm canvas, a file, whatever).
//!
//! ## Design
//!
//! | Layer | Type |
//! |---|---|
//! | pixels | [`Framebuffer`], [`Target`] |
//! | rasterizer | [`draw`] (lines, triangles, z-buffer) |
//! | transforms | [`Vec3`], [`Mat4`], [`Camera`] |
//! | scene | [`Scene`], [`Object`], [`Material`], [`DirectionalLight`] |
//! | frame composition | [`Renderer`], [`AntiAliasing`] |

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod camera;
pub mod draw;
pub mod framebuffer;
pub mod math;
pub mod mesh;
pub mod pipeline;
pub mod renderer;
pub mod scene;

#[cfg(feature = "window")]
pub mod window;

mod aa;
mod light;
mod material;

pub use aa::AntiAliasing;
pub use camera::Camera;
pub use framebuffer::{Framebuffer, Target};
pub use light::DirectionalLight;
pub use material::Material;
pub use math::{Mat4, Vec3};
pub use mesh::{Mesh, Vertex};
pub use pipeline::RenderContext;
pub use renderer::{RenderOptions, Renderer};
pub use scene::{DrawMode, Ground, Object, Scene};

#[cfg(feature = "window")]
pub use window::AeroglWindow;
