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
//! | rasterizer | [`draw`] (lines, triangles, z-buffer, textures, blending) |
//! | transforms | [`Vec3`], [`Mat4`], [`Camera`] |
//! | scene | [`Scene`], [`Object`], [`Material`], [`Texture`], [`DirectionalLight`], [`Light`] |
//! | shadows | [`ShadowMap`], [`ShadowSettings`] |
//! | post | [`PostProcess`], [`ToneMapping`] |
//! | frame composition | [`Renderer`], [`AntiAliasing`] |

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod camera;
pub mod draw;
pub mod framebuffer;
pub mod math;
pub mod mesh;
pub mod obj;
pub mod pipeline;
pub mod post;
pub mod renderer;
pub mod scene;
pub mod shadow_map;
pub mod texture;

#[cfg(feature = "window")]
pub mod window;

mod aa;
mod light;
mod material;
mod parallel;

pub use aa::AntiAliasing;
pub use camera::Camera;
pub use framebuffer::{Framebuffer, FramebufferError, Target};
pub use light::{DirectionalLight, Light, PointLight, SpotLight};
pub use material::Material;
pub use math::{Mat4, Vec3};
pub use mesh::{Mesh, MeshError, Vertex};
pub use obj::ObjError;
pub use pipeline::{RenderContext, SceneRender};
pub use post::{PostProcess, ToneMapping};
pub use renderer::{RenderOptions, Renderer};
pub use scene::{DrawMode, Fog, Ground, Object, Scene};
pub use shadow_map::{ShadowMap, ShadowSettings};
pub use texture::{Texture, TextureError, WrapMode};

#[cfg(feature = "window")]
pub use window::AeroglWindow;
