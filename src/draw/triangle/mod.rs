//! Triangle rasterization: barycentric fill with z-buffer, Gouraud shading,
//! perspective-correct texture mapping, alpha blending, and depth-only passes.
//! Made by AeroForger

mod buffers;
mod flat;
mod shaded;

pub use flat::{draw_triangle_shadow, draw_triangle_z};
pub use shaded::{ShadedVertex, draw_triangle_gouraud, draw_triangle_shaded};

pub(crate) use buffers::{draw_triangle_cover, draw_triangle_depth};

#[cfg(test)]
mod tests;
