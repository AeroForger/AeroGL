//! Rasterization: lines, triangles, z-buffer. Resolution independent.
//! Made by AeroForger

pub mod line;
pub mod pixel;
pub mod triangle;

pub use line::{ScreenSeg, clip_segment_screen, draw_line, draw_line_z};
pub use pixel::{edge, lerp, luma, rgb, unpack};
pub use triangle::{
    ShadedVertex, draw_triangle_gouraud, draw_triangle_shaded, draw_triangle_shadow,
    draw_triangle_z,
};
