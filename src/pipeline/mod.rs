//! Scene pipeline: transforms, clipping, lighting, textures, fog, grid,
//! planar shadows, shadow mapping, and draw dispatch. Resolution independent,
//! since everything works off a `Target`.
//! Made by AeroForger

mod clip;
mod frame;
mod grid;
mod object;
mod shade;
mod shadow;

pub use frame::{render_scene, render_scene_clear, render_scene_clear_with_shadow_map};
pub use grid::draw_grid;
pub use object::render_object;
pub use shade::{RenderContext, ResolvedLight, shade_vertex};
pub use shadow::{ShadowMask, composite_shadow, project_to_plane, render_shadow_into};
