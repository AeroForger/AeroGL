//! Object preparation, clipping, and rasterization.

use crate::draw::{draw_line_z, draw_triangle_shaded, rgb};
use crate::framebuffer::Target;
use crate::math::{Mat4, Vec3, finite_rgb_or};
use crate::scene::{DrawMode, Object};

use super::clip::{ClipVertex, clip_far, clip_near, ndc_to_screen};
use super::shade::{RenderContext, shade_vertex_base};

/// Renders one meshed object, honoring its [`DrawMode`].
///
/// The object is prepared once and then rasterized. The scene pipeline uses
/// its internal preparation pass so all bands can share one preparation.
pub fn render_object(
    target: &mut Target,
    object: &Object,
    view: Mat4,
    proj: Mat4,
    near: f32,
    ctx: &RenderContext,
) {
    let view_proj = proj.multiply(&view);
    let prepared = prepare_object(object, view_proj, ctx);
    render_prepared_object(target, object, &prepared, near);
}

/// Transforms and lights every vertex of one object, once per frame.
///
/// Bands only clip and rasterize these vertices, so shared mesh vertices are
/// shaded once instead of once per triangle per band.
pub(super) fn prepare_object(
    object: &Object,
    view_proj: Mat4,
    ctx: &RenderContext,
) -> Vec<ClipVertex> {
    let model = object.transform;
    let mvp = view_proj.multiply(&model);
    let shaded = matches!(object.mode, DrawMode::Shaded | DrawMode::ShadedWireframe);
    let textured = object.material.texture.is_some();
    // Textured materials light a white surface and multiply the texel in per
    // pixel. Untextured ones use the material color as the albedo.
    let base = if textured {
        [1.0, 1.0, 1.0]
    } else {
        object.material.color
    };
    let alpha = object.material.effective_alpha();
    let mut out = Vec::with_capacity(object.mesh.vertices.len());
    for v in &object.mesh.vertices {
        let color = if shaded {
            let world = model.transform_point(v.pos);
            let normal = model.transform_dir(v.normal).normalize();
            shade_vertex_base(world, normal, base, &object.material, ctx)
        } else {
            [0.0; 3]
        };
        out.push(ClipVertex {
            pos: mvp.transform_h(v.pos),
            color,
            uv: v.uv,
            alpha,
        });
    }
    out
}

/// Clips and rasterizes one object from precomputed clip-space vertices.
pub(super) fn render_prepared_object(
    target: &mut Target,
    object: &Object,
    prepared: &[ClipVertex],
    near: f32,
) {
    let base = finite_rgb_or(object.material.color, [0.0; 3]);
    let wire_col = rgb(
        base[0] * 0.3 + 0.5,
        base[1] * 0.3 + 0.5,
        base[2] * 0.3 + 0.5,
    );
    let shaded = matches!(object.mode, DrawMode::Shaded | DrawMode::ShadedWireframe);
    let wireframe = matches!(object.mode, DrawMode::Wireframe | DrawMode::ShadedWireframe);

    debug_assert!(
        object.mesh.indices.len().is_multiple_of(3),
        "mesh index buffer is not a whole number of triangles"
    );
    for tri in object.mesh.indices.as_chunks::<3>().0 {
        let (Some(&p0), Some(&p1), Some(&p2)) = (
            prepared.get(tri[0] as usize),
            prepared.get(tri[1] as usize),
            prepared.get(tri[2] as usize),
        ) else {
            debug_assert!(false, "mesh triangle references a missing vertex");
            continue;
        };
        let clipped = clip_near([p0, p1, p2], near);
        if clipped.len() < 3 {
            continue;
        }
        let clipped = clip_far(clipped);
        if clipped.len() < 3 {
            continue;
        }

        // Fan the clipped polygon into triangles.
        let verts = clipped.as_slice();
        for i in 1..verts.len() - 1 {
            let v = [verts[0], verts[i], verts[i + 1]];
            let n: [Vec3; 3] = [v[0].ndc(), v[1].ndc(), v[2].ndc()];
            let winding =
                (n[1].x - n[0].x) * (n[2].y - n[0].y) - (n[2].x - n[0].x) * (n[1].y - n[0].y);

            if shaded && winding > 0.0 {
                draw_triangle_shaded(
                    target,
                    v[0].shaded(n[0], target.width, target.height),
                    v[1].shaded(n[1], target.width, target.height),
                    v[2].shaded(n[2], target.width, target.height),
                    object.material.texture.as_deref(),
                );
            }

            if wireframe {
                let s = [
                    ndc_to_screen(n[0], target.width, target.height),
                    ndc_to_screen(n[1], target.width, target.height),
                    ndc_to_screen(n[2], target.width, target.height),
                ];
                for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                    draw_line_z(
                        target, s[a].0, s[a].1, n[a].z, s[b].0, s[b].1, n[b].z, wire_col,
                    );
                }
            }
        }
    }
}
