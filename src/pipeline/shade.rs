//! Vertex lighting: sun, point and spot lights, shadows, rim, and fog.

use crate::draw::lerp;
use crate::light::{DirectionalLight, Light};
use crate::material::Material;
use crate::math::Vec3;
use crate::scene::{Fog, Scene};
use crate::shadow_map::ShadowMap;

/// One extra local light, resolved from [`crate::light::Light`] for shading.
#[derive(Clone, Copy, Debug)]
pub enum ResolvedLight {
    /// A point light with color already scaled by intensity.
    Point {
        /// World-space position.
        position: Vec3,
        /// Linear RGB, intensity included.
        color: [f32; 3],
        /// Falloff range (`0` = no falloff).
        range: f32,
    },
    /// A spot light with color already scaled by intensity.
    Spot {
        /// World-space position.
        position: Vec3,
        /// Normalized cone axis, pointing toward the scene.
        direction: Vec3,
        /// Linear RGB, intensity included.
        color: [f32; 3],
        /// Falloff range (`0` = no falloff).
        range: f32,
        /// Cosine of the full-intensity cone angle.
        inner_cos: f32,
        /// Cosine of the zero-intensity cone angle.
        outer_cos: f32,
    },
}

/// Per-frame lighting state shared by every object.
pub struct RenderContext<'a> {
    /// World-space camera position, for specular highlights and fog.
    pub camera_pos: Vec3,
    /// Normalized direction the primary light travels (toward the scene).
    pub light_dir: Vec3,
    /// Primary light color, linear RGB.
    pub light_color: [f32; 3],
    /// Ambient fill.
    pub ambient: f32,
    /// Additional point and spot lights.
    pub lights: Vec<ResolvedLight>,
    /// Optional distance fog.
    pub fog: Option<Fog>,
    /// Optional directional shadow map sampled per vertex.
    pub shadow: Option<&'a ShadowMap>,
}

impl<'a> RenderContext<'a> {
    /// Builds a context from a camera position and the primary light only.
    pub fn new(camera_pos: Vec3, light: &DirectionalLight) -> Self {
        Self {
            camera_pos,
            light_dir: light.direction.normalize(),
            light_color: light.color,
            ambient: light.ambient,
            lights: Vec::new(),
            fog: None,
            shadow: None,
        }
    }

    /// Builds a context from the scene's sun, extra lights, and fog.
    pub fn from_scene(camera_pos: Vec3, scene: &Scene) -> Self {
        let mut ctx = Self::new(camera_pos, &scene.light);
        ctx.fog = scene.fog;
        ctx.lights = scene
            .lights
            .iter()
            .map(|light| match light {
                Light::Point(p) => ResolvedLight::Point {
                    position: p.position,
                    color: [
                        p.color[0] * p.intensity,
                        p.color[1] * p.intensity,
                        p.color[2] * p.intensity,
                    ],
                    range: p.range,
                },
                Light::Spot(s) => ResolvedLight::Spot {
                    position: s.position,
                    direction: s.direction.normalize(),
                    color: [
                        s.color[0] * s.intensity,
                        s.color[1] * s.intensity,
                        s.color[2] * s.intensity,
                    ],
                    range: s.range,
                    inner_cos: s.inner_angle_deg.to_radians().cos(),
                    outer_cos: s.outer_angle_deg.to_radians().cos(),
                },
            })
            .collect();
        ctx
    }
}

/// Blinn-Phong vertex lighting: ambient + sun + point/spot lights + rim, then
/// shadow visibility, emission, and fog.
///
/// Colors are linear and may exceed `1.0`; the rasterizer clamps when packing.
pub fn shade_vertex(pos: Vec3, normal: Vec3, material: &Material, ctx: &RenderContext) -> [f32; 3] {
    shade_vertex_base(pos, normal, material.color, material, ctx)
}

/// Lighting with an explicit albedo, so textured materials can light a white
/// surface and multiply the texel in per pixel.
pub(super) fn shade_vertex_base(
    pos: Vec3,
    normal: Vec3,
    base: [f32; 3],
    material: &Material,
    ctx: &RenderContext,
) -> [f32; 3] {
    let normal = normal.normalize();
    let to_cam = ctx.camera_pos.sub(pos).normalize();
    let shadow = ctx.shadow.map(|map| map.visibility(pos)).unwrap_or(1.0);

    // Primary directional light.
    let to_light = Vec3::ZERO.sub(ctx.light_dir);
    let ndotl = normal.dot(to_light).max(0.0);
    let spec = if ndotl > 0.0 {
        let half = to_light.add(to_cam).normalize();
        normal.dot(half).max(0.0).powf(material.shininess.max(1.0))
    } else {
        0.0
    };
    let ambient = ctx.ambient;

    let mut out = [0.0; 3];
    for (i, channel) in out.iter_mut().enumerate() {
        let ambient_term = base[i] * ambient * ctx.light_color[i];
        let diffuse_term = base[i] * (1.0 - ambient) * ndotl * ctx.light_color[i] * shadow;
        let spec_term = spec * material.specular * ctx.light_color[i] * shadow;
        *channel = ambient_term + diffuse_term + spec_term;
    }

    // Additional local lights.
    for light in &ctx.lights {
        let (position, color, range, cone) = match *light {
            ResolvedLight::Point {
                position,
                color,
                range,
            } => (position, color, range, 1.0),
            ResolvedLight::Spot {
                position,
                direction,
                color,
                range,
                inner_cos,
                outer_cos,
            } => {
                let delta = position.sub(pos);
                let dist = delta.length().max(1e-5);
                let to_light = delta.scale(1.0 / dist);
                let cos = Vec3::ZERO.sub(to_light).dot(direction);
                let cone = if cos <= outer_cos {
                    0.0
                } else if cos >= inner_cos {
                    1.0
                } else {
                    let t = (cos - outer_cos) / (inner_cos - outer_cos).max(1e-6);
                    t * t * (3.0 - 2.0 * t)
                };
                (position, color, range, cone)
            }
        };
        if cone <= 0.0 {
            continue;
        }
        let delta = position.sub(pos);
        let dist = delta.length().max(1e-5);
        let to_light = delta.scale(1.0 / dist);
        let ndotl = normal.dot(to_light).max(0.0);
        if ndotl <= 0.0 {
            continue;
        }
        let attenuation = if range <= 0.0 {
            1.0
        } else {
            let t = (dist / range).clamp(0.0, 1.0);
            (1.0 - t * t) * (1.0 - t * t)
        };
        let half = to_light.add(to_cam).normalize();
        let spec = normal.dot(half).max(0.0).powf(material.shininess.max(1.0));
        let factor = attenuation * cone;
        for ((channel, &base_channel), &light_channel) in
            out.iter_mut().zip(base.iter()).zip(color.iter())
        {
            *channel += (base_channel * ndotl + spec * material.specular) * light_channel * factor;
        }
    }

    let rim = (1.0 - normal.dot(to_cam).max(0.0)).powi(2) * material.rim;
    for ((channel, &base_channel), &emissive) in out
        .iter_mut()
        .zip(base.iter())
        .zip(material.emissive.iter())
    {
        *channel += rim * base_channel + emissive;
    }

    if let Some(fog) = ctx.fog {
        let factor = fog.factor(ctx.camera_pos.sub(pos).length());
        for (channel, &fog_channel) in out.iter_mut().zip(fog.color.iter()) {
            *channel = lerp(*channel, fog_channel, factor);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_phong_lit_side_is_brighter() {
        let ctx = RenderContext::new(Vec3::new(0.0, 5.0, 8.0), &DirectionalLight::default());
        let up = -ctx.light_dir;
        let m = Material::new([1.0, 0.3, 0.2]);
        let lit = shade_vertex(Vec3::ZERO, up, &m, &ctx);
        let dark = shade_vertex(Vec3::ZERO, -up, &m, &ctx);
        assert!(lit[0] > dark[0] * 2.0, "lit {lit:?} vs dark {dark:?}");
    }
}
