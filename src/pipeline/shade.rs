//! Vertex lighting: sun, point and spot lights, shadows, rim, and fog.

use crate::draw::lerp;
use crate::light::{
    DEFAULT_LIGHT_DIR, DirectionalLight, Light, cone_cosines, falloff, sanitize_cone_cosines,
};
use crate::material::Material;
use crate::math::{Vec3, finite_or, finite_rgb_or, finite_vec_or, normalized_or};
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
            camera_pos: finite_vec_or(camera_pos, Vec3::ZERO),
            light_dir: normalized_or(light.direction, DEFAULT_LIGHT_DIR),
            light_color: sanitized_light_color(light.color),
            ambient: finite_or(light.ambient, 0.15).clamp(0.0, 1.0),
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
                    position: finite_vec_or(p.position, Vec3::ZERO),
                    color: scaled_light_color(p.color, p.intensity),
                    range: finite_or(p.range, 0.0).max(0.0),
                },
                Light::Spot(s) => {
                    let (inner_cos, outer_cos) = cone_cosines(s.inner_angle_deg, s.outer_angle_deg);
                    ResolvedLight::Spot {
                        position: finite_vec_or(s.position, Vec3::ZERO),
                        direction: normalized_or(s.direction, DEFAULT_LIGHT_DIR),
                        color: scaled_light_color(s.color, s.intensity),
                        range: finite_or(s.range, 0.0).max(0.0),
                        inner_cos,
                        outer_cos,
                    }
                }
            })
            .collect();
        ctx
    }
}

fn sanitized_light_color(color: [f32; 3]) -> [f32; 3] {
    let color = finite_rgb_or(color, [0.0; 3]);
    [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)]
}

fn scaled_light_color(color: [f32; 3], intensity: f32) -> [f32; 3] {
    let intensity = finite_or(intensity, 0.0).max(0.0);
    let color = sanitized_light_color(color);
    [
        color[0] * intensity,
        color[1] * intensity,
        color[2] * intensity,
    ]
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
    let base = finite_rgb_or(base, [0.0; 3]);
    let normal = normal.normalize();
    let camera_pos = finite_vec_or(ctx.camera_pos, Vec3::ZERO);
    let to_cam = camera_pos.sub(pos).normalize();
    let shadow = ctx.shadow.map(|map| map.visibility(pos)).unwrap_or(1.0);
    let light_dir = normalized_or(ctx.light_dir, DEFAULT_LIGHT_DIR);
    let light_color = sanitized_light_color(ctx.light_color);
    let ambient = finite_or(ctx.ambient, 0.15).clamp(0.0, 1.0);
    let specular = finite_or(material.specular, 0.0).max(0.0);
    let shininess = finite_or(material.shininess, 48.0).max(1.0);
    let rim_strength = finite_or(material.rim, 0.0).max(0.0);
    let emissive = finite_rgb_or(material.emissive, [0.0; 3]);

    // Primary directional light.
    let to_light = Vec3::ZERO.sub(light_dir);
    let ndotl = normal.dot(to_light).max(0.0);
    let spec = if ndotl > 0.0 {
        let half = to_light.add(to_cam).normalize();
        normal.dot(half).max(0.0).powf(shininess)
    } else {
        0.0
    };

    let mut out = [0.0; 3];
    for (i, channel) in out.iter_mut().enumerate() {
        let ambient_term = base[i] * ambient * light_color[i];
        let diffuse_term = base[i] * (1.0 - ambient) * ndotl * light_color[i] * shadow;
        let spec_term = spec * specular * light_color[i] * shadow;
        *channel = ambient_term + diffuse_term + spec_term;
    }

    // Additional local lights.
    for light in &ctx.lights {
        let (position, color, range, cone) = match *light {
            ResolvedLight::Point {
                position,
                color,
                range,
            } => (
                finite_vec_or(position, Vec3::ZERO),
                sanitized_light_color(color),
                finite_or(range, 0.0).max(0.0),
                1.0,
            ),
            ResolvedLight::Spot {
                position,
                direction,
                color,
                range,
                inner_cos,
                outer_cos,
            } => {
                let position = finite_vec_or(position, Vec3::ZERO);
                let direction = normalized_or(direction, DEFAULT_LIGHT_DIR);
                let (inner_cos, outer_cos) = sanitize_cone_cosines(inner_cos, outer_cos);
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
                (
                    position,
                    sanitized_light_color(color),
                    finite_or(range, 0.0).max(0.0),
                    cone,
                )
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
        let attenuation = falloff(dist, range);
        let half = to_light.add(to_cam).normalize();
        let spec = normal.dot(half).max(0.0).powf(shininess);
        let factor = attenuation * cone;
        for ((channel, &base_channel), &light_channel) in
            out.iter_mut().zip(base.iter()).zip(color.iter())
        {
            *channel += (base_channel * ndotl + spec * specular) * light_channel * factor;
        }
    }

    let rim = (1.0 - normal.dot(to_cam).max(0.0)).powi(2) * rim_strength;
    for ((channel, &base_channel), &emissive) in
        out.iter_mut().zip(base.iter()).zip(emissive.iter())
    {
        *channel += rim * base_channel + emissive;
    }

    if let Some(fog) = ctx.fog {
        let factor = fog.factor(camera_pos.sub(pos).length());
        let fog_color = finite_rgb_or(fog.color, [0.0; 3]);
        for (channel, &fog_channel) in out.iter_mut().zip(fog_color.iter()) {
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

    #[test]
    fn invalid_public_shading_values_stay_finite() {
        let light = DirectionalLight {
            direction: Vec3::ZERO,
            color: [f32::NAN, -1.0, f32::INFINITY],
            ambient: f32::NAN,
        };
        let ctx = RenderContext::new(Vec3::new(f32::NAN, 0.0, 0.0), &light);
        let material = Material {
            color: [f32::NAN, 1.0, -1.0],
            specular: f32::NAN,
            shininess: f32::INFINITY,
            rim: f32::NAN,
            texture: None,
            alpha: f32::NAN,
            emissive: [f32::NAN, 1.0, 0.0],
        };
        let color = shade_vertex(
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            &material,
            &ctx,
        );
        assert!(color.iter().all(|channel| channel.is_finite()), "{color:?}");
    }
}
