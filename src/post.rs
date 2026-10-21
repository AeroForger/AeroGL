//! Post-processing: exposure, tone mapping, saturation, vignette, and gamma.
//! Made by AeroForger

use crate::draw::rgb;
use crate::math::finite_or;

/// Tone-mapping curve applied to the final image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToneMapping {
    /// Straight clamp to `[0,1]`.
    #[default]
    None,
    /// `x / (1 + x)`, a soft filmic rolloff.
    Reinhard,
    /// ACES filmic approximation, with a punchier highlight rolloff.
    Aces,
}

/// Display transform applied after anti-aliasing, before pixels are returned.
///
/// The defaults are an identity transform. Set `exposure`, a `tone_mapping`
/// curve, and optionally `gamma`/`vignette`/`saturation` for a graded look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostProcess {
    /// Linear exposure multiplier applied before tone mapping.
    pub exposure: f32,
    /// Highlight rolloff curve.
    pub tone_mapping: ToneMapping,
    /// Display gamma. `1.0` leaves values untouched; `2.2` applies the usual
    /// sRGB-style encode (`pow(1/2.2)`).
    pub gamma: f32,
    /// Color saturation (`1.0` = unchanged, `0.0` = grayscale).
    pub saturation: f32,
    /// Corner darkening in `[0,1]` (`0.0` = off).
    pub vignette: f32,
}

impl Default for PostProcess {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            tone_mapping: ToneMapping::None,
            gamma: 1.0,
            saturation: 1.0,
            vignette: 0.0,
        }
    }
}

impl PostProcess {
    /// Whether this transform would leave every pixel unchanged.
    pub fn is_identity(&self) -> bool {
        let exposure = finite_or(self.exposure, 1.0).max(0.0);
        let gamma_value = finite_or(self.gamma, 1.0).max(0.01);
        let saturation_value = finite_or(self.saturation, 1.0).max(0.0);
        let vignette_value = finite_or(self.vignette, 0.0).clamp(0.0, 1.0);
        (exposure - 1.0).abs() < 1e-6
            && self.tone_mapping == ToneMapping::None
            && (gamma_value - 1.0).abs() < 1e-6
            && (saturation_value - 1.0).abs() < 1e-6
            && vignette_value <= 1e-6
    }

    /// Sets the exposure (builder style).
    pub fn with_exposure(mut self, exposure: f32) -> Self {
        self.exposure = finite_or(exposure, 1.0).max(0.0);
        self
    }

    /// Sets the tone-mapping curve (builder style).
    pub fn with_tone_mapping(mut self, tone_mapping: ToneMapping) -> Self {
        self.tone_mapping = tone_mapping;
        self
    }

    /// Sets the display gamma (builder style).
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = finite_or(gamma, 1.0).max(0.01);
        self
    }

    /// Sets the saturation (builder style).
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = finite_or(saturation, 1.0).max(0.0);
        self
    }

    /// Sets the vignette amount (builder style).
    pub fn with_vignette(mut self, vignette: f32) -> Self {
        self.vignette = finite_or(vignette, 0.0).clamp(0.0, 1.0);
        self
    }
}

/// Tone-maps one linear RGB channel.
pub fn tone_map_channel(value: f32, tone_mapping: ToneMapping) -> f32 {
    let x = finite_or(value, 0.0).max(0.0);
    match tone_mapping {
        ToneMapping::None => x,
        ToneMapping::Reinhard => x / (1.0 + x),
        ToneMapping::Aces => {
            let a = 2.51;
            let b = 0.03;
            let c = 2.43;
            let d = 0.59;
            let e = 0.14;
            ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
        }
    }
}

/// Applies `settings` to a packed `0x00RRGGBB` image in place.
pub fn apply(pixels: &mut [u32], width: usize, height: usize, settings: &PostProcess) {
    if settings.is_identity() || width == 0 || height == 0 {
        return;
    }
    let Some(len) = width.checked_mul(height) else {
        return;
    };
    if pixels.len() < len {
        return;
    }

    let params = Params::new(width, height, settings);
    let pixels = &mut pixels[..len];
    let threads = crate::parallel::threads_for(len, 400_000);
    if threads <= 1 {
        apply_rows(pixels, width, 0, height, &params);
        return;
    }

    // Rows are independent, so each worker owns a contiguous band and no
    // synchronization is needed inside the loop.
    let rows_per = height.div_ceil(threads);
    let mut rest: &mut [u32] = pixels;
    let mut y = 0;
    let params = &params;
    std::thread::scope(|scope| {
        while y < height {
            let rows = rows_per.min(height - y);
            let (band, tail) = rest.split_at_mut(rows * width);
            scope.spawn(move || apply_rows(band, width, y, rows, params));
            rest = tail;
            y += rows;
        }
    });
}

/// Constants and the per-channel table shared by all row bands.
struct Params {
    channel: [f32; 256],
    saturation: f32,
    saturation_inv: f32,
    saturation_on: bool,
    vignette_scale: f32,
    vignette_on: bool,
    gamma: bool,
    inv_gamma: f32,
    cx: f32,
    cy: f32,
}

impl Params {
    fn new(width: usize, height: usize, settings: &PostProcess) -> Self {
        let exposure = finite_or(settings.exposure, 1.0).max(0.0);
        let gamma_value = finite_or(settings.gamma, 1.0).max(0.01);
        let saturation_value = finite_or(settings.saturation, 1.0).max(0.0);
        let vignette_value = finite_or(settings.vignette, 0.0).clamp(0.0, 1.0);
        let gamma = (gamma_value - 1.0).abs() > 1e-6;
        let vignette_on = vignette_value > 1e-6;
        let saturation_on = (saturation_value - 1.0).abs() > 1e-6;

        // The input channels are already 8-bit, so exposure and tone mapping
        // can be done once per byte instead of once per pixel channel.
        let mut channel = [0.0; 256];
        for (value, out) in channel.iter_mut().enumerate() {
            let mut channel = value as f32 / 255.0;
            channel *= exposure;
            if settings.tone_mapping != ToneMapping::None {
                channel = tone_map_channel(channel, settings.tone_mapping);
            }
            *out = channel;
        }

        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        let max_dist = (cx * cx + cy * cy).sqrt().max(1.0);
        Self {
            channel,
            saturation: saturation_value,
            saturation_inv: 1.0 - saturation_value,
            saturation_on,
            vignette_scale: vignette_value / max_dist,
            vignette_on,
            gamma,
            inv_gamma: 1.0 / gamma_value,
            cx,
            cy,
        }
    }
}

/// Applies the transform to `rows` starting at `y_start`.
fn apply_rows(pixels: &mut [u32], width: usize, y_start: usize, rows: usize, params: &Params) {
    for row in 0..rows {
        let y = y_start + row;
        let dy = y as f32 + 0.5 - params.cy;
        let row_start = row * width;
        for x in 0..width {
            let idx = row_start + x;
            let packed = pixels[idx];
            let mut c = [
                params.channel[((packed >> 16) & 0xFF) as usize],
                params.channel[((packed >> 8) & 0xFF) as usize],
                params.channel[(packed & 0xFF) as usize],
            ];
            let factor = if params.vignette_on {
                let dx = x as f32 + 0.5 - params.cx;
                1.0 - (dx * dx + dy * dy).sqrt() * params.vignette_scale
            } else {
                1.0
            };
            if params.saturation_on {
                let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                let gray = luma * params.saturation_inv;
                c[0] = c[0].mul_add(params.saturation, gray) * factor;
                c[1] = c[1].mul_add(params.saturation, gray) * factor;
                c[2] = c[2].mul_add(params.saturation, gray) * factor;
            } else if params.vignette_on {
                c[0] *= factor;
                c[1] *= factor;
                c[2] *= factor;
            }
            if params.gamma {
                c[0] = c[0].clamp(0.0, 1.0).powf(params.inv_gamma);
                c[1] = c[1].clamp(0.0, 1.0).powf(params.inv_gamma);
                c[2] = c[2].clamp(0.0, 1.0).powf(params.inv_gamma);
            }
            pixels[idx] = rgb(c[0], c[1], c[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::unpack;

    #[test]
    fn identity_does_not_touch_pixels() {
        let mut pixels = vec![0x123456, 0xABCDEF];
        apply(&mut pixels, 2, 1, &PostProcess::default());
        assert_eq!(pixels, vec![0x123456, 0xABCDEF]);
    }

    #[test]
    fn exposure_brightens_and_tone_mapping_rolls_off() {
        let mut pixels = vec![0x808080];
        apply(
            &mut pixels,
            1,
            1,
            &PostProcess::default().with_exposure(2.0),
        );
        assert!(pixels[0] > 0x808080);

        let mut toned = vec![0xFFFFFF];
        let s = PostProcess::default().with_tone_mapping(ToneMapping::Aces);
        apply(&mut toned, 1, 1, &s);
        assert!(toned[0] < 0xFFFFFF, "ACES should roll off pure white");
        assert!(tone_map_channel(0.0, ToneMapping::Aces).abs() < 1e-6);
    }

    #[test]
    fn gamma_lifts_midtones() {
        let mut pixels = vec![0x404040];
        apply(&mut pixels, 1, 1, &PostProcess::default().with_gamma(2.2));
        assert!(pixels[0] > 0x404040);
    }

    #[test]
    fn vignette_darkens_corners_more_than_the_center() {
        let mut pixels = vec![0xFFFFFFFF; 9];
        apply(
            &mut pixels,
            3,
            3,
            &PostProcess::default().with_vignette(0.8),
        );
        assert!(pixels[4] > pixels[0], "center should stay brighter");
    }

    #[test]
    fn short_buffers_are_left_alone() {
        let mut pixels = vec![0x123456; 3];
        let before = pixels.clone();
        apply(
            &mut pixels,
            2,
            2,
            &PostProcess::default().with_exposure(2.0),
        );
        assert_eq!(pixels, before);
    }

    #[test]
    fn invalid_public_values_do_not_produce_nans() {
        let mut pixels = vec![0x808080];
        let settings = PostProcess {
            exposure: f32::NAN,
            gamma: 0.0,
            saturation: -2.0,
            vignette: 99.0,
            ..PostProcess::default()
        };
        apply(&mut pixels, 1, 1, &settings);
        let [r, g, b] = unpack(pixels[0]);
        assert!(r.is_finite() && g.is_finite() && b.is_finite());
    }

    #[test]
    fn identity_check_uses_sanitized_values() {
        let settings = PostProcess {
            exposure: f32::NAN,
            gamma: f32::NAN,
            saturation: f32::NAN,
            vignette: f32::NAN,
            ..PostProcess::default()
        };
        assert!(settings.is_identity());

        let mut pixels = vec![0x123456];
        apply(&mut pixels, 1, 1, &settings);
        assert_eq!(pixels, vec![0x123456]);
    }

    #[test]
    fn builders_reject_non_finite_values() {
        let post = PostProcess::default()
            .with_exposure(f32::NAN)
            .with_gamma(f32::NAN)
            .with_saturation(f32::NAN)
            .with_vignette(f32::NAN);
        assert_eq!(post.exposure, 1.0);
        assert_eq!(post.gamma, 1.0);
        assert_eq!(post.saturation, 1.0);
        assert_eq!(post.vignette, 0.0);
        assert!(post.is_identity());
    }

    #[test]
    fn banded_apply_matches_a_single_band() {
        let (width, height) = (800, 600);
        let source: Vec<u32> = (0..width * height)
            .map(|i| {
                let x = (i % width) as u32;
                let y = (i / width) as u32;
                ((x * 255 / width as u32) << 16)
                    | ((y * 255 / height as u32) << 8)
                    | ((x + y) & 0xFF)
            })
            .collect();
        let settings = PostProcess::default()
            .with_exposure(1.25)
            .with_tone_mapping(ToneMapping::Aces)
            .with_saturation(1.08)
            .with_vignette(0.22)
            .with_gamma(2.2);

        let mut expected = source.clone();
        let params = Params::new(width, height, &settings);
        apply_rows(&mut expected, width, 0, height, &params);

        let mut actual = source;
        apply(&mut actual, width, height, &settings);

        assert_eq!(actual, expected);
    }
}
