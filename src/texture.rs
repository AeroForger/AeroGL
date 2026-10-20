//! Textures and sampling: packed RGB images with repeat/clamp/mirror wrapping.
//! Made by AeroForger

use crate::draw::rgb;

/// How texture coordinates outside `[0,1]` are resolved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WrapMode {
    /// Tile the texture (the default for surfaces like floors and walls).
    #[default]
    Repeat,
    /// Stretch the edge texel outward.
    Clamp,
    /// Tile with every other copy flipped.
    Mirror,
}

/// A packed `0x00RRGGBB` image with bilinear sampling.
///
/// Textures are plain data and can be shared between materials through an
/// [`std::sync::Arc`]. Generate one with [`Texture::from_fn`],
/// [`Texture::checker`], or [`Texture::solid`], or build one from your own
/// pixels with [`Texture::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
    /// Coordinate wrapping mode.
    pub wrap: WrapMode,
}

impl Texture {
    /// Builds a texture from packed pixels, row-major, top-left origin.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() != width * height` or either dimension is zero.
    pub fn new(width: usize, height: usize, pixels: Vec<u32>) -> Self {
        assert!(width > 0 && height > 0, "texture needs a positive size");
        assert_eq!(
            pixels.len(),
            width * height,
            "texture pixels must be width * height"
        );
        Self {
            width,
            height,
            pixels,
            wrap: WrapMode::Repeat,
        }
    }

    /// Builds a texture by calling `f(u, v)` for every texel, with `u` and `v`
    /// at texel centers in `[0,1)`.
    pub fn from_fn(width: usize, height: usize, mut f: impl FnMut(f32, f32) -> [f32; 3]) -> Self {
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let [r, g, b] = f(u, v);
                pixels.push(rgb(r, g, b));
            }
        }
        Self::new(width, height, pixels)
    }

    /// A solid-color texture.
    pub fn solid(size: usize, color: [f32; 3]) -> Self {
        Self::from_fn(size, size, move |_, _| color)
    }

    /// A square checkerboard with `cells` squares per side.
    pub fn checker(size: usize, cells: usize, a: [f32; 3], b: [f32; 3]) -> Self {
        let cells = cells.max(1);
        Self::from_fn(size, size, move |u, v| {
            let x = (u * cells as f32).floor() as usize;
            let y = (v * cells as f32).floor() as usize;
            if (x + y).is_multiple_of(2) { a } else { b }
        })
    }

    /// Sets the wrapping mode (builder style).
    pub fn with_wrap(mut self, wrap: WrapMode) -> Self {
        self.wrap = wrap;
        self
    }

    /// Width in texels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in texels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// The packed texels, row-major.
    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    /// Nearest-neighbor sample. Cheap and crisp; useful for pixel art.
    pub fn sample_nearest(&self, u: f32, v: f32) -> [f32; 3] {
        let (x, y) = self.texel_coords(u, v);
        crate::draw::unpack(self.pixels[y * self.width + x])
    }

    /// Bilinear sample in linear-ish RGB space.
    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let (u, v) = if u.is_finite() && v.is_finite() {
            (u, v)
        } else {
            (0.0, 0.0)
        };
        // Texel centers live at (i + 0.5) / size.
        let x = u * self.width as f32 - 0.5;
        let y = v * self.height as f32 - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let (x0, y0) = (x0 as i32, y0 as i32);
        let (x1, y1) = (x0 + 1, y0 + 1);

        let c00 = self.texel(x0, y0);
        let c10 = self.texel(x1, y0);
        let c01 = self.texel(x0, y1);
        let c11 = self.texel(x1, y1);

        let mut out = [0.0; 3];
        for (channel, out) in out.iter_mut().enumerate() {
            let top = c00[channel] + (c10[channel] - c00[channel]) * fx;
            let bottom = c01[channel] + (c11[channel] - c01[channel]) * fx;
            *out = top + (bottom - top) * fy;
        }
        out
    }

    /// Integer texel coordinates for a `(u, v)` pair, honoring [`WrapMode`].
    fn texel_coords(&self, u: f32, v: f32) -> (usize, usize) {
        let x = (u * self.width as f32).floor();
        let y = (v * self.height as f32).floor();
        let x = if x.is_finite() { x as i32 } else { 0 };
        let y = if y.is_finite() { y as i32 } else { 0 };
        (
            self.wrap_index(x, self.width),
            self.wrap_index(y, self.height),
        )
    }

    #[inline]
    fn texel(&self, x: i32, y: i32) -> [f32; 3] {
        let x = self.wrap_index(x, self.width);
        let y = self.wrap_index(y, self.height);
        crate::draw::unpack(self.pixels[y * self.width + x])
    }

    #[inline]
    fn wrap_index(&self, index: i32, len: usize) -> usize {
        let len = len as i32;
        match self.wrap {
            WrapMode::Repeat => index.rem_euclid(len) as usize,
            WrapMode::Clamp => index.clamp(0, len - 1) as usize,
            WrapMode::Mirror => {
                let period = len * 2;
                let m = index.rem_euclid(period);
                if m < len {
                    m as usize
                } else {
                    (period - 1 - m) as usize
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red_green() -> Texture {
        // Left half red, right half green.
        Texture::from_fn(2, 1, |u, _| {
            if u < 0.5 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            }
        })
    }

    #[test]
    fn bilinear_midpoint_blends_neighbors() {
        let t = red_green();
        let [r, g, _] = t.sample(0.5, 0.5);
        assert!((r - 0.5).abs() < 0.02 && (g - 0.5).abs() < 0.02, "{r} {g}");
    }

    #[test]
    fn repeat_and_clamp_resolve_outside_coordinates_differently() {
        let repeat = red_green().with_wrap(WrapMode::Repeat);
        let clamp = red_green().with_wrap(WrapMode::Clamp);
        // u = 1.25 is red under repeat and green under clamp.
        assert!(repeat.sample(1.25, 0.5)[0] > 0.9);
        assert!(clamp.sample(1.25, 0.5)[1] > 0.9);
    }

    #[test]
    fn checker_alternates_cells() {
        let t = Texture::checker(64, 2, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
        assert!(t.sample_nearest(0.25, 0.25)[0] > 0.9);
        assert!(t.sample_nearest(0.75, 0.25)[0] < 0.1);
        assert!(t.sample_nearest(0.75, 0.75)[0] > 0.9);
    }

    #[test]
    fn mirror_wraps_with_flipped_copies() {
        let t = red_green().with_wrap(WrapMode::Mirror);
        // 1.75 maps to the mirrored x = 0.25, which is red.
        assert!(t.sample(1.75, 0.5)[0] > 0.9);
    }
}
