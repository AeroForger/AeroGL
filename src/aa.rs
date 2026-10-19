//! Anti-aliasing: supersampled MSAA and a simplified SMAA post-process.
//! Made by AeroForger

/// Anti-aliasing strategy used by [`crate::Renderer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AntiAliasing {
    /// No anti-aliasing: one sample per pixel, hard edges.
    #[default]
    None,
    /// Multi-sample anti-aliasing, done in software as full-scene
    /// supersampling: the frame renders at `factor`x resolution and is box
    /// resolved down.
    ///
    /// `factor: 2` gives 4 samples per pixel and is usually the one you want.
    /// `3` costs 9x the fill but looks great. This works like MSAA with
    /// per-sample shading, and it also smooths wireframes, grid lines, and
    /// shadows, which a triangle-only MSAA would leave aliased.
    Msaa {
        /// Supersampling factor per axis (clamped to 1..=4).
        factor: u32,
    },
    /// Simplified SMAA: luma-based edge detection plus directional blending,
    /// run as a single post-process pass at native resolution.
    ///
    /// About half the cost of `Msaa { factor: 2 }`. It is a filter, so it
    /// cannot bring back detail that was never sampled, and hard sub-pixel
    /// geometry can still shimmer.
    Smaa,
}

impl AntiAliasing {
    /// Render scale implied by this mode (1 for none/SMAA).
    pub fn scale(self) -> u32 {
        match self {
            AntiAliasing::Msaa { factor } => factor.clamp(1, 4),
            _ => 1,
        }
    }
}

/// Box-downsamples `src` (`factor`x bigger) into `dst`, averaging
/// `factor * factor` pixels.
pub fn resolve_downsample(
    src: &[u32],
    dst: &mut [u32],
    width: usize,
    height: usize,
    factor: usize,
) {
    if factor <= 1 {
        dst.copy_from_slice(&src[..width * height]);
        return;
    }
    let sw = width * factor;
    for y in 0..height {
        for x in 0..width {
            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            for sy in 0..factor {
                let row = (y * factor + sy) * sw + x * factor;
                for sx in 0..factor {
                    let p = src[row + sx];
                    r += (p >> 16) & 0xFF;
                    g += (p >> 8) & 0xFF;
                    b += p & 0xFF;
                }
            }
            let n = (factor * factor) as u32;
            dst[y * width + x] = ((r / n) << 16) | ((g / n) << 8) | (b / n);
        }
    }
}

/// Simplified SMAA.
///
/// Pass 1 finds luma edges and classifies them horizontal/vertical. Pass 2
/// blends the lit side of each edge toward its contrasting neighbour by half a
/// pixel of coverage, which removes the staircase without a full distance-field
/// search or the lookup textures the real SMAA uses.
pub fn apply_smaa(color: &mut [u32], width: usize, height: usize) {
    let luma: Vec<f32> = color.iter().map(|&p| crate::draw::pixel::luma(p)).collect();
    let src = color.to_vec();
    const THRESHOLD: f32 = 0.06;

    let at = |x: isize, y: isize| -> f32 {
        let x = x.clamp(0, width as isize - 1) as usize;
        let y = y.clamp(0, height as isize - 1) as usize;
        luma[y * width + x]
    };

    for y in 0..height as isize {
        for x in 0..width as isize {
            let c = luma[y as usize * width + x as usize];
            let l = at(x - 1, y);
            let r = at(x + 1, y);
            let u = at(x, y - 1);
            let d = at(x, y + 1);

            let dh = (c - l).abs().max((c - r).abs());
            let dv = (c - u).abs().max((c - d).abs());
            if dh.max(dv) < THRESHOLD {
                continue; // flat region: leave untouched
            }

            // Skip local minima (the background side of an edge). Blending
            // *into* an edge from the dark side is what produces bright halos
            // around bright objects; the lit side alone is enough to soften
            // the silhouette.
            if c <= l && c <= r && c <= u && c <= d {
                continue;
            }

            let (dx, dy) = if dh >= dv {
                // Horizontal edge: blend along x, toward the contrasting side.
                (if (c - l).abs() > (c - r).abs() { -1 } else { 1 }, 0)
            } else {
                (0, if (c - u).abs() > (c - d).abs() { -1 } else { 1 })
            };

            let nx = (x + dx).clamp(0, width as isize - 1) as usize;
            let ny = (y + dy).clamp(0, height as isize - 1) as usize;
            let a = src[y as usize * width + x as usize];
            let b = src[ny * width + nx];
            color[y as usize * width + x as usize] = blend_half(a, b);
        }
    }
}

/// 50/50 blend of two packed colors.
fn blend_half(a: u32, b: u32) -> u32 {
    let r = (((a >> 16) & 0xFF) + ((b >> 16) & 0xFF)) / 2;
    let g = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF)) / 2;
    let bl = ((a & 0xFF) + (b & 0xFF)) / 2;
    (r << 16) | (g << 8) | bl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_averages_four_samples() {
        // 2x2 input to 1x1 output: red, red, blue, blue => 50/50.
        let src = [0xFF0000, 0xFF0000, 0x0000FF, 0x0000FF];
        let mut dst = [0u32; 1];
        resolve_downsample(&src, &mut dst, 1, 1, 2);
        let (r, g, b) = ((dst[0] >> 16) & 0xFF, (dst[0] >> 8) & 0xFF, dst[0] & 0xFF);
        assert_eq!((r, g, b), (127, 0, 127));
    }

    #[test]
    fn downsample_onep_to_one_copies() {
        let src = [1u32, 2, 3, 4];
        let mut dst = [0u32; 4];
        resolve_downsample(&src, &mut dst, 4, 1, 1);
        assert_eq!(dst, src);
    }

    #[test]
    fn smaa_leaves_flat_regions_untouched() {
        let mut img = vec![0x404040u32; 16 * 16];
        let before = img.clone();
        apply_smaa(&mut img, 16, 16);
        assert_eq!(img, before, "uniform image must not change");
    }

    #[test]
    fn smaa_softens_a_hard_edge() {
        // Left half black, right half white.
        let w = 16;
        let h = 4;
        let mut img = vec![0u32; w * h];
        for y in 0..h {
            for x in w / 2..w {
                img[y * w + x] = 0xFFFFFF;
            }
        }
        apply_smaa(&mut img, w, h);
        let edge_before = 0x000000;
        assert_ne!(img[w / 2], edge_before, "edge pixel should be blended");
        let mid = img[w / 2];
        let r = (mid >> 16) & 0xFF;
        assert!(r > 20 && r < 235, "edge should be a gray blend, got {r}");
        // Away from the edge, colors stay pure.
        assert_eq!(img[2], 0);
        assert_eq!(img[w - 2], 0xFFFFFF);
    }
}
