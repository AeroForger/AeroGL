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
    match factor {
        0 | 1 => dst.copy_from_slice(&src[..width * height]),
        2 => resolve2(src, dst, width, height),
        4 => resolve4(src, dst, width, height),
        _ => resolve_generic(src, dst, width, height, factor),
    }
}

/// 2x2 box resolve. Separate loops and fixed-size chunk iterators let the
/// compiler drop the bounds checks and the per-pixel divide.
fn resolve2(src: &[u32], dst: &mut [u32], width: usize, height: usize) {
    let sw = width * 2;
    for y in 0..height {
        let row = &src[y * 2 * sw..(y * 2 + 2) * sw];
        let (r0, r1) = row.split_at(sw);
        let out = &mut dst[y * width..(y + 1) * width];
        let (a_chunks, _) = r0.as_chunks::<2>();
        let (b_chunks, _) = r1.as_chunks::<2>();
        for (o, (a, b)) in out.iter_mut().zip(a_chunks.iter().zip(b_chunks.iter())) {
            let r = ((a[0] >> 16) & 0xFF)
                + ((a[1] >> 16) & 0xFF)
                + ((b[0] >> 16) & 0xFF)
                + ((b[1] >> 16) & 0xFF);
            let g = ((a[0] >> 8) & 0xFF)
                + ((a[1] >> 8) & 0xFF)
                + ((b[0] >> 8) & 0xFF)
                + ((b[1] >> 8) & 0xFF);
            let bl = (a[0] & 0xFF) + (a[1] & 0xFF) + (b[0] & 0xFF) + (b[1] & 0xFF);
            *o = ((r >> 2) << 16) | ((g >> 2) << 8) | (bl >> 2);
        }
    }
}

/// 4x4 box resolve.
fn resolve4(src: &[u32], dst: &mut [u32], width: usize, height: usize) {
    let sw = width * 4;
    for y in 0..height {
        let row = &src[y * 4 * sw..(y * 4 + 4) * sw];
        let out = &mut dst[y * width..(y + 1) * width];
        for (x, o) in out.iter_mut().enumerate() {
            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            for sy in 0..4 {
                let line = &row[sy * sw + x * 4..sy * sw + x * 4 + 4];
                for &p in line {
                    r += (p >> 16) & 0xFF;
                    g += (p >> 8) & 0xFF;
                    b += p & 0xFF;
                }
            }
            *o = ((r >> 4) << 16) | ((g >> 4) << 8) | (b >> 4);
        }
    }
}

fn resolve_generic(src: &[u32], dst: &mut [u32], width: usize, height: usize, factor: usize) {
    let sw = width * factor;
    let n = (factor * factor) as u32;
    for y in 0..height {
        let out_row = y * width;
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
            dst[out_row + x] = ((r / n) << 16) | ((g / n) << 8) | (b / n);
        }
    }
}

/// Scratch buffers for [`apply_smaa_with_scratch`].
///
/// Reusing this between frames keeps SMAA from allocating a full-size buffer
/// every time it runs.
pub(crate) struct SmaaScratch {
    luma: Vec<u8>,
}

impl SmaaScratch {
    pub(crate) fn new() -> Self {
        Self { luma: Vec::new() }
    }
}

/// Simplified SMAA.
///
/// Pass 1 finds luma edges and classifies them horizontal/vertical. Pass 2
/// blends the lit side of each edge toward its contrasting neighbour by half a
/// pixel of coverage, which removes the staircase without a full distance-field
/// search or the lookup textures the real SMAA uses.
#[cfg(test)]
fn apply_smaa(color: &mut [u32], width: usize, height: usize) {
    let mut scratch = SmaaScratch::new();
    apply_smaa_with_scratch(color, width, height, &mut scratch);
}

/// Same as [`apply_smaa`], but reuses the supplied buffers.
pub(crate) fn apply_smaa_with_scratch(
    color: &mut [u32],
    width: usize,
    height: usize,
    scratch: &mut SmaaScratch,
) {
    // The outer pixel ring is left alone. It is the edge of the image, so AA
    // there is invisible, and skipping it lets the inner loop use direct
    // neighbor indices instead of clamping every lookup.
    if width < 3 || height < 3 {
        return;
    }
    let len = width * height;
    scratch.luma.resize(len, 0);
    for (l, &p) in scratch.luma.iter_mut().zip(color.iter()) {
        *l = luma_u8(p);
    }

    const THRESHOLD: u8 = 15;
    let luma = &scratch.luma;

    for y in 1..height - 1 {
        let row = y * width;
        let up = row - width;
        let down = row + width;
        for x in 1..width - 1 {
            let i = row + x;
            let c = luma[i];
            let l = luma[i - 1];
            let r = luma[i + 1];
            let u = luma[up + x];
            let d = luma[down + x];

            let dl = c.abs_diff(l);
            let dr = c.abs_diff(r);
            let du = c.abs_diff(u);
            let dd = c.abs_diff(d);
            let dh = dl.max(dr);
            let dv = du.max(dd);
            if dh.max(dv) < THRESHOLD {
                continue; // flat region: leave untouched
            }

            // Skip local minima (the background side of an edge). Blending
            // into an edge from the dark side is what produces bright halos
            // around bright objects; the lit side alone is enough to soften
            // the silhouette.
            if c <= l && c <= r && c <= u && c <= d {
                continue;
            }

            let (dx, dy) = if dh >= dv {
                // Horizontal edge: blend along x, toward the contrasting side.
                (if dl > dr { -1isize } else { 1 }, 0)
            } else {
                (0, if du > dd { -1isize } else { 1 })
            };

            let ni = (i as isize + dy * width as isize + dx) as usize;
            color[i] = blend_half(color[i], color[ni]);
        }
    }
}

/// Integer luma in 0..=255. Same weights as `draw::pixel::luma`, but avoiding
/// a float round trip makes the SMAA edge pass noticeably cheaper.
#[inline]
fn luma_u8(color: u32) -> u8 {
    let r = (color >> 16) & 0xFF;
    let g = (color >> 8) & 0xFF;
    let b = color & 0xFF;
    ((r * 54 + g * 182 + b * 19) >> 8) as u8
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
        // The outer pixel ring is skipped, so inspect row 1.
        let row = w;
        let mid = img[row + w / 2];
        let r = (mid >> 16) & 0xFF;
        assert!(r > 20 && r < 235, "edge should be a gray blend, got {r}");
        // Away from the edge, colors stay pure.
        assert_eq!(img[row + 2], 0);
        assert_eq!(img[row + w - 2], 0xFFFFFF);
    }
}
