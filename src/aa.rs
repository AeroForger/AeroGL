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
    let band = |src: &[u32], dst: &mut [u32], rows: usize| match factor {
        2 => resolve2(src, dst, width, rows),
        4 => resolve4(src, dst, width, rows),
        _ => resolve_generic(src, dst, width, rows, factor),
    };

    let threads = crate::parallel::threads_for(width * height, 400_000);
    if threads <= 1 {
        band(src, dst, height);
        return;
    }

    // Each output row only reads its own factor x factor block of the source,
    // so the source and destination can be split into matching row bands.
    let rows_per = height.div_ceil(threads);
    let mut dst_rest: &mut [u32] = dst;
    let mut y = 0;
    std::thread::scope(|scope| {
        while y < height {
            let rows = rows_per.min(height - y);
            let out_len = rows * width;
            let src_len = rows * factor * factor * width;
            let (out_band, out_tail) = dst_rest.split_at_mut(out_len);
            let src_off = y * factor * factor * width;
            let src_band = &src[src_off..src_off + src_len];
            scope.spawn(move || band(src_band, out_band, rows));
            dst_rest = out_tail;
            y += rows;
        }
    });
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
/// Reusing these between frames keeps SMAA from allocating full-size buffers
/// every time it runs.
pub(crate) struct SmaaScratch {
    luma: Vec<u8>,
    /// One byte per tile: non-zero when the tile plus its one-pixel halo has
    /// enough luma contrast to contain an edge.
    tiles: Vec<u8>,
    /// Reusable per-band lists of `(local pixel index, blended color)`.
    ///
    /// The edge pass reads the untouched frame, so it can defer its writes and
    /// avoid copying the whole color buffer just to keep neighbor reads stable.
    changes: Vec<Vec<u64>>,
}

impl SmaaScratch {
    pub(crate) fn new() -> Self {
        Self {
            luma: Vec::new(),
            tiles: Vec::new(),
            changes: Vec::new(),
        }
    }
}

/// Edge tiles are this many pixels square. A flat tile lets the blend pass
/// skip a whole block instead of branching on every pixel.
const TILE: usize = 16;

/// Minimum luma difference that counts as an edge.
const THRESHOLD: u8 = 15;

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
    let threads = crate::parallel::threads_for(len, 200_000);

    // Luma only. The edge pass reads the original color buffer directly and
    // records its blends, so no full-frame copy is needed.
    if threads <= 1 {
        for (lum, &p) in scratch.luma.iter_mut().zip(color.iter()) {
            *lum = luma_u8(p);
        }
    } else {
        let chunk = len.div_ceil(threads);
        let luma = &mut scratch.luma;
        std::thread::scope(|scope| {
            for (luma_chunk, color_chunk) in luma.chunks_mut(chunk).zip(color.chunks(chunk)) {
                scope.spawn(move || {
                    for (lum, &p) in luma_chunk.iter_mut().zip(color_chunk) {
                        *lum = luma_u8(p);
                    }
                });
            }
        });
    }

    // A flat tile cannot contain an edge even at its border, as long as the
    // scan includes a one-pixel halo. Building this tiny map first lets the
    // blend pass skip most of a typical frame (sky, large faces, empty ground).
    let tile_w = width.div_ceil(TILE);
    let tile_h = height.div_ceil(TILE);
    scratch.tiles.resize(tile_w * tile_h, 0);
    let tile_threads = crate::parallel::threads_for(tile_w * tile_h, 256);
    if tile_threads <= 1 {
        fill_tiles(
            &mut scratch.tiles,
            &scratch.luma,
            width,
            height,
            tile_w,
            0,
            tile_h,
        );
    } else {
        let rows_per = tile_h.div_ceil(tile_threads);
        let luma = &scratch.luma;
        let mut tiles: &mut [u8] = &mut scratch.tiles;
        let mut ty = 0;
        std::thread::scope(|scope| {
            while ty < tile_h {
                let rows = rows_per.min(tile_h - ty);
                let (band, tail) = tiles.split_at_mut(rows * tile_w);
                scope.spawn(move || {
                    fill_tiles(band, luma, width, height, tile_w, ty, rows);
                });
                tiles = tail;
                ty += rows;
            }
        });
    }

    // Edge pass. Each band records its blends instead of writing into the
    // frame while neighbors are still being read. The change lists are reused
    // between frames and are much smaller than a full-frame copy.
    let tiles = &scratch.tiles;
    let luma = &scratch.luma;
    let band_count = if threads <= 1 { 1 } else { threads };
    if scratch.changes.len() < band_count {
        scratch.changes.resize_with(band_count, Vec::new);
    }
    let changes = &mut scratch.changes[..band_count];
    let rows_per = height.div_ceil(band_count);
    let color_read: &[u32] = color;

    if threads <= 1 {
        let change_buf = &mut changes[0];
        change_buf.clear();
        smaa_changes(
            color_read, luma, tiles, tile_w, width, height, 0, height, change_buf,
        );
    } else {
        let mut y = 0;
        let mut change_iter = changes.iter_mut();
        std::thread::scope(|scope| {
            while y < height {
                let rows = rows_per.min(height - y);
                let change_buf = change_iter.next().expect("one change list per band");
                scope.spawn(move || {
                    change_buf.clear();
                    smaa_changes(
                        color_read, luma, tiles, tile_w, width, height, y, rows, change_buf,
                    );
                });
                y += rows;
            }
        });
    }

    // Applying the changes is cheap because only edge pixels were recorded.
    let mut y = 0;
    for change_buf in changes.iter() {
        let rows = rows_per.min(height - y);
        if !change_buf.is_empty() {
            let band = &mut color[y * width..(y + rows) * width];
            apply_changes(band, change_buf);
        }
        y += rows;
    }
}

/// Marks which tiles (plus a one-pixel halo) contain enough luma contrast to
/// need the blend pass.
fn fill_tiles(
    tiles: &mut [u8],
    luma: &[u8],
    width: usize,
    height: usize,
    tile_w: usize,
    ty_start: usize,
    ty_rows: usize,
) {
    for local in 0..ty_rows {
        let ty = ty_start + local;
        let y0 = ty * TILE;
        let ys = y0.saturating_sub(1);
        let ye = (y0 + TILE + 1).min(height);
        for tx in 0..tile_w {
            let x0 = tx * TILE;
            let xs = x0.saturating_sub(1);
            let xe = (x0 + TILE + 1).min(width);
            let (mut min, mut max) = (u8::MAX, 0u8);
            for y in ys..ye {
                let row = &luma[y * width + xs..y * width + xe];
                let (rmin, rmax) = row
                    .iter()
                    .fold((u8::MAX, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                min = min.min(rmin);
                max = max.max(rmax);
            }
            tiles[local * tile_w + tx] = u8::from(max - min >= THRESHOLD);
        }
    }
}

/// Finds SMAA edges over `rows` rows starting at `y_start` and records the
/// blended colors in `changes`.
///
/// `color`, `luma` and `tiles` are the full-size buffers, so neighbor reads
/// always see the untouched frame. Each entry packs a local pixel index in the
/// low 32 bits and the new color in the high 32 bits.
#[allow(clippy::too_many_arguments)] // pixel buffers plus band geometry
fn smaa_changes(
    color: &[u32],
    luma: &[u8],
    tiles: &[u8],
    tile_w: usize,
    width: usize,
    height: usize,
    y_start: usize,
    rows: usize,
    changes: &mut Vec<u64>,
) {
    let y0 = y_start.max(1);
    let y1 = (y_start + rows).min(height - 1);
    for y in y0..y1 {
        let row = y * width;
        let up = row - width;
        let down = row + width;
        let out_row = (y - y_start) * width;
        let ty = y / TILE;
        let tile_row = &tiles[ty * tile_w..ty * tile_w + tile_w];

        for (tx, &active) in tile_row.iter().enumerate() {
            if active == 0 {
                continue; // no contrast anywhere in this tile and its halo
            }
            let xa = (tx * TILE).max(1);
            let xb = ((tx + 1) * TILE).min(width - 1);
            if xa >= xb {
                continue;
            }
            // Carry l and c along the row. Only the right, up and down
            // neighbours need fresh loads, which drops the edge scan from five
            // luma reads per pixel to three.
            let start = row + xa;
            let mut l = luma[start - 1];
            let mut c = luma[start];
            for x in xa..xb {
                let i = row + x;
                let r = luma[i + 1];
                let u = luma[up + x];
                let d = luma[down + x];

                let dl = c.abs_diff(l);
                let dr = c.abs_diff(r);
                let du = c.abs_diff(u);
                let dd = c.abs_diff(d);
                let dh = dl.max(dr);
                let dv = du.max(dd);
                if dh.max(dv) >= THRESHOLD
                    // Skip local minima (the background side of an edge).
                    // Blending into an edge from the dark side is what
                    // produces bright halos around bright objects; the lit
                    // side alone is enough to soften the silhouette.
                    && !(c <= l && c <= r && c <= u && c <= d)
                {
                    let (dx, dy) = if dh >= dv {
                        // Horizontal edge: blend along x, toward the contrast.
                        (if dl > dr { -1isize } else { 1 }, 0)
                    } else {
                        (0, if du > dd { -1isize } else { 1 })
                    };

                    let ni = (i as isize + dy * width as isize + dx) as usize;
                    let blended = blend_half(color[i], color[ni]);
                    changes.push(((out_row + x) as u64) | (u64::from(blended) << 32));
                }

                l = c;
                c = r;
            }
        }
    }
}

/// Writes the recorded blends into one band.
fn apply_changes(band: &mut [u32], changes: &[u64]) {
    for &change in changes {
        let index = (change & 0xFFFF_FFFF) as usize;
        band[index] = (change >> 32) as u32;
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

    /// The tile map is only a shortcut. This reference does the same blend over
    /// every pixel, with no tiling, so the two must agree exactly.
    #[cfg(test)]
    fn smaa_reference(color: &mut [u32], width: usize, height: usize) {
        if width < 3 || height < 3 {
            return;
        }
        let src = color.to_vec();
        let luma: Vec<u8> = src.iter().map(|&p| luma_u8(p)).collect();
        for y in 1..height - 1 {
            let row = y * width;
            let up = row - width;
            let down = row + width;
            for x in 1..width - 1 {
                let i = row + x;
                let c = luma[i];
                let (l, r) = (luma[i - 1], luma[i + 1]);
                let (u, d) = (luma[up + x], luma[down + x]);
                let dh = c.abs_diff(l).max(c.abs_diff(r));
                let dv = c.abs_diff(u).max(c.abs_diff(d));
                if dh.max(dv) < THRESHOLD || (c <= l && c <= r && c <= u && c <= d) {
                    continue;
                }
                let (dx, dy) = if dh >= dv {
                    (
                        if c.abs_diff(l) > c.abs_diff(r) {
                            -1isize
                        } else {
                            1
                        },
                        0,
                    )
                } else {
                    (
                        0,
                        if c.abs_diff(u) > c.abs_diff(d) {
                            -1isize
                        } else {
                            1
                        },
                    )
                };
                let ni = (i as isize + dy * width as isize + dx) as usize;
                color[i] = blend_half(src[i], src[ni]);
            }
        }
    }

    #[test]
    fn smaa_tiles_match_the_full_scan() {
        // Edges at many offsets, including tile boundaries and diagonals.
        let w = 71;
        let h = 53;
        let mut img = vec![0x202028u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let on = (x * 7 + y * 3) % 29 < 13
                    || (x % 16 == 0)
                    || (y % 16 == 0)
                    || (x + y) % 23 == 0;
                if on {
                    img[y * w + x] = 0xE8E0D0;
                }
            }
        }
        let mut tiled = img.clone();
        apply_smaa(&mut tiled, w, h);
        let mut reference = img;
        smaa_reference(&mut reference, w, h);
        assert_eq!(tiled, reference, "tile culling changed the SMAA result");
    }

    #[test]
    fn smaa_parallel_matches_serial() {
        use crate::parallel::set_thread_limit_for_tests;

        let (w, h) = (700, 700);
        let mut img = vec![0x202028u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let on = (x * 7 + y * 3) % 29 < 13
                    || (x % 16 == 0)
                    || (y % 16 == 0)
                    || (x + y) % 23 == 0
                    || (x * x + y * y) % 101 < 30;
                if on {
                    img[y * w + x] = 0xE8E0D0;
                }
            }
        }

        set_thread_limit_for_tests(1);
        let mut serial = img.clone();
        let mut scratch = SmaaScratch::new();
        apply_smaa_with_scratch(&mut serial, w, h, &mut scratch);

        set_thread_limit_for_tests(8);
        let mut parallel = img;
        let mut scratch = SmaaScratch::new();
        apply_smaa_with_scratch(&mut parallel, w, h, &mut scratch);
        set_thread_limit_for_tests(0);

        assert_eq!(serial, parallel, "SMAA differs across thread counts");
    }
}
