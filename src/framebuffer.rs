//! Framebuffer and render target: the pixel + depth storage every pass writes.
//! Made by AeroForger

/// A color + depth buffer with fixed dimensions.
///
/// Colors are packed `0x00RRGGBB`. Depth is NDC depth (near = -1, far = +1),
/// so smaller values are closer.
pub struct Framebuffer {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Packed `0x00RRGGBB` pixels, row-major, top-left origin.
    pub color: Vec<u32>,
    /// NDC depth values, same layout as `color`.
    pub depth: Vec<f32>,
}

impl Framebuffer {
    /// Creates a framebuffer. Does not allocate more than `width * height`.
    pub fn new(width: usize, height: usize) -> Self {
        let len = width * height;
        Self {
            width,
            height,
            color: vec![0; len],
            depth: vec![f32::INFINITY; len],
        }
    }

    /// Resizes in place if the dimensions changed (no-op otherwise).
    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let len = width * height;
        self.color.resize(len, 0);
        self.depth.resize(len, f32::INFINITY);
    }

    /// Clears color to `background` and depth to infinity.
    ///
    /// Large buffers are cleared in parallel. Filling a 4K color and depth
    /// buffer is mostly memory bandwidth, so it is one of the few passes that
    /// scales well with cores.
    pub fn clear(&mut self, background: u32) {
        let len = self.color.len();
        let threads = crate::parallel::threads_for(len, 500_000);
        if threads <= 1 {
            self.color.fill(background);
            self.depth.fill(f32::INFINITY);
            return;
        }
        let chunk = len.div_ceil(threads);
        std::thread::scope(|scope| {
            for (color, depth) in self
                .color
                .chunks_mut(chunk)
                .zip(self.depth.chunks_mut(chunk))
            {
                scope.spawn(move || {
                    color.fill(background);
                    depth.fill(f32::INFINITY);
                });
            }
        });
    }

    /// Borrows the buffers as a [`Target`] for the rasterizer.
    pub fn target(&mut self) -> Target<'_> {
        Target::new(self.width, self.height, &mut self.color, &mut self.depth)
    }

    /// The pixel slice, for presenting or post-processing.
    pub fn pixels(&self) -> &[u32] {
        &self.color
    }
}

/// A borrowed view over color + depth, with the dimensions needed to index them.
///
/// This is what every rasterizer call writes into, so the library never assumes
/// a fixed resolution.
pub struct Target<'a> {
    /// Width in pixels.
    pub width: usize,
    /// Full screen height in pixels. Projection and clipping use this, even
    /// when this target only owns a horizontal band.
    pub height: usize,
    /// Packed `0x00RRGGBB` pixels for this band, row-major.
    pub color: &'a mut [u32],
    /// NDC depth values (smaller is closer) for this band.
    pub depth: &'a mut [f32],
    /// Whether shading passes may update the depth buffer. The sorted
    /// transparent pass turns this off so later geometry still resolves
    /// against the opaque depth, but glass does not occlude other glass.
    pub depth_write: bool,
    /// First screen row owned by this target.
    y_offset: usize,
    /// Number of rows owned by this target.
    rows: usize,
}

impl<'a> Target<'a> {
    /// Builds a full-screen target from raw slices. Panics if the lengths
    /// don't match.
    pub fn new(width: usize, height: usize, color: &'a mut [u32], depth: &'a mut [f32]) -> Self {
        Self::band(width, height, 0, height, color, depth)
    }

    /// Builds a horizontal band of a larger screen.
    ///
    /// `height` is still the full screen height, so projection matches the
    /// other bands. `color` and `depth` only need to hold `rows * width`
    /// pixels. This is what lets a software renderer split a frame across
    /// threads.
    pub fn band(
        width: usize,
        height: usize,
        y_offset: usize,
        rows: usize,
        color: &'a mut [u32],
        depth: &'a mut [f32],
    ) -> Self {
        assert!(width > 0 && height > 0, "target needs a positive size");
        assert!(rows > 0, "target band needs at least one row");
        assert!(y_offset + rows <= height, "band must fit inside the screen");
        let len = width * rows;
        assert_eq!(color.len(), len, "color buffer must be width * rows");
        assert_eq!(depth.len(), len, "depth buffer must be width * rows");
        Self {
            width,
            height,
            color,
            depth,
            depth_write: true,
            y_offset,
            rows,
        }
    }

    /// First screen row owned by this target.
    #[inline]
    pub fn band_offset(&self) -> usize {
        self.y_offset
    }

    /// Number of screen rows owned by this target.
    #[inline]
    pub fn band_rows(&self) -> usize {
        self.rows
    }

    /// One past the last screen row owned by this target.
    #[inline]
    pub fn band_end(&self) -> usize {
        self.y_offset + self.rows
    }

    /// Whether a screen row is inside this band.
    #[inline]
    pub fn contains_row(&self, y: i32) -> bool {
        y >= self.y_offset as i32 && y < self.band_end() as i32
    }

    /// Splits this target into up to `count` horizontal bands.
    ///
    /// Useful with [`std::thread::scope`]: the returned targets own disjoint
    /// pixel and depth slices, so they can be rendered in parallel. The split
    /// follows this target's own band, so nested bands are fine.
    pub fn split_bands(&mut self, count: usize) -> Vec<Target<'_>> {
        let count = count.clamp(1, self.rows);
        let base = self.rows / count;
        let extra = self.rows % count;
        let mut color: &mut [u32] = &mut self.color[..];
        let mut depth: &mut [f32] = &mut self.depth[..];
        let mut out = Vec::with_capacity(count);
        let mut y = self.y_offset;
        for i in 0..count {
            let rows = base + usize::from(i < extra);
            let len = rows * self.width;
            let (band_color, rest_color) = color.split_at_mut(len);
            let (band_depth, rest_depth) = depth.split_at_mut(len);
            out.push(Target {
                width: self.width,
                height: self.height,
                color: band_color,
                depth: band_depth,
                depth_write: self.depth_write,
                y_offset: y,
                rows,
            });
            color = rest_color;
            depth = rest_depth;
            y += rows;
        }
        out
    }

    /// Clears color to `background` and depth to infinity.
    pub fn clear(&mut self, background: u32) {
        self.color.fill(background);
        self.depth.fill(f32::INFINITY);
    }

    /// Enables or disables depth writes for subsequent draws.
    pub fn set_depth_write(&mut self, enabled: bool) {
        self.depth_write = enabled;
    }

    /// Writes one pixel, ignoring out-of-bounds coordinates.
    #[inline]
    pub fn set_pixel(&mut self, x: i32, y: i32, color: u32) {
        if (0..self.width as i32).contains(&x) && self.contains_row(y) {
            let idx = (y as usize - self.y_offset) * self.width + x as usize;
            self.color[idx] = color;
        }
    }

    /// Reads one pixel, or the background color when out of bounds.
    #[inline]
    pub fn pixel(&self, x: i32, y: i32) -> u32 {
        if (0..self.width as i32).contains(&x) && self.contains_row(y) {
            let idx = (y as usize - self.y_offset) * self.width + x as usize;
            self.color[idx]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_resets_color_and_depth() {
        let mut fb = Framebuffer::new(4, 3);
        fb.clear(0x112233);
        assert_eq!(fb.color, vec![0x112233; 12]);
        assert!(fb.depth.iter().all(|d| d.is_infinite()));
    }

    #[test]
    fn target_bounds_check() {
        let mut fb = Framebuffer::new(4, 4);
        let mut t = fb.target();
        t.set_pixel(-1, 0, 7);
        t.set_pixel(4, 0, 7);
        t.set_pixel(0, 4, 7);
        t.set_pixel(1, 1, 9);
        assert_eq!(fb.color[5], 9);
        assert_eq!(fb.color.iter().filter(|&&p| p == 7).count(), 0);
    }

    #[test]
    fn resize_reallocates_only_when_needed() {
        let mut fb = Framebuffer::new(4, 4);
        fb.clear(1);
        fb.resize(4, 4);
        assert_eq!(fb.color[0], 1, "same size is a no-op");
        fb.resize(2, 2);
        assert_eq!(fb.color.len(), 4);
        assert_eq!(fb.depth.len(), 4);
    }
}
