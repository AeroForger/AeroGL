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
    pub fn clear(&mut self, background: u32) {
        self.color.fill(background);
        self.depth.fill(f32::INFINITY);
    }

    /// Borrows the buffers as a [`Target`] for the rasterizer.
    pub fn target(&mut self) -> Target<'_> {
        Target {
            width: self.width,
            height: self.height,
            color: &mut self.color,
            depth: &mut self.depth,
        }
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
    /// Height in pixels.
    pub height: usize,
    /// Packed `0x00RRGGBB` pixels.
    pub color: &'a mut [u32],
    /// NDC depth values (smaller is closer).
    pub depth: &'a mut [f32],
}

impl<'a> Target<'a> {
    /// Builds a target from raw slices. Panics if the lengths don't match.
    pub fn new(width: usize, height: usize, color: &'a mut [u32], depth: &'a mut [f32]) -> Self {
        let len = width * height;
        assert_eq!(color.len(), len, "color buffer must be width * height");
        assert_eq!(depth.len(), len, "depth buffer must be width * height");
        Self {
            width,
            height,
            color,
            depth,
        }
    }

    /// Clears color to `background` and depth to infinity.
    pub fn clear(&mut self, background: u32) {
        self.color.fill(background);
        self.depth.fill(f32::INFINITY);
    }

    /// Writes one pixel, ignoring out-of-bounds coordinates.
    #[inline]
    pub fn set_pixel(&mut self, x: i32, y: i32, color: u32) {
        if (0..self.width as i32).contains(&x) && (0..self.height as i32).contains(&y) {
            self.color[y as usize * self.width + x as usize] = color;
        }
    }

    /// Reads one pixel, or the background color when out of bounds.
    #[inline]
    pub fn pixel(&self, x: i32, y: i32) -> u32 {
        if (0..self.width as i32).contains(&x) && (0..self.height as i32).contains(&y) {
            self.color[y as usize * self.width + x as usize]
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
