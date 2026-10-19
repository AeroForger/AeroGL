//! Pixel-level primitives: color packing, edge functions, interpolation.
//! Made by AeroForger

/// Linear interpolation between `a` and `b`.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Signed area of edge `(b - a)` crossed with `(p - a)`.
#[inline]
pub fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

/// Packs `[0,1]` rgb into the `0x00RRGGBB` framebuffer format, clamping.
#[inline]
pub fn rgb(r: f32, g: f32, b: f32) -> u32 {
    let r = (r.clamp(0.0, 1.0) * 255.0) as u32;
    let g = (g.clamp(0.0, 1.0) * 255.0) as u32;
    let b = (b.clamp(0.0, 1.0) * 255.0) as u32;
    (r << 16) | (g << 8) | b
}

/// Unpacks a framebuffer color into `[0,1]` rgb.
#[inline]
pub fn unpack(color: u32) -> [f32; 3] {
    [
        ((color >> 16) & 0xFF) as f32 / 255.0,
        ((color >> 8) & 0xFF) as f32 / 255.0,
        (color & 0xFF) as f32 / 255.0,
    ]
}

/// Perceptual luminance of a packed color, in `[0,1]` (used by SMAA).
#[inline]
pub fn luma(color: u32) -> f32 {
    let [r, g, b] = unpack(color);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_packs_and_unpacks_channels() {
        assert_eq!(rgb(1.0, 0.0, 0.0), 0xFF0000);
        assert_eq!(rgb(0.0, 1.0, 0.0), 0x00FF00);
        assert_eq!(rgb(0.0, 0.0, 1.0), 0x0000FF);
        assert_eq!(rgb(2.0, -1.0, 0.5), rgb(1.0, 0.0, 0.5), "clamps inputs");
        let [r, g, b] = unpack(0x336699);
        assert!((r - 0.2).abs() < 0.01 && (g - 0.4).abs() < 0.01 && (b - 0.6).abs() < 0.01);
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        assert_eq!(lerp(2.0, 10.0, 0.0), 2.0);
        assert_eq!(lerp(2.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(2.0, 10.0, 0.5), 6.0);
    }

    #[test]
    fn luma_weights_green_most() {
        assert!(luma(0x00FF00) > luma(0xFF0000));
        assert!(luma(0xFF0000) > luma(0x0000FF));
    }
}
