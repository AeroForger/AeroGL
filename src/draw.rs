// Made by AeroForger

use crate::window::{HEIGHT, WIDTH};

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
pub fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

pub fn set_pixel(buffer: &mut [u32], x: i32, y: i32, color: u32) {
    if x < 0 || y < 0 || x >= WIDTH as i32 || y >= HEIGHT as i32 {
        return;
    }
    buffer[y as usize * WIDTH + x as usize] = color;
}

pub fn draw_rect(buffer: &mut [u32], x0: i32, x1: i32, y0: i32, y1: i32, color: u32) {
    for y in y0..y1 {
        for x in x0..x1 {
            set_pixel(buffer, x, y, color);
        }
    }
}
pub fn draw_line_bresenham(buffer: &mut [u32], x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let (mut x, mut y) = (x0, y0);
    loop {
        set_pixel(buffer, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}
pub fn draw_line_float(buffer: &mut [u32], x0: f32, y0: f32, x1: f32, y1: f32, color: u32) {
    let dx = x1 - x0;
    let dy = y1 - y0;

    let steps = dx.abs().max(dy.abs()).round() as i32;

    if steps == 0 {
        set_pixel(buffer, x0.round() as i32, y0.round() as i32, color);
        return;
    }

    let x_inc = dx / steps as f32;
    let y_inc = dy / steps as f32;

    let mut x = x0;
    let mut y = y0;

    for _ in 0..=steps {
        set_pixel(buffer, x.round() as i32, y.round() as i32, color);
        x += x_inc;
        y += y_inc;
    }
}

pub fn draw_triangle_filled(
    buffer: &mut [u32],
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
    color: u32,
) {
    let min_x = a.0.min(b.0).min(c.0).floor() as i32;
    let max_x = a.0.max(b.0).max(c.0).ceil() as i32;
    let min_y = a.1.min(b.1).min(c.1).floor() as i32;
    let max_y = a.1.max(b.1).max(c.1).ceil() as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let w0 = edge(b.0, b.1, c.0, c.1, px, py);
            let w1 = edge(c.0, c.1, a.0, a.1, px, py);
            let w2 = edge(a.0, a.1, b.0, b.1, px, py);

            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                set_pixel(buffer, x, y, color);
            }
        }
    }
}
