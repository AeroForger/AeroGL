// Made by AeroForger

use crate::draw;

pub fn birdie_move_float(
    buffer: &mut [u32],
    x: &mut f32,
    y: &mut f32,
    heading: f32,
    len: f32,
    color: u32,
) {
    let nx = *x + heading.cos() * len;
    let ny = *y + heading.sin() * len;
    draw::draw_line_float(buffer, *x, *y, nx, ny, color);
    *x = nx;
    *y = ny;
}
pub fn birdie_move(
    buffer: &mut [u32],
    x: &mut f32,
    y: &mut f32,
    heading: f32,
    len: f32,
    color: u32,
) {
    let nx = *x + heading.cos() * len;
    let ny = *y + heading.sin() * len;
    draw::draw_line_bresenham(
        buffer,
        x.round() as i32,
        y.round() as i32,
        nx.round() as i32,
        ny.round() as i32,
        color,
    );
    *x = nx;
    *y = ny;
}
pub fn birdie_goto(buffer: &mut [u32], x: &mut f32, y: &mut f32, tx: f32, ty: f32, color: u32) {
    draw::draw_line_bresenham(
        buffer,
        x.round() as i32,
        y.round() as i32,
        tx.round() as i32,
        ty.round() as i32,
        color,
    );
    *x = tx;
    *y = ty;
}
