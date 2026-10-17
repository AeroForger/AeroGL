// main.rs
// Made by AeroForger

mod birdie;
mod draw;
mod mesh;
mod window;

use minifb::Key;

fn main() {
    let (mut window, mut color_buffer) = window::init_window();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        color_buffer.fill(0x001A2B3C);

        let tx = 400.0;
        let ty = 400.0;
        // birdie_move(&mut color_buffer, &mut tx, &mut ty, std::f32::consts::FRAC_PI_4, 150.0, 0x00FFFF00); // down-right, 45°
        // birdie_move(&mut color_buffer, &mut tx, &mut ty, 3.0 * std::f32::consts::FRAC_PI_4, 150.0, 0x00FF00FF); // down-left, 135°
        // birdie_move(&mut color_buffer, &mut tx, &mut ty, -std::f32::consts::FRAC_PI_2, 212.0, 0x00FFFFFF); // straight up, 270° (or -90°)
        // birdie_goto(&mut color_buffer, &mut tx, &mut ty, 506.0, 506.0, 0x00FFFF00);
        // birdie_goto(&mut color_buffer, &mut tx, &mut ty, 400.0, 612.0, 0x00FF00FF);
        // birdie_goto(&mut color_buffer, &mut tx, &mut ty, 400.0, 400.0, 0x00FFFFFF);
        draw::draw_triangle_filled(
            &mut color_buffer,
            (400.0, 400.0),
            (506.0, 506.0),
            (400.0, 612.0),
            0x00FFFF00,
        );
        draw::draw_triangle_filled(
            &mut color_buffer,
            (400.0, 400.0),
            (206.0, 206.0),
            (400.0, 612.0),
            0x00FF00FF,
        );
        //draw_rect(&mut color_buffer, 200, 600, 200, 600, 0x00FF0000);

        window
            .update_with_buffer(&color_buffer, window::WIDTH, window::HEIGHT)
            .unwrap();
    }
}
