// Made by AeroForger

use minifb::{Window, WindowOptions};

pub const WIDTH: usize = 800;
pub const HEIGHT: usize = 800;

// Window setup
pub fn init_window() -> (Window, Vec<u32>) {
    let mut window = Window::new("AeroGl", WIDTH, HEIGHT, WindowOptions::default())
        .unwrap_or_else(|e| panic!("sorry window init failed! {e}"));

    window.set_target_fps(120);
    let color_buffer = vec![0u32; WIDTH * HEIGHT];
    (window, color_buffer)
}
