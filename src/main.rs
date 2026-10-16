use minifb::{Key, Window, WindowOptions};
//sorry for swearing

const WIDTH: usize = 800;
const HEIGHT: usize = 800;

#[derive(Clone, Copy)] 
struct Vertex {
    position: [f32; 3], 
    color: [f32; 3], 
}

struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>, 
}


trait FragmentShader {
    fn shade(&self, color: [f32; 3]) -> u32;
}

struct FlatShader;

impl FragmentShader for FlatShader {
    fn shade(&self, color: [f32; 3]) -> u32 {
        let r = (color[0] * 255.0) as u32;
        let g = (color[1] * 255.0) as u32;
        let b = (color[2] * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }
}//ok

fn set_pixel(buffer: &mut [u32], x: i32, y: i32, color: u32) {
    if x < 0 || y < 0 || x >= WIDTH as i32 || y >= HEIGHT as i32 {
        return;
    }
    buffer[y as usize * WIDTH + x as usize] = color;
}

fn init_window() -> Window {
    let mut window = Window::new(
        "AeroGl",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| panic!("sorry window init failed! {e}"));

    window.set_target_fps(120);

    window 
}

fn draw_rect(buffer: &mut [u32], x0: i32, x1: i32, y0: i32, y1: i32, color: u32) {
    for y in y0..y1 {
        for x in x0..x1 {
            set_pixel(buffer, x, y, color);
        }
    }
}
fn draw_line_float(buffer: &mut [u32], x0: f32, y0: f32, x1: f32, y1: f32, color: u32) {
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
fn draw_line(buffer: &mut [u32], x: &mut f32, y: &mut f32, heading: f32, len: f32, color: u32) {
    let nx = *x + heading.cos() * len;
    let ny = *y + heading.sin() * len;
    draw_line_float(buffer, *x, *y, nx, ny, color);
    *x = nx;
    *y = ny;
}

fn main() {
    let mut window = init_window();
    let mut color_buffer = vec![0u32; WIDTH * HEIGHT];

    while window.is_open() && !window.is_key_down(Key::Escape) {        
        color_buffer.fill(0x001A2B3C);

            
        draw_rect(&mut color_buffer, 200, 600, 200, 600, 0x00FF0000);


        window
            .update_with_buffer(&color_buffer, WIDTH, HEIGHT)
            .unwrap();
        

    }
}