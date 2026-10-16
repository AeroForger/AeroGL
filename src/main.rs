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

fn ast(){
    //will build the mesh from inputs
    // app flow:
    // cli -> this shit -> window init -> calculations -> drawing -> resetting
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

fn draw_triangle(){

}

fn main() {
    let mut window = init_window();

    let mut color_buffer = vec![0u32; WIDTH * HEIGHT];

    while window.is_open() && !window.is_key_down(Key::Escape) {        // Same result as the loop, clearer and usually just as fast.
        color_buffer.fill(0x001A2B3C);

        for y in 200..600 {
            for x in 300..500 {
                set_pixel(&mut color_buffer, x, y, 0x00FF0000);
            }
        }

        window
            .update_with_buffer(&color_buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}