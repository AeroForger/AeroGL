// Made by AeroForger

#[derive(Clone, Copy)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

pub trait FragmentShader {
    fn shade(&self, color: [f32; 3]) -> u32;
}

pub struct FlatShader;

impl FragmentShader for FlatShader {
    fn shade(&self, color: [f32; 3]) -> u32 {
        let r = (color[0] * 255.0) as u32;
        let g = (color[1] * 255.0) as u32;
        let b = (color[2] * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }
}
