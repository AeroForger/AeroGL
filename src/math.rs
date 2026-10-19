//! Vec3 and Mat4: the minimum math you need for a 3D pipeline.
//! Made by AeroForger

use std::ops::{Add, Div, Index, Mul, Neg, Sub};

/// A 3-component vector / point in space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

#[allow(dead_code)] // small vector library: the full API is intentional
impl Vec3 {
    /// The zero vector.
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    /// The vector `(1, 1, 1)`.
    pub const ONE: Vec3 = Vec3::new(1.0, 1.0, 1.0);

    /// Builds a vector.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector with every component set to `v`.
    pub fn splat(v: f32) -> Self {
        Self::new(v, v, v)
    }

    /// Component-wise addition.
    pub fn add(&self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    /// Component-wise subtraction.
    pub fn sub(&self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    /// Multiplies every component by `s`.
    pub fn scale(&self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot product.
    pub fn dot(&self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Right-handed cross product.
    pub fn cross(&self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Euclidean length.
    pub fn length(&self) -> f32 {
        self.dot(*self).sqrt()
    }

    /// Unit vector, or zero for a zero-length vector.
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > f32::EPSILON {
            self.scale(1.0 / len)
        } else {
            Self::ZERO
        }
    }

    /// Linear interpolation toward `o`.
    pub fn lerp(&self, o: Self, t: f32) -> Self {
        self.add(o.sub(*self).scale(t))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Self) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Self) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        self.scale(s)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Self::ZERO.sub(self)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f32) -> Vec3 {
        self.scale(1.0 / s)
    }
}

/// A 4x4 row-major matrix, used for model, view, and projection transforms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    /// Row-major elements: `m[row][column]`.
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    /// The all-zero matrix.
    pub const ZERO: Mat4 = Mat4 { m: [[0.0; 4]; 4] };

    /// The identity matrix.
    #[allow(clippy::needless_range_loop)] // i is both the loop var and the row
    pub fn identity() -> Self {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            m[i][i] = 1.0;
        }
        Self { m }
    }

    /// Returns `self * other` (apply `other` first, then `self`).
    pub fn multiply(&self, other: &Mat4) -> Mat4 {
        let mut out = Mat4::ZERO;
        for r in 0..4 {
            for c in 0..4 {
                out.m[r][c] = self.m[r][0] * other.m[0][c]
                    + self.m[r][1] * other.m[1][c]
                    + self.m[r][2] * other.m[2][c]
                    + self.m[r][3] * other.m[3][c];
            }
        }
        out
    }

    /// Transforms a point to clip space, returning `(x, y, z, w)` without the
    /// perspective divide, which clipping needs.
    pub fn transform_h(&self, p: Vec3) -> [f32; 4] {
        [
            self.m[0][0] * p.x + self.m[0][1] * p.y + self.m[0][2] * p.z + self.m[0][3],
            self.m[1][0] * p.x + self.m[1][1] * p.y + self.m[1][2] * p.z + self.m[1][3],
            self.m[2][0] * p.x + self.m[2][1] * p.y + self.m[2][2] * p.z + self.m[2][3],
            self.m[3][0] * p.x + self.m[3][1] * p.y + self.m[3][2] * p.z + self.m[3][3],
        ]
    }

    /// Transforms a point and performs the perspective divide.
    #[allow(dead_code)] // used by tests and handy for probes
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let c = self.transform_h(p);
        if c[3].abs() > f32::EPSILON {
            Vec3::new(c[0] / c[3], c[1] / c[3], c[2] / c[3])
        } else {
            Vec3::new(c[0], c[1], c[2])
        }
    }

    /// Transforms a direction (ignores translation).
    pub fn transform_dir(&self, p: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0] * p.x + self.m[0][1] * p.y + self.m[0][2] * p.z,
            self.m[1][0] * p.x + self.m[1][1] * p.y + self.m[1][2] * p.z,
            self.m[2][0] * p.x + self.m[2][1] * p.y + self.m[2][2] * p.z,
        )
    }

    /// Perspective projection: y-up, right-handed, looking down -Z.
    /// Maps near to `z_ndc = -1` and far to `+1`, with `w_clip = -z_view`.
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let nf = 1.0 / (near - far);
        let mut m = Mat4::ZERO;
        m.m[0][0] = f / aspect;
        m.m[1][1] = f;
        m.m[2][2] = (far + near) * nf;
        m.m[2][3] = 2.0 * far * near * nf;
        m.m[3][2] = -1.0;
        m
    }

    /// Look-at view matrix: eye looking at target, `up` defines roll.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let fwd = target.sub(eye).normalize();
        let right = fwd.cross(up).normalize();
        let true_up = right.cross(fwd);

        let mut m = Mat4::identity();
        m.m[0][0] = right.x;
        m.m[0][1] = right.y;
        m.m[0][2] = right.z;
        m.m[1][0] = true_up.x;
        m.m[1][1] = true_up.y;
        m.m[1][2] = true_up.z;
        m.m[2][0] = -fwd.x;
        m.m[2][1] = -fwd.y;
        m.m[2][2] = -fwd.z;
        m.m[0][3] = -right.dot(eye);
        m.m[1][3] = -true_up.dot(eye);
        m.m[2][3] = fwd.dot(eye);
        m
    }

    /// Translation matrix.
    pub fn translation(t: Vec3) -> Mat4 {
        let mut m = Mat4::identity();
        m.m[0][3] = t.x;
        m.m[1][3] = t.y;
        m.m[2][3] = t.z;
        m
    }

    /// Rotation about the Y axis, in radians.
    pub fn rotation_y(rad: f32) -> Mat4 {
        let (s, c) = rad.sin_cos();
        let mut m = Mat4::identity();
        m.m[0][0] = c;
        m.m[0][2] = s;
        m.m[2][0] = -s;
        m.m[2][2] = c;
        m
    }

    /// Rotation about the X axis, in radians.
    pub fn rotation_x(rad: f32) -> Mat4 {
        let (s, c) = rad.sin_cos();
        let mut m = Mat4::identity();
        m.m[1][1] = c;
        m.m[1][2] = -s;
        m.m[2][1] = s;
        m.m[2][2] = c;
        m
    }

    /// Rotation about the Z axis, in radians.
    pub fn rotation_z(rad: f32) -> Mat4 {
        let (s, c) = rad.sin_cos();
        let mut m = Mat4::identity();
        m.m[0][0] = c;
        m.m[0][1] = -s;
        m.m[1][0] = s;
        m.m[1][1] = c;
        m
    }

    /// Non-uniform scale matrix.
    pub fn scale(s: Vec3) -> Mat4 {
        let mut m = Mat4::identity();
        m.m[0][0] = s.x;
        m.m[1][1] = s.y;
        m.m[2][2] = s.z;
        m
    }
}

impl Index<usize> for Mat4 {
    type Output = [f32; 4];
    fn index(&self, i: usize) -> &[f32; 4] {
        &self.m[i]
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, o: Self) -> Mat4 {
        self.multiply(&o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a} (+-{tol})");
    }

    #[test]
    fn perspective_maps_near_to_minus_one_and_far_to_plus_one() {
        let p = Mat4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        approx(p.transform_point(Vec3::new(0.0, 0.0, -0.1)).z, -1.0, 1e-4);
        approx(p.transform_point(Vec3::new(0.0, 0.0, -100.0)).z, 1.0, 1e-3);
    }

    #[test]
    fn look_at_maps_target_to_minus_z_and_eye_to_origin() {
        let v = Mat4::look_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        );
        let eye = v.transform_point(Vec3::new(0.0, 0.0, 5.0));
        approx(eye.x, 0.0, 1e-4);
        approx(eye.y, 0.0, 1e-4);
        approx(eye.z, 0.0, 1e-4);
        approx(v.transform_point(Vec3::ZERO).z, -5.0, 1e-4);
        // +X stays right, +Y stays up.
        let rx = v.transform_point(Vec3::new(1.0, 0.0, 5.0));
        approx(rx.x, 1.0, 1e-4);
        approx(rx.y, 0.0, 1e-4);
    }

    #[test]
    fn transform_h_keeps_w_for_clipping() {
        let p = Mat4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        approx(p.transform_h(Vec3::new(0.0, 0.0, -5.0))[3], 5.0, 1e-4);
    }

    #[test]
    fn matrix_multiply_with_identity_is_noop() {
        let m = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.multiply(&Mat4::identity()), m);
    }

    #[test]
    fn rotations_are_orthonormal() {
        let m = Mat4::rotation_y(0.7).multiply(&Mat4::rotation_x(0.3));
        let x = m.transform_dir(Vec3::new(1.0, 0.0, 0.0));
        let y = m.transform_dir(Vec3::new(0.0, 1.0, 0.0));
        approx(x.length(), 1.0, 1e-5);
        approx(y.length(), 1.0, 1e-5);
        approx(x.dot(y), 0.0, 1e-5);
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        assert_eq!(
            Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 0.0, 1.0)
        );
    }
}
