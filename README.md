# fftconvolve
Rust implementations of Fast Fourier Transform convolution and correlation for n-dimensional arrays

# Examples

## 1-dimensional
```rust
use fftconvolve::{fftconvolve, Mode}

let in1 = Array1::range(1.0, 4.0, 1.0);
let in2 = Array1::range(6.0, 3.0, -1.0);
let out = fftconvolve(&in1, &in2, Mode::Full).unwrap();
let expected = Array1::<f64>::from_vec(vec![6., 17., 32., 23., 12.]);
out.iter().zip(expected.iter()).for_each(|(a, b)| assert_aclose!(*a, *b, 1e-6));
```

## 2-dimensional
```rust
use fftconvolve::{fftconvolve, Mode}

let mat = Array2::from_shape_vec((3, 3), 
    vec![
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]).unwrap();
let kernel = Array2::from_shape_vec((2, 3), 
    vec![
        1., 2., 3., 
        4., 5., 6.,
    ]).unwrap();
let output = fftconvolve(&mat, &kernel, Mode::Full).unwrap();
let expected = Array2::from_shape_vec((4, 5), 
    vec![
        0., 0., 0., 0., 0., 
        0., 1., 2., 3., 0., 
        0., 4., 5., 6., 0.,
        0., 0., 0., 0., 0.,  
    ]).unwrap();
output.iter().zip(expected.iter()).for_each(|(a, b)| assert_aclose!(*a, *b, 1e-6));
```
