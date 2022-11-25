extern crate ndarray;
extern crate ndarray_linalg;
extern crate num;
extern crate num_traits;
extern crate rustfft;

use ndarray::{prelude::*, IxDynImpl, OwnedRepr};
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, Slice};
use num::FromPrimitive;
use num_traits::Zero;
use rustfft::{num_complex::Complex, FftNum, FftPlanner};
use std::error::Error;

/// Pad the edges of an array with zeros.
///
/// `pad_width` specifies the length of the padding at the beginning
/// and end of each axis.
///
/// Returns a Result. Errors if arr.ndim() != pad_width.len().
fn pad_with_zeros<A, S, D>(
    arr: &ArrayBase<S, D>,
    pad_width: Vec<[usize; 2]>,
) -> Result<Array<A, D>, Box<dyn Error>>
where
    A: FftNum,
    S: Data<Elem = A>,
    D: Dimension,
{
    if arr.ndim() != pad_width.len() {
        return Err("arr.ndim() != pad_width.len()".into());
    }

    // Compute shape of final padded array.
    let mut padded_shape = arr.raw_dim();
    for (ax, (&ax_len, &[pad_lo, pad_hi])) in arr.shape().iter().zip(&pad_width).enumerate() {
        padded_shape[ax] = ax_len + pad_lo + pad_hi;
    }

    let mut padded = Array::zeros(padded_shape);
    let padded_dim = padded.raw_dim();
    {
        // Select portion of padded array that needs to be copied from the
        // original array.
        let mut orig_portion = padded.view_mut();
        for (ax, &[pad_lo, pad_hi]) in pad_width.iter().enumerate() {
            orig_portion.slice_axis_inplace(
                Axis(ax),
                Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)),
            );
        }
        // Copy the data from the original array.
        orig_portion.assign(arr);
    }
    Ok(padded)
}

/// Generates a Vec<[usize; 2]> specifying how much to pad each axis.
fn generate_pad_vector<A, S, D>(arr: &ArrayBase<S, D>, shape: &[usize]) -> Vec<[usize; 2]>
where
    A: FftNum,
    S: Data<Elem = A>,
    D: Dimension,
{
    arr.shape()
        .into_iter()
        .zip(shape.iter())
        .map(|(arr_size, new_size)| [0, new_size - arr_size])
        .collect()
}

/// Convolve two N-dimensional arrays using FFT.
pub fn fftconvolve<A, S, D>(
    in1: &ArrayBase<S, D>,
    in2: &ArrayBase<S, D>,
) -> Result<ArrayBase<OwnedRepr<A>, Dim<IxDynImpl>>, Box<dyn Error>>
where
    A: FftNum + FromPrimitive,
    S: Data<Elem = A>,
    D: Dimension,
{
    // check that arrays have the same number of dimensions
    if in1.ndim() != in2.ndim() {
        return Err("Input arrays must have the same number of dimensions.".into());
    }

    // Pad the arrays to the next power of 2.
    let mut shape = in1.shape().to_vec();
    for (s, s_other) in shape.iter_mut().zip(in2.shape().iter()) {
        *s = *s + *s_other - 1;
    }
    let in1 = pad_with_zeros(in1, generate_pad_vector(&in1, shape.as_slice()))?;
    let in2 = pad_with_zeros(in2, generate_pad_vector(&in2, shape.as_slice()))?;

    // multiple values in shape together to get total size
    let total_size = shape.iter().fold(1, |acc, x| acc * x);

    let mut in1 = in1.mapv(|x| Complex::new(x, Zero::zero()));
    let mut in2 = in2.mapv(|x| Complex::new(x, Zero::zero()));

    let mut scratch = vec![Complex::new(Zero::zero(), Zero::zero()); total_size];
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(total_size);
    fft.process_with_scratch(in1.as_slice_mut().unwrap(), scratch.as_mut_slice());
    fft.process_with_scratch(in2.as_slice_mut().unwrap(), scratch.as_mut_slice());

    // Multiply the FFTs.
    let mut out = in1 * in2;

    // Perform the inverse FFT.
    let fft = planner.plan_fft_inverse(total_size);
    fft.process_with_scratch(out.as_slice_mut().unwrap(), scratch.as_mut_slice());

    // Return the real part of the result. Note normalise by 1/total_size
    let total_size = A::from_usize(total_size).unwrap();
    // convert shape to a tuple of length shape.len()
    Ok(Array::from_shape_vec(
        shape,
        out.into_iter().map(|c| c.re / total_size).collect(),
    )?)
}

/// Cross-correlate two N-dimensional arrays using FFT.
/// Complex conjugate of second array is calculate in function.
pub fn fftcorrelate<A, S, D>(
    in1: &ArrayBase<S, D>,
    in2: &ArrayBase<S, D>,
) -> Result<ArrayBase<OwnedRepr<A>, Dim<IxDynImpl>>, Box<dyn Error>>
where
    A: FftNum + FromPrimitive,
    S: Data<Elem = A>,
    D: Dimension,
{
    // check that arrays have the same number of dimensions
    if in1.ndim() != in2.ndim() {
        return Err("Input arrays must have the same number of dimensions.".into());
    }
    // reverse the second array
    let shape = in2.shape().to_vec();
    let mut in2 = in2.to_owned().into_iter().collect::<Vec<_>>();
    in2.reverse();
    // collect into an array of original shape and get complex conjugate
    let in2 = Array::from_shape_vec(shape, in2)?;

    let mut shape = in1.shape().to_vec();
    for (s, s_other) in shape.iter_mut().zip(in2.shape().iter()) {
        *s = *s + *s_other - 1;
    }
    let in1 = pad_with_zeros(in1, generate_pad_vector(&in1, shape.as_slice()))?;
    let in2 = pad_with_zeros(&in2, generate_pad_vector(&in2, shape.as_slice()))?;

    // multiple values in shape together to get total size
    let total_size = shape.iter().fold(1, |acc, x| acc * x);

    let mut in1 = in1.mapv(|x| Complex::new(x, Zero::zero()));
    let mut in2 = in2.mapv(|x| Complex::new(x, Zero::zero()));

    let mut scratch = vec![Complex::new(Zero::zero(), Zero::zero()); total_size];
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(total_size);
    fft.process_with_scratch(in1.as_slice_mut().unwrap(), scratch.as_mut_slice());
    fft.process_with_scratch(in2.as_slice_mut().unwrap(), scratch.as_mut_slice());

    // Multiply the FFTs.
    let mut out = in1 * in2;

    // Perform the inverse FFT.
    let fft = planner.plan_fft_inverse(total_size);
    fft.process_with_scratch(out.as_slice_mut().unwrap(), scratch.as_mut_slice());

    // Return the real part of the result. Note normalise by 1/total_size
    let total_size = A::from_usize(total_size).unwrap();
    // convert shape to a tuple of length shape.len()
    Ok(Array::from_shape_vec(
        shape,
        out.into_iter().map(|c| c.re / total_size).collect(),
    )?)
}

// create tests
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_linalg::assert_aclose;

    #[test]
    fn reverse_array() {
        let standard = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut standard = standard.into_iter().collect::<Vec<_>>();
        standard.reverse();
        // reverse axes
        let reversed = Array2::from_shape_vec((3, 3), standard).unwrap();
        let expected = Array2::from_shape_vec((3, 3), vec![9, 8, 7, 6, 5, 4, 3, 2, 1]).unwrap();
        assert_eq!(reversed, expected);
    }

    #[test]
    fn test_pad() {
        let to_pad = Array2::<f32>::ones((3, 4));
        let padded = pad_with_zeros(&to_pad, vec![[2, 2], [3, 3]]).unwrap();
        let expected = Array2::<f32>::from_shape_vec(
            (7, 10),
            vec![
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
                0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0.,
            ],
        )
        .unwrap();
        // assert that the padded array is equal to the expected array
        assert_eq!(padded, expected);
    }

    #[test]
    fn test_fftconvolve_1d() {
        let in1 = Array1::range(1.0, 4.0, 1.0);
        let in2 = Array1::range(6.0, 3.0, -1.0);
        let out = fftconvolve(&in1, &in2).unwrap();
        let expected = Array1::<f64>::from_vec(vec![6., 17., 32., 23., 12.]);
        out.iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_aclose!(*a, *b, 1e-6));
    }

    #[test]
    fn test_fftconvolve_2d_1() {
        let mat = Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let kernel =
            Array2::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
        let output = fftconvolve(&mat, &kernel).unwrap();
        let expected = Array2::from_shape_vec(
            (5, 5),
            vec![
                0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0., 4., 5., 6., 0., 0., 7., 8., 9., 0., 0.,
                0., 0., 0., 0.,
            ],
        )
        .unwrap();
        output
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_aclose!(*a, *b, 1e-6));
    }

    #[test]
    fn test_fftconvolve_2d_2() {
        let mat = Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let kernel = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let output = fftconvolve(&mat, &kernel).unwrap();
        let expected = Array2::from_shape_vec(
            (4, 5),
            vec![
                0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0., 4., 5., 6., 0., 0., 0., 0., 0., 0.,
            ],
        )
        .unwrap();
        output
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_aclose!(*a, *b, 1e-6));
    }

    #[test]
    fn test_fftcorrelate_2d_1() {
        let mat = Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let kernel =
            Array2::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
        let output = fftcorrelate(&mat, &kernel).unwrap();
        let expected = Array2::from_shape_vec(
            (5, 5),
            vec![
                0., 0., 0., 0., 0., 0., 9., 8., 7., 0., 0., 6., 5., 4., 0., 0., 3., 2., 1., 0., 0.,
                0., 0., 0., 0.,
            ],
        )
        .unwrap();
        output
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_aclose!(*a, *b, 1e-6));
    }

    #[test]
    fn test_3d() {
        // 3-dimensional array of shape (2, 2, 2) filled with ones
        let mat = Array3::<f32>::ones((2, 2, 2));
        // 3-dimensional array of shape (2, 2, 2) values ranges from 0 to 7
        let kernel =
            Array3::from_shape_vec((2, 2, 2), vec![0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();

        let output_correlate = fftcorrelate(&mat, &kernel).unwrap();
        let output_convolve = fftconvolve(&mat, &kernel).unwrap();
        let expected_correlate = Array3::from_shape_vec(
            (3, 3, 3),
            vec![
                7., 13., 6., 12., 22., 10., 5., 9., 4., 10., 18., 8., 16., 28., 12., 6., 10., 4.,
                3., 5., 2., 4., 6., 2., 1., 1., 0.,
            ],
        )
        .unwrap();
        let expected_convolve = Array3::from_shape_vec(
            (3, 3, 3),
            vec![
                0., 1., 1., 2., 6., 4., 2., 5., 3., 4., 10., 6., 12., 28., 16., 8., 18., 10., 4.,
                9., 5., 10., 22., 12., 6., 13., 7.,
            ],
        )
        .unwrap();

        output_convolve
            .iter()
            .zip(expected_convolve.iter())
            .for_each(|(a, b)| assert_aclose!(*a, *b, 1e-5));
        output_correlate
            .iter()
            .zip(expected_correlate.iter())
            .for_each(|(a, b)| assert_aclose!(*a, *b, 1e-5));
    }
}
