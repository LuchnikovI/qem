use std::collections::HashSet;

use crate::ptrwrap::UnsafeSyncSendPtr;
use crate::utils::*;
use num_complex::Complex64;
use num_traits::{One, Zero};
use numpy::{PyArray1, PyArray2, PyArray4, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    prelude::{pyclass, pymethods, Bound, Py, Python},
    pyfunction,
    types::{PyAnyMethods, PyTuple},
    PyResult,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::{current_num_threads, scope};

// TODO: reduce code repetition

/// Quantum state class.
#[derive(Debug, Clone)]
#[pyclass]
pub struct QuantumState {
    state: Py<PyArray1<Complex64>>,
    qubits_number: u32,
}

/// Creates a quantum state from an array.
/// Args:
///     array: array representing a state.
/// Returns:
///     a quantum state.
/// Notes:
///     it does not check the normalization of the input vector. If it is not
///     normalized, results of computation could have no sense. It raises an error
///     in the following cases:
///         1) array is not C contiguous;
///         2) array has size that does not correspond to
///             some number of qubits (is not power of 2);
///         3) if array has rank not equal to 1;
///         4) a type of elements in array a different to `np.complex128`.
#[pyfunction]
#[pyo3(signature = (array,))]
pub fn array2state(array: Bound<PyArray1<Complex64>>) -> PyResult<QuantumState> {
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Array must be C contiguous".to_string(),
        ));
    }
    let size = array.shape()[0];
    let qubits_number = get_qubits_number(size);
    if size != 2usize.pow(qubits_number) {
        return Err(PyValueError::new_err(format!("Arrays of size {size} cannot be seen as a state of a qubits set, since its size is not a power of 2")));
    }
    Ok(QuantumState {
        state: array.into(),
        qubits_number,
    })
}

#[pymethods]
impl QuantumState {
    /// Creates a quantum state with a given number of qubits.
    /// All qubits are initialized in `0` state.
    /// Args:
    ///     qubits_number: a number of qubits.
    #[new]
    #[pyo3(signature = (qubits_number,))]
    fn new(py: Python<'_>, qubits_number: u32) -> Self {
        let size = 1 << qubits_number;
        let array = unsafe { PyArray1::<Complex64>::new_bound(py, [size], false) };
        let ptr = UnsafeSyncSendPtr(unsafe { array.uget_raw(0) });
        (1..size).into_par_iter().for_each(|idx| {
            unsafe { *ptr.add(idx) = Complex64::zero() };
        });
        unsafe { *ptr.add(0) = Complex64::one() };
        QuantumState {
            state: array.into(),
            qubits_number,
        }
    }
    /// Applies a one-qubit gate to a state inplace.
    /// Args:
    ///     pos: position of a qubits;
    ///     gate: a matrix representing a one-qubit quantum gate.
    /// Notes:
    ///     it raises an error in following cases:
    ///         1) is `pos` is out of bound;
    ///         2) if `gate` has shape different to (2, 2);
    ///         3) if elements of `gate` have type different to `np.complex128`.
    #[pyo3(signature = (pos, gate,))]
    fn apply1(
        &mut self,
        py: Python<'_>,
        pos: usize,
        gate: Bound<PyArray2<Complex64>>,
    ) -> PyResult<()> {
        if pos >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos` cannot be >= qubits number, get `pos` {pos} while number of qubits is {}",
                self.qubits_number
            )));
        }
        if gate.shape() != [2, 2] {
            return Err(PyValueError::new_err(format!(
                "Shape of a one-qubit gate must be (2, 2), got {:?}",
                gate.shape()
            )));
        }
        let pos = pos_fortran2c(self.qubits_number, pos);
        let batch_size = 1 << (self.qubits_number - 1);
        let left_mask = get_left_mask(pos);
        let right_mask = get_right_mask(pos);
        let stride = get_stride(pos);
        let state_ptr = unsafe { UnsafeSyncSendPtr(self.state.bind(py).uget_mut(0)) };
        let (&g00, &g01, &g10, &g11) = unsafe {
            (
                gate.uget([0, 0]),
                gate.uget([0, 1]),
                gate.uget([1, 0]),
                gate.uget([1, 1]),
            )
        };
        (0..batch_size).into_par_iter().for_each(|batch_index| {
            let batch_index = insert_zero_bit(batch_index, left_mask, right_mask);
            unsafe {
                let dst0 = state_ptr.add(batch_index);
                let dst1 = state_ptr.add(batch_index + stride);
                let src0 = *dst0;
                let src1 = *dst1;
                *dst0 = g00 * src0 + g01 * src1;
                *dst1 = g10 * src0 + g11 * src1;
            }
        });
        Ok(())
    }
    /// Applies a two-qubit gate to a state inplace.
    /// Args:
    ///     pos1: position of a first qubit;
    ///     pos2: position of a second qubit;
    ///     gate: a tensor representing a two-qubit quantum gate.
    /// Notes:
    ///     it raises an error in following cases:
    ///         1) is `pos1` of `pos2` is out of bound;
    ///         2) if `gate` has shape different to (2, 2, 2, 2);
    ///         3) if elements of `gate` have type different to `np.complex128`.
    #[pyo3(signature = (pos1, pos2, gate,))]
    fn apply2(
        &mut self,
        py: Python<'_>,
        pos1: usize,
        pos2: usize,
        gate: Bound<PyArray4<Complex64>>,
    ) -> PyResult<()> {
        if pos1 >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos1` cannot be >= qubits number, get `pos` {pos1} while number of qubits is {}",
                self.qubits_number
            )));
        }
        if pos2 >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos2` cannot be >= qubits number, get `pos` {pos2} while number of qubits is {}",
                self.qubits_number
            )));
        }
        if pos1 == pos2 {
            return Err(PyValueError::new_err(format!(
                "`pos1` must not be equal to `pos2`, got both equal to {pos1}"
            )));
        }
        if gate.shape() != [2, 2, 2, 2] {
            return Err(PyValueError::new_err(format!(
                "Shape of a two-qubit gate must be (2, 2, 2, 2), got {:?}",
                gate.shape()
            )));
        }
        let pos1 = pos_fortran2c(self.qubits_number, pos1);
        let pos2 = pos_fortran2c(self.qubits_number, pos2);
        let min_pos = std::cmp::min(pos1, pos2);
        let max_pos = std::cmp::max(pos1, pos2);
        let batch_size = 1 << (self.qubits_number - 2);
        let left_mask_min = get_left_mask(min_pos);
        let right_mask_min = get_right_mask(min_pos);
        let left_mask_max = get_left_mask(max_pos);
        let right_mask_max = get_right_mask(max_pos);
        let stride1 = get_stride(pos1);
        let stride0 = get_stride(pos2);
        let state_ptr = unsafe { UnsafeSyncSendPtr(self.state.bind(py).uget_mut(0)) };
        let (
            &g0000,
            &g0001,
            &g0010,
            &g0011,
            &g0100,
            &g0101,
            &g0110,
            &g0111,
            &g1000,
            &g1001,
            &g1010,
            &g1011,
            &g1100,
            &g1101,
            &g1110,
            &g1111,
        ) = unsafe {
            (
                gate.uget([0, 0, 0, 0]),
                gate.uget([0, 0, 0, 1]),
                gate.uget([0, 0, 1, 0]),
                gate.uget([0, 0, 1, 1]),
                gate.uget([0, 1, 0, 0]),
                gate.uget([0, 1, 0, 1]),
                gate.uget([0, 1, 1, 0]),
                gate.uget([0, 1, 1, 1]),
                gate.uget([1, 0, 0, 0]),
                gate.uget([1, 0, 0, 1]),
                gate.uget([1, 0, 1, 0]),
                gate.uget([1, 0, 1, 1]),
                gate.uget([1, 1, 0, 0]),
                gate.uget([1, 1, 0, 1]),
                gate.uget([1, 1, 1, 0]),
                gate.uget([1, 1, 1, 1]),
            )
        };
        (0..batch_size).into_par_iter().for_each(|batch_index| {
            let batch_index = insert_zero_bit(batch_index, left_mask_min, right_mask_min);
            let batch_index = insert_zero_bit(batch_index, left_mask_max, right_mask_max);
            unsafe {
                let dst00 = state_ptr.add(batch_index);
                let dst01 = state_ptr.add(batch_index + stride0);
                let dst10 = state_ptr.add(batch_index + stride1);
                let dst11 = state_ptr.add(batch_index + stride0 + stride1);
                let src00 = *dst00;
                let src01 = *dst01;
                let src10 = *dst10;
                let src11 = *dst11;
                *dst00 = g0000 * src00 + g0001 * src01 + g0010 * src10 + g0011 * src11;
                *dst01 = g0100 * src00 + g0101 * src01 + g0110 * src10 + g0111 * src11;
                *dst10 = g1000 * src00 + g1001 * src01 + g1010 * src10 + g1011 * src11;
                *dst11 = g1100 * src00 + g1101 * src01 + g1110 * src10 + g1111 * src11;
            }
        });
        Ok(())
    }
    /// Computes a density matrix of a given qubit.
    /// Args:
    ///     pos: a position of a qubit.
    /// Returns:
    ///     a matrix representing a density matrix of a qubit.
    /// Notes:
    ///     it raises an error if `pos` is out of bound.
    #[pyo3(signature = (pos,))]
    fn dens1<'py>(&self, py: Python<'py>, pos: usize) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        if pos >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos` cannot be >= qubits number, get `pos` {pos} while number of qubits is {}",
                self.qubits_number
            )));
        }
        let pos = pos_fortran2c(self.qubits_number, pos);
        let batch_size = 1 << (self.qubits_number - 1);
        let left_mask = get_left_mask(pos);
        let right_mask = get_right_mask(pos);
        let stride = get_stride(pos);
        let state_ptr = unsafe { UnsafeSyncSendPtr(self.state.bind(py).uget_mut(0)) };
        let threads_number = current_num_threads();
        let task_size = get_task_size(batch_size, threads_number);
        let mut density_matrices = vec![[Complex64::zero(); 4]; threads_number];
        scope(|s| {
            for (task_id, density_matrix) in (0..threads_number).zip(&mut density_matrices) {
                let task_start = task_id * task_size;
                let task_end = std::cmp::min(batch_size, (task_id + 1) * task_size);
                s.spawn(move |_| {
                    for batch_index in task_start..task_end {
                        let batch_index = insert_zero_bit(batch_index, left_mask, right_mask);
                        unsafe {
                            let src0 = *state_ptr.add(batch_index);
                            let src0_conj = src0.conj();
                            let src1 = *state_ptr.add(batch_index + stride);
                            let src1_conj = src1.conj();
                            density_matrix[0] += src0 * src0_conj;
                            density_matrix[1] += src0 * src1_conj;
                            density_matrix[2] += src1 * src0_conj;
                            density_matrix[3] += src1 * src1_conj;
                        }
                    }
                })
            }
        });
        let density_matrix = unsafe { PyArray2::<Complex64>::new_bound(py, [2, 2], false) };
        let density_matrix_ptr = unsafe { density_matrix.uget_raw([0, 0]) };
        for i in 0..4 {
            unsafe {
                *density_matrix_ptr.add(i) = Complex64::zero();
            }
        }
        for dens in &density_matrices {
            for (i, src) in dens.iter().enumerate() {
                unsafe {
                    *density_matrix_ptr.add(i) += src;
                }
            }
        }
        Ok(density_matrix)
    }
    /// Computes a density matrix of two-given qubits.
    /// Args:
    ///     pos1: position of a first qubit;
    ///     pos2: position of a second qubit.
    /// Returns:
    ///     array of rank 4 representing a density matrix.
    /// Notes:
    ///     it raises an error in the following cases:
    ///         1) if `pos1` or `pos2` is out of bound;
    ///         2) if `pos1` == `pos2`.
    #[pyo3(signature = (pos1, pos2,))]
    fn dens2<'py>(
        &self,
        py: Python<'py>,
        pos1: usize,
        pos2: usize,
    ) -> PyResult<Bound<'py, PyArray4<Complex64>>> {
        if pos1 >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos1` cannot be >= qubits number, get `pos` {pos1} while number of qubits is {}",
                self.qubits_number
            )));
        }
        if pos2 >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos2` cannot be >= qubits number, get `pos` {pos2} while number of qubits is {}",
                self.qubits_number
            )));
        }
        if pos1 == pos2 {
            return Err(PyValueError::new_err(format!(
                "`pos1` must not be equal to `pos2`, got both equal to {pos1}"
            )));
        }
        let pos1 = pos_fortran2c(self.qubits_number, pos1);
        let pos2 = pos_fortran2c(self.qubits_number, pos2);
        let min_pos = std::cmp::min(pos1, pos2);
        let max_pos = std::cmp::max(pos1, pos2);
        let batch_size = 1 << (self.qubits_number - 2);
        let left_mask_min = get_left_mask(min_pos);
        let right_mask_min = get_right_mask(min_pos);
        let left_mask_max = get_left_mask(max_pos);
        let right_mask_max = get_right_mask(max_pos);
        let stride1 = get_stride(pos1);
        let stride0 = get_stride(pos2);
        let state_ptr = unsafe { UnsafeSyncSendPtr(self.state.bind(py).uget_mut(0)) };
        let threads_number = current_num_threads();
        let task_size = get_task_size(batch_size, threads_number);
        let mut density_matrices = vec![[Complex64::zero(); 16]; threads_number];
        scope(|s| {
            for (task_id, density_matrix) in (0..threads_number).zip(&mut density_matrices) {
                let task_start = task_id * task_size;
                let task_end = std::cmp::min(batch_size, (task_id + 1) * task_size);
                s.spawn(move |_| {
                    for batch_index in task_start..task_end {
                        let batch_index =
                            insert_zero_bit(batch_index, left_mask_min, right_mask_min);
                        let batch_index =
                            insert_zero_bit(batch_index, left_mask_max, right_mask_max);
                        unsafe {
                            let src0 = *state_ptr.add(batch_index);
                            let src0_conj = src0.conj();
                            let src1 = *state_ptr.add(batch_index + stride0);
                            let src1_conj = src1.conj();
                            let src2 = *state_ptr.add(batch_index + stride1);
                            let src2_conj = src2.conj();
                            let src3 = *state_ptr.add(batch_index + stride0 + stride1);
                            let src3_conj = src3.conj();
                            density_matrix[0] += src0 * src0_conj;
                            density_matrix[1] += src0 * src1_conj;
                            density_matrix[2] += src0 * src2_conj;
                            density_matrix[3] += src0 * src3_conj;
                            density_matrix[4] += src1 * src0_conj;
                            density_matrix[5] += src1 * src1_conj;
                            density_matrix[6] += src1 * src2_conj;
                            density_matrix[7] += src1 * src3_conj;
                            density_matrix[8] += src2 * src0_conj;
                            density_matrix[9] += src2 * src1_conj;
                            density_matrix[10] += src2 * src2_conj;
                            density_matrix[11] += src2 * src3_conj;
                            density_matrix[12] += src3 * src0_conj;
                            density_matrix[13] += src3 * src1_conj;
                            density_matrix[14] += src3 * src2_conj;
                            density_matrix[15] += src3 * src3_conj;
                        }
                    }
                })
            }
        });
        let density_matrix = unsafe { PyArray4::<Complex64>::new_bound(py, [2, 2, 2, 2], false) };
        let density_matrix_ptr = unsafe { density_matrix.uget_raw([0, 0, 0, 0]) };
        for i in 0..16 {
            unsafe {
                *density_matrix_ptr.add(i) = Complex64::zero();
            }
        }
        for dens in &density_matrices {
            for (i, src) in dens.iter().enumerate() {
                unsafe {
                    *density_matrix_ptr.add(i) += src;
                }
            }
        }
        Ok(density_matrix)
    }
    /// Performs a measurement of a given qubit.
    /// Args:
    ///     pos: a position of a qubit;
    ///     uniform_sample: a random sample from `uniform(0, 1)`
    ///         determining the measurement outcome;
    /// Returns:
    ///     measurement result (either `0` or `1`).
    /// Notes:
    ///     it raises an error in the following cases:
    ///         1) if `pos` is out of bound;
    ///         2) if `uniform_sample` has 0 or > 1 elements;
    ///         3) if type of element in `uniform_sample` is not `np.float64`.
    #[pyo3(signature = (pos, uniform_sample,))]
    fn measure(
        &mut self,
        py: Python<'_>,
        pos: usize,
        uniform_sample: Bound<PyArray1<f64>>,
    ) -> PyResult<u8> {
        if pos >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos` cannot be >= qubits number, get `pos` {pos} while number of qubits is {}",
                self.qubits_number
            )));
        }
        let pos = pos_fortran2c(self.qubits_number, pos);
        let batch_size = 1 << (self.qubits_number - 1);
        let left_mask = get_left_mask(pos);
        let right_mask = get_right_mask(pos);
        let state_ptr = unsafe { UnsafeSyncSendPtr(self.state.bind(py).uget_mut(0)) };
        let threads_number = current_num_threads();
        let task_size = get_task_size(batch_size, threads_number);
        let mut probabilities = vec![0f64; threads_number];
        scope(|s| {
            for (task_id, probability) in (0..threads_number).zip(&mut probabilities) {
                let task_start = task_id * task_size;
                let task_end = std::cmp::min(batch_size, (task_id + 1) * task_size);
                s.spawn(move |_| {
                    for batch_index in task_start..task_end {
                        let batch_index = insert_zero_bit(batch_index, left_mask, right_mask);
                        unsafe {
                            let src0 = *state_ptr.add(batch_index);
                            *probability += (src0 * src0.conj()).re;
                        }
                    }
                })
            }
        });
        let probability = probabilities.iter().sum::<f64>();
        let (result, norm, zeroed_stride, renorm_stride) =
            if unsafe { *uniform_sample.uget(0) } < probability {
                (0, probability.sqrt(), 1 << pos, 0)
            } else {
                (1, (1f64 - probability).sqrt(), 0, 1 << pos)
            };
        (0..batch_size).into_par_iter().for_each(|batch_index| {
            let batch_index = insert_zero_bit(batch_index, left_mask, right_mask);
            unsafe {
                *state_ptr.add(batch_index + renorm_stride) /= norm;
                *state_ptr.add(batch_index + zeroed_stride) = Complex64::zero();
            }
        });
        Ok(result)
    }
    /// Resets a given qubit to `0` state.
    /// Args:
    ///     pos: a position of a qubit;
    ///     uniform_sample: a random sample from `uniform(0, 1)`
    ///         determining the measurement outcome;
    /// Notes:
    ///     it raise an error in the following cases:
    ///         1) if `pos` is out of bound;
    ///         2) if `uniform_sample` has 0 or > 1 elements;
    ///         3) if type of element in `uniform_sample` is not `np.float64`.
    #[pyo3(signature = (pos, uniform_sample,))]
    fn reset(
        &mut self,
        py: Python<'_>,
        pos: usize,
        uniform_sample: Bound<PyArray1<f64>>,
    ) -> PyResult<()> {
        if pos >= self.qubits_number as usize {
            return Err(PyValueError::new_err(format!(
                "`pos` cannot be >= qubits number, get `pos` {pos} while number of qubits is {}",
                self.qubits_number
            )));
        }
        let pos = pos_fortran2c(self.qubits_number, pos);
        let batch_size = 1 << (self.qubits_number - 1);
        let left_mask = get_left_mask(pos);
        let right_mask = get_right_mask(pos);
        let state_ptr = unsafe { UnsafeSyncSendPtr(self.state.bind(py).uget_mut(0)) };
        let threads_number = current_num_threads();
        let task_size = get_task_size(batch_size, threads_number);
        let mut probabilities = vec![0f64; threads_number];
        scope(|s| {
            for (task_id, probability) in (0..threads_number).zip(&mut probabilities) {
                let task_start = task_id * task_size;
                let task_end = std::cmp::min(batch_size, (task_id + 1) * task_size);
                s.spawn(move |_| {
                    for batch_index in task_start..task_end {
                        let batch_index = insert_zero_bit(batch_index, left_mask, right_mask);
                        unsafe {
                            let src0 = *state_ptr.add(batch_index);
                            *probability += (src0 * src0.conj()).re;
                        }
                    }
                })
            }
        });
        let probability = probabilities.iter().sum::<f64>();
        let (result, norm, zeroed_stride, renorm_stride) =
            if unsafe { *uniform_sample.uget(0) } < probability {
                (0, probability.sqrt(), 1 << pos, 0)
            } else {
                (1, (1f64 - probability).sqrt(), 0, 1 << pos)
            };
        (0..batch_size).into_par_iter().for_each(|batch_index| {
            let batch_index = insert_zero_bit(batch_index, left_mask, right_mask);
            unsafe {
                if result == 0 {
                    *state_ptr.add(batch_index + renorm_stride) /= norm;
                    *state_ptr.add(batch_index + zeroed_stride) = Complex64::zero();
                } else {
                    let src = state_ptr.add(batch_index + renorm_stride);
                    *state_ptr.add(batch_index + zeroed_stride) = *src / norm;
                    *src = Complex64::zero();
                }
            }
        });
        Ok(())
    }
    /// Resets all qubits to `0` state.
    #[pyo3(signature = ())]
    fn total_reset(&mut self, py: Python<'_>) {
        let size = 1 << self.qubits_number;
        let ptr = UnsafeSyncSendPtr(unsafe { self.state.bind(py).uget_raw(0) });
        (1..size).into_par_iter().for_each(|idx| {
            unsafe { *ptr.add(idx) = Complex64::zero() };
        });
        unsafe { *ptr.add(0) = Complex64::one() };
    }
    /// Returns a density matrix of an arbitrary subset of qubits.
    /// Args:
    ///     qubit_positions: positions of qubits.
    /// Returns:
    ///     array of rank 2 * len(qubits_position) representing a density matrix.
    /// Notes:
    ///     this routine is for computing of large partial density matrices,
    ///     do not use it to compute one- or two-qubits partial density matrix,
    ///     since it would be slow;
    ///     it raises an error in the following cases:
    ///         1) if one or more positions are out of bound;
    ///         2) if some positions are equal to each other.
    #[pyo3(signature = (qubit_positions,))]
    fn dens_large<'py>(
        &self,
        py: Python<'py>,
        qubit_positions: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let qubit_positions = qubit_positions
            .into_iter()
            .rev()
            .map(|pos| {
                pos.extract::<usize>()
                    .map(|pos| pos_fortran2c(self.qubits_number, pos))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let positions_number = qubit_positions.len();
        let mut seen_positions = HashSet::with_capacity(positions_number);
        for pos in &qubit_positions {
            if !seen_positions.insert(*pos) {
                return Err(PyValueError::new_err(format!(
                    "Position {pos} appears more than once"
                )));
            }
            if *pos as u32 >= self.qubits_number {
                return Err(PyValueError::new_err(format!(
                    "Position {pos} is out of bound of state consisting of {} qubits",
                    self.qubits_number
                )));
            }
        }
        let mut sorted_qubits_positions = qubit_positions.clone();
        sorted_qubits_positions.sort();
        let left_masks = get_left_masks(&sorted_qubits_positions);
        let right_masks = get_right_masks(&sorted_qubits_positions);
        let strides = get_strides(&qubit_positions);
        let batch_size = 1 << (self.qubits_number - positions_number as u32);
        let dens_size = 1 << positions_number;
        let state_ptr = unsafe { UnsafeSyncSendPtr(self.state.bind(py).uget_mut(0)) };
        let density_matrix =
            unsafe { PyArrayDyn::<Complex64>::new_bound(py, vec![2; 2 * positions_number], false) };
        let density_matrix_ptr =
            unsafe { UnsafeSyncSendPtr(density_matrix.uget_raw(vec![0; 2 * positions_number])) };
        (0..dens_size).into_par_iter().for_each(|row_number| {
            (0..dens_size).into_par_iter().for_each(|col_number| {
                let density_matrix_ptr =
                    unsafe { density_matrix_ptr.add(col_number + row_number * dens_size) };
                unsafe { *density_matrix_ptr = Complex64::zero() };
                let row_number = dens_idx2state_idx(row_number, &strides);
                let col_number = dens_idx2state_idx(col_number, &strides);
                for bidx in 0..batch_size {
                    let bidx = insert_zero_bits(bidx, &left_masks, &right_masks);
                    unsafe {
                        let rhs = *state_ptr.add(bidx + col_number);
                        let lhs = *state_ptr.add(bidx + row_number);
                        *density_matrix_ptr += lhs * rhs.conj();
                    }
                }
            })
        });
        Ok(density_matrix)
    }
    #[getter]
    fn state_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.state.bind(py).clone()
    }
    #[getter]
    fn qubits_number(&self) -> u32 {
        self.qubits_number
    }
    fn __repr__(&self) -> String {
        format!(
            "Qubits number: {},\nState array: {}",
            self.qubits_number, self.state
        )
    }
}
