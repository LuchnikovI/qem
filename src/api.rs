use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArray4};
use pyo3::{
    prelude::{pyclass, pymethods, Bound},
    pyfunction, PyResult,
};

/// Quantum state class.
#[derive(Debug, Clone)]
#[pyclass]
pub struct QuantumState;

/// Creates a quantum state from an array.
/// Args:
///     array: array representing a state.
/// Returns:
///     a quantum state.
/// Notes:
///     it does not check the normalization of the input vector. If it is not
///     normalized, results of computation could have no sense. It raises an error
///     in the following cases:
///         1) array has size that does not correspond to
///             some number of qubits (is not power of 2);
///         2) if array has rank not equal to 1;
///         3) a type of elements in array a different to `np.complex128`.
#[pyfunction]
#[pyo3(signature = (array,))]
pub fn array2state(array: Bound<PyArray1<Complex64>>) -> PyResult<QuantumState> {
    todo!()
}

#[pymethods]
impl QuantumState {
    /// Creates a quantum state with a given number of qubits.
    /// All qubits are initialized in `0` state.
    /// Args:
    ///     qubits_number: a number of qubits.
    #[new]
    #[pyo3(signature = (qubits_number,))]
    fn new(qubits_number: usize) -> Self {
        todo!()
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
    fn apply1(&mut self, pos: usize, gate: Bound<PyArray2<Complex64>>) -> PyResult<()> {
        todo!()
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
        pos1: usize,
        pos2: usize,
        gate: Bound<PyArray4<Complex64>>,
    ) -> PyResult<()> {
        todo!()
    }
    /// Computes a density matrix of a given qubit.
    /// Args:
    ///     pos: a position of a qubit.
    /// Returns:
    ///     a matrix representing a density matrix of a qubit.
    /// Notes:
    ///     it raises an error if `pos` is out of bound.
    #[pyo3(signature = (pos,))]
    fn dens1(&self, pos: usize) -> PyResult<PyArray2<Complex64>> {
        todo!()
    }
    /// Computes a density matrix of two-given qubits.
    /// Args:
    ///     pos1: position of a first qubit;
    ///     pos2: position of a second qubit.
    /// Returns:
    ///     it raises an error in the following cases:
    ///         1) if `pos1` or `pos2` is out of bound;
    ///         2) if `pos1` == `pos2`.
    #[pyo3(signature = (pos1, pos2,))]
    fn dens2(&self, pos1: usize, pos2: usize) -> PyResult<PyArray4<Complex64>> {
        todo!()
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
    fn measure(&mut self, pos: usize, uniform_sample: Bound<PyArray1<f64>>) -> PyResult<u8> {
        todo!()
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
    fn reset(&mut self, pos: usize, uniform_sample: Bound<PyArray1<f64>>) -> PyResult<()> {
        todo!()
    }
    /// Resets all qubits to `0` state.
    #[pyo3(signature = ())]
    fn total_reset(&mut self) {
        todo!()
    }
    #[getter]
    fn state_array(&self) -> Bound<PyArray1<Complex64>> {
        todo!()
    }
    #[getter]
    fn qubits_number(&self) -> usize {
        todo!()
    }
    fn __str__(&self) -> String {
        todo!()
    }
    fn __repr__(&self) -> String {
        todo!()
    }
}
