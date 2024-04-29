use pyo3::prelude::PyModule;
use pyo3::{pymodule, wrap_pyfunction, Bound, PyResult};

mod api;

#[pymodule]
fn qem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<api::QuantumState>()?;
    m.add_function(wrap_pyfunction!(api::array2state, m)?)?;
    Ok(())
}
