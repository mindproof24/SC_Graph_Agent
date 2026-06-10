// src/sparse_utils.rs
// ============================================================
// Sparse matrix utilities (SnapATAC2 / Kai Zhang pattern)
// Python scipy.sparse.csr_matrix → Rust CsrMatrix<f64>
// ============================================================

use pyo3::prelude::*;
use numpy::PyReadonlyArrayDyn;
use nalgebra_sparse::csr::CsrMatrix;

/// Python scipy.sparse.csr_matrix → nalgebra CsrMatrix<f64>
///
/// Extracts shape, indices, indptr, data attributes from the Python object
/// and constructs a Rust CsrMatrix without any dense conversion.
pub fn csr_to_rust(csr: &Bound<'_, PyAny>) -> PyResult<CsrMatrix<f64>> {
    let shape: Vec<usize> = csr.getattr("shape")?.extract()?;
    let indptr: Vec<usize> = cast_to_usize(&csr.getattr("indptr")?)?;
    let indices: Vec<usize> = cast_to_usize(&csr.getattr("indices")?)?;
    let data: Vec<f64> = cast_to_f64(&csr.getattr("data")?)?;

    CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to create CsrMatrix: {:?}", e)
        ))
}

/// numpy array → Vec<usize> (for indptr, indices)
fn cast_to_usize(arr: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    match dtype_name.as_str() {
        "int32" => {
            let a = arr.extract::<PyReadonlyArrayDyn<i32>>()?;
            Ok(a.as_slice().unwrap().iter().map(|&v| v as usize).collect())
        }
        "int64" => {
            let a = arr.extract::<PyReadonlyArrayDyn<i64>>()?;
            Ok(a.as_slice().unwrap().iter().map(|&v| v as usize).collect())
        }
        "uint32" => {
            let a = arr.extract::<PyReadonlyArrayDyn<u32>>()?;
            Ok(a.as_slice().unwrap().iter().map(|&v| v as usize).collect())
        }
        "uint64" => {
            let a = arr.extract::<PyReadonlyArrayDyn<u64>>()?;
            Ok(a.as_slice().unwrap().iter().map(|&v| v as usize).collect())
        }
        ty => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            format!("Expected integer dtype for indices/indptr, got: {}", ty)
        )),
    }
}

/// numpy array → Vec<f64> (for data values)
fn cast_to_f64(arr: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    match dtype_name.as_str() {
        "float64" => {
            let a = arr.extract::<PyReadonlyArrayDyn<f64>>()?;
            Ok(a.as_slice().unwrap().to_vec())
        }
        "float32" => {
            let a = arr.extract::<PyReadonlyArrayDyn<f32>>()?;
            Ok(a.as_slice().unwrap().iter().map(|&v| v as f64).collect())
        }
        "int32" => {
            let a = arr.extract::<PyReadonlyArrayDyn<i32>>()?;
            Ok(a.as_slice().unwrap().iter().map(|&v| v as f64).collect())
        }
        "int64" => {
            let a = arr.extract::<PyReadonlyArrayDyn<i64>>()?;
            Ok(a.as_slice().unwrap().iter().map(|&v| v as f64).collect())
        }
        ty => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            format!("Expected numeric dtype for data, got: {}", ty)
        )),
    }
}
