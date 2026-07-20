//! Metal backend configuration.

use std::{ffi::CString, path::Path};

use crate::{
    error::{IoError, Result},
    utils::guard::Guarded,
};

/// Check whether the Metal backend is available.
pub fn is_available() -> Result<bool> {
    bool::try_from_op(|res| unsafe { mlx_sys::mlx_metal_is_available(res) })
}

/// Set the path to `mlx.metallib` before the first MLX operation.
///
/// MLX initializes its Metal device lazily. Calling this after an array or
/// stream has initialized the device does not replace the loaded library.
pub fn set_metallib_path(path: impl AsRef<Path>) -> std::result::Result<(), IoError> {
    let path = path.as_ref().to_str().ok_or(IoError::InvalidUtf8)?;
    let path = CString::new(path)?;
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_metal_set_metallib_path(path.as_ptr()) })
        .map_err(Into::into)
}
