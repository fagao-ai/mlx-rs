//! Fast implementations of commonly used multi-op functions.

use std::ffi::CStr;

use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{Array, Stream};
use mlx_internal_macros::{default_device, generate_macro};

/// Compute `silu(gate) * x` with a process-wide shapeless compiled function.
///
/// Unlike [`crate::transforms::compile::compile`], this helper preserves the
/// compiled closure across calls, matching Python's module-level `@mx.compile`
/// behavior for decode-heavy inference.
pub fn compiled_swiglu(gate: impl AsRef<Array>, x: impl AsRef<Array>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_compiled_swiglu(res, gate.as_ref().as_ptr(), x.as_ref().as_ptr())
    })
}

/// Compute `argmax(hidden @ embedding.T)` without materializing full logits.
///
/// The GPU path follows MLX's GEMV accumulation and output rounding order for
/// matching FP32, FP16, or BF16 inputs. CPU streams use the equivalent MLX
/// graph without materializing logits outside the helper.
///
/// `hidden` must contain exactly one vector. `embedding` must be rank 2 with
/// at least 8192 rows and a row count divisible by 32.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn lm_head_argmax_device(
    hidden: impl AsRef<Array>,
    embedding: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_lm_head_argmax(
            res,
            hidden.as_ref().as_ptr(),
            embedding.as_ref().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compute a single-vector row-major matrix product with a fixed SmolLM tile.
///
/// `weight` is `[N, K]`, while `input` must contain one K-element vector.
/// `residual`, when present, must contain N elements. The Metal kernel accepts
/// one or four output rows per SIMD group so the two launch layouts can be
/// benchmarked without rebuilding MLX.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn smollm_gemv_device<'a>(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    results_per_simdgroup: i32,
    #[optional] residual: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let residual = residual.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_smollm_gemv(
            res,
            input.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            residual
                .map(|array| array.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            results_per_simdgroup,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Fuse SmolLM's one-token YaRN scale and RoPE for Q/K and pack K/V.
///
/// Returns `(rotated_queries, packed_kv)`, where packed KV has shape
/// `[2, B, kv_heads, 1, 64]`. This specialization requires FP32 frequencies.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn smollm_yarn_rope_qkv_device(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    freqs: impl AsRef<Array>,
    attention_factor: f32,
    offset: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    <(Array, Array)>::try_from_op(|(queries_res, packed_kv_res)| unsafe {
        mlx_sys::mlx_fast_smollm_yarn_rope_qkv(
            queries_res,
            packed_kv_res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            freqs.as_ref().as_ptr(),
            attention_factor,
            offset,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Fuse SmolLM's one-token YaRN scale and RoPE for Q/K.
///
/// Returns `(rotated_queries, rotated_keys)`. This specialization requires
/// FP32 frequencies and preserves MLX's low-precision scaling order.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn smollm_yarn_rope_qk_device(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    freqs: impl AsRef<Array>,
    attention_factor: f32,
    offset: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    <(Array, Array)>::try_from_op(|(queries_res, keys_res)| unsafe {
        mlx_sys::mlx_fast_smollm_yarn_rope_qk(
            queries_res,
            keys_res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            freqs.as_ref().as_ptr(),
            attention_factor,
            offset,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Apply PaddleOCR-VL's two-dimensional RoPE to Q and K in a packed QKV
/// projection. The result is ordered as `[Q, K]` with shape `[2, 16, L, 72]`.
///
/// This is a Metal-only fused custom kernel. It accepts matching FP32, FP16,
/// or BF16 `qkv=[L,3456]` and `cosine/sine=[L,72]` or `[L,1,72]`, preserving
/// the model's original trigonometric inputs while avoiding the composed
/// slice/negate/concatenate/elementwise graph.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn paddleocr_rope_2d_qk_device(
    qkv: impl AsRef<Array>,
    cosine: impl AsRef<Array>,
    sine: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_paddleocr_rope_2d_qk(
            res,
            qkv.as_ref().as_ptr(),
            cosine.as_ref().as_ptr(),
            sine.as_ref().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional with the same size as the last axis of `x`.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rms_norm_device(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops,
        ops::indexing::{argmax_axis, argmax_axis_device, ArrayIndexOp, IndexOp},
        random::normal,
        Dtype, Stream,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_compiled_swiglu_matches_eager_graph_across_shapes() {
        let gate = Array::from_slice(&[-2.0_f32, -0.5, 0.0, 1.0, 3.0, 4.0], &[2, 3]);
        let x = Array::from_slice(&[0.5_f32, -1.0, 2.0, -3.0, 1.5, 0.25], &[2, 3]);
        let expected = &(&gate * &crate::ops::sigmoid(&gate).unwrap()) * &x;
        let result = compiled_swiglu(&gate, &x).unwrap();
        crate::assert_array_eq!(result, expected, 1e-6);

        // The compiled closure is shapeless, so a later invocation can reuse
        // it with a different sequence length.
        let gate = Array::from_slice(&[-1.0_f32, 0.5, 2.0, 5.0], &[1, 4]);
        let x = Array::from_slice(&[3.0_f32, -2.0, 0.25, 1.5], &[1, 4]);
        let expected = &(&gate * &crate::ops::sigmoid(&gate).unwrap()) * &x;
        let result = compiled_swiglu(&gate, &x).unwrap();
        crate::assert_array_eq!(result, expected, 1e-6);
    }

    fn lm_head_test_inputs() -> (Array, Array, i32) {
        const HIDDEN_SIZE: i32 = 128;
        const VOCAB_SIZE: i32 = 49_152;
        const EXPECTED_TOKEN: i32 = 31_337;

        let mut hidden_values = vec![0.0_f32; HIDDEN_SIZE as usize];
        hidden_values[0] = 1.0;
        hidden_values[1] = -0.5;
        let mut embedding_values = vec![0.0_f32; (VOCAB_SIZE * HIDDEN_SIZE) as usize];
        let row = EXPECTED_TOKEN as usize * HIDDEN_SIZE as usize;
        embedding_values[row] = 2.0;
        embedding_values[row + 1] = -1.0;

        (
            Array::from_slice(&hidden_values, &[1, 1, HIDDEN_SIZE]),
            Array::from_slice(&embedding_values, &[VOCAB_SIZE, HIDDEN_SIZE]),
            EXPECTED_TOKEN,
        )
    }

    #[test]
    fn test_lm_head_argmax_matches_bf16_gpu_graph() {
        let (hidden, embedding, expected_token) = lm_head_test_inputs();
        let hidden = hidden.as_dtype(Dtype::Bfloat16).unwrap();
        let embedding = embedding.as_dtype(Dtype::Bfloat16).unwrap();
        let logits = crate::ops::matmul(&hidden, embedding.t()).unwrap();
        let expected = argmax_axis(&logits, -1, None)
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap();
        let actual = lm_head_argmax(&hidden, &embedding).unwrap();

        assert_eq!(actual.shape(), [1]);
        assert_eq!(actual.dtype(), Dtype::Int32);
        assert_eq!(actual.item::<i32>(), expected_token);
        assert_eq!(actual.item::<i32>(), expected.item::<i32>());
    }

    #[test]
    fn test_lm_head_argmax_cpu_fallback_matches_graph() {
        let (hidden, embedding, expected_token) = lm_head_test_inputs();
        let stream = Stream::cpu();
        let embedding_t = embedding.transpose_device(&stream).unwrap();
        let logits = hidden.matmul_device(&embedding_t, &stream).unwrap();
        let expected = argmax_axis_device(&logits, -1, None, &stream)
            .unwrap()
            .as_dtype_device(Dtype::Int32, &stream)
            .unwrap();
        let actual = lm_head_argmax_device(&hidden, &embedding, &stream).unwrap();

        assert_eq!(actual.shape(), [1]);
        assert_eq!(actual.dtype(), Dtype::Int32);
        assert_eq!(actual.item::<i32>(), expected_token);
        assert_eq!(actual.item::<i32>(), expected.item::<i32>());
    }

    fn deterministic_array(shape: &[i32], dtype: Dtype) -> Array {
        let size = shape.iter().map(|&dimension| dimension as usize).product();
        let values = (0..size)
            .map(|index| {
                let value = ((index * 37 + 11) % 251) as f32 - 125.0;
                value / 128.0
            })
            .collect::<Vec<_>>();
        Array::from_slice(&values, shape).as_dtype(dtype).unwrap()
    }

    fn assert_arrays_exact(actual: &Array, expected: &Array) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), expected.dtype());
        assert!(
            actual.array_eq(expected, None).unwrap().item::<bool>(),
            "arrays differ for shape {:?} and dtype {:?}",
            actual.shape(),
            actual.dtype()
        );
    }

    #[test]
    fn test_smollm_gemv_matches_mlx_for_model_projection_shapes() {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for (input_size, output_size) in [(576, 1536), (960, 2560)] {
                let input = deterministic_array(&[1, 1, input_size], dtype);
                let weight = deterministic_array(&[output_size, input_size], dtype);
                let residual = deterministic_array(&[1, 1, output_size], dtype);
                let expected = ops::matmul(&input, weight.t()).unwrap();
                let expected_add = ops::addmm(&residual, &input, weight.t(), None, None).unwrap();

                for results_per_simdgroup in [1, 4] {
                    let actual = smollm_gemv(&input, &weight, results_per_simdgroup, None).unwrap();
                    assert_arrays_exact(&actual, &expected);

                    let actual_add =
                        smollm_gemv(&input, &weight, results_per_simdgroup, Some(&residual))
                            .unwrap();
                    assert_arrays_exact(&actual_add, &expected_add);
                }
            }
        }
    }

    #[test]
    fn test_smollm_yarn_rope_qkv_matches_standard_graph() {
        let frequencies = Array::from_slice(
            &(0..32)
                .map(|index| 100_000.0_f32.powf(index as f32 / 32.0))
                .collect::<Vec<_>>(),
            &[32],
        );
        let attention_factor = 1.069_314_7_f32;
        let offset = 8191;

        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let queries = deterministic_array(&[1, 9, 1, 64], dtype);
            let keys = deterministic_array(&[1, 3, 1, 64], dtype);
            let values = deterministic_array(&[1, 3, 1, 64], dtype);
            let factor = Array::from_f32(attention_factor).as_dtype(dtype).unwrap();
            let scaled_queries = &queries * &factor;
            let scaled_keys = &keys * &factor;
            let expected_queries = rope(
                &scaled_queries,
                64,
                false,
                None,
                1.0,
                offset,
                Some(&frequencies),
            )
            .unwrap();
            let expected_keys = rope(
                &scaled_keys,
                64,
                false,
                None,
                1.0,
                offset,
                Some(&frequencies),
            )
            .unwrap();
            let expected_packed = ops::stack_axis(&[&expected_keys, &values], 0).unwrap();

            let (actual_queries, actual_packed) = smollm_yarn_rope_qkv(
                &queries,
                &keys,
                &values,
                &frequencies,
                attention_factor,
                offset,
            )
            .unwrap();
            assert_arrays_exact(&actual_queries, &expected_queries);
            assert_arrays_exact(&actual_packed, &expected_packed);

            let (actual_queries, actual_keys) =
                smollm_yarn_rope_qk(&queries, &keys, &frequencies, attention_factor, offset)
                    .unwrap();
            assert_arrays_exact(&actual_queries, &expected_queries);
            assert_arrays_exact(&actual_keys, &expected_keys);
        }
    }

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_paddleocr_rope_2d_qk_matches_composed_graph() {
        use crate::ops::{concatenate_axis, indexing::TryIndexOp};

        fn assert_matches(label: &str, fused: &Array, expected: &Array, token_count: usize) {
            fused.eval().unwrap();
            expected.eval().unwrap();
            let fused_values = fused.as_slice::<f32>();
            let expected_values = expected.as_slice::<f32>();
            let (flat_index, max_difference) = fused_values
                .iter()
                .zip(expected_values)
                .enumerate()
                .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap();
            let per_kind = 16 * token_count * 72;
            let kind = flat_index / per_kind;
            let within_kind = flat_index % per_kind;
            let head = within_kind / (token_count * 72);
            let within_head = within_kind % (token_count * 72);
            let token = within_head / 72;
            let dim = within_head % 72;
            assert!(
                max_difference < 1e-5,
                "{label}: max difference was {max_difference} at \
                 [kind={kind}, head={head}, token={token}, dim={dim}] \
                 (fused={}, expected={})",
                fused_values[flat_index],
                expected_values[flat_index],
            );
        }

        let token_count = 2_i32;
        let qkv_values = (0..token_count * 3456)
            .map(|value| value as f32 * 0.001 - 2.0)
            .collect::<Vec<_>>();
        let cosine_values = (0..token_count * 72)
            .map(|value| 0.25 + value as f32 * 0.0001)
            .collect::<Vec<_>>();
        let sine_values = (0..token_count * 72)
            .map(|value| -0.5 + value as f32 * 0.0002)
            .collect::<Vec<_>>();
        let qkv = Array::from_slice(&qkv_values, &[token_count, 3456]);
        let cosine_input = Array::from_slice(&cosine_values, &[token_count, 1, 72]);
        let sine_input = Array::from_slice(&sine_values, &[token_count, 1, 72]);

        let qk = qkv
            .try_index((.., 0..2304))
            .unwrap()
            .reshape(&[token_count, 2, 16, 72])
            .unwrap()
            .transpose_axes(&[1, 2, 0, 3])
            .unwrap();
        let first = qk.try_index((.., .., .., 0..36)).unwrap();
        let second = qk.try_index((.., .., .., 36..72)).unwrap();
        let rotated = concatenate_axis(&[&(-&second), &first], -1).unwrap();
        let cosine = cosine_input.reshape(&[1, 1, token_count, 72]).unwrap();
        let sine = sine_input.reshape(&[1, 1, token_count, 72]).unwrap();
        let composed = &qk * &cosine + &rotated * &sine;

        let fused = paddleocr_rope_2d_qk(&qkv, &cosine_input, &sine_input).unwrap();
        assert_matches("general rotation", &fused, &composed, token_count as usize);

        let identity_cosine = Array::ones::<f32>(&[token_count, 1, 72]).unwrap();
        let identity_sine = Array::zeros::<f32>(&[token_count, 1, 72]).unwrap();
        let identity_cosine_broadcast = identity_cosine.reshape(&[1, 1, token_count, 72]).unwrap();
        let identity_sine_broadcast = identity_sine.reshape(&[1, 1, token_count, 72]).unwrap();
        let identity_expected =
            &qk * &identity_cosine_broadcast + &rotated * &identity_sine_broadcast;
        let identity_fused = paddleocr_rope_2d_qk(&qkv, &identity_cosine, &identity_sine).unwrap();
        assert_matches(
            "identity rotation",
            &identity_fused,
            &identity_expected,
            token_count as usize,
        );

        let rotation_cosine = Array::zeros::<f32>(&[token_count, 1, 72]).unwrap();
        let rotation_sine = Array::ones::<f32>(&[token_count, 1, 72]).unwrap();
        let rotation_cosine_broadcast = rotation_cosine.reshape(&[1, 1, token_count, 72]).unwrap();
        let rotation_sine_broadcast = rotation_sine.reshape(&[1, 1, token_count, 72]).unwrap();
        let rotation_expected =
            &qk * &rotation_cosine_broadcast + &rotated * &rotation_sine_broadcast;
        let rotation_fused = paddleocr_rope_2d_qk(&qkv, &rotation_cosine, &rotation_sine).unwrap();
        assert_matches(
            "pure rotation",
            &rotation_fused,
            &rotation_expected,
            token_count as usize,
        );
    }

    #[test]
    fn test_paddleocr_rope_2d_qk_accepts_low_precision_inputs() {
        let token_count = 2_i32;
        let qkv = Array::from_slice(
            &(0..token_count * 3456)
                .map(|value| value as f32 * 0.001 - 2.0)
                .collect::<Vec<_>>(),
            &[token_count, 3456],
        );
        let cosine = Array::ones::<f32>(&[token_count, 1, 72]).unwrap();
        let sine = Array::full::<f32>(&[token_count, 1, 72], crate::array!(0.125)).unwrap();

        for dtype in [crate::Dtype::Float16, crate::Dtype::Bfloat16] {
            let output = paddleocr_rope_2d_qk(
                qkv.as_dtype(dtype).unwrap(),
                cosine.as_dtype(dtype).unwrap(),
                sine.as_dtype(dtype).unwrap(),
            )
            .unwrap();
            assert_eq!(output.shape(), [2, 16, token_count, 72]);
            assert_eq!(output.dtype(), dtype);

            let output = output.as_dtype(crate::Dtype::Float32).unwrap();
            output.eval().unwrap();
            assert!(output
                .as_slice::<f32>()
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, weight, 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }
}
