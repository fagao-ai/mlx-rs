extern crate cmake;

use cmake::Config;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const MAX_JIT_METALLIB_SIZE: u64 = 16 * 1024 * 1024;

/// Find the clang runtime library path dynamically using xcrun
fn find_clang_rt_path() -> Option<String> {
    // Use xcrun to find the active toolchain path
    let output = Command::new("xcrun")
        .args(["--show-sdk-platform-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Get the developer directory which contains the toolchain
    let output = Command::new("xcode-select")
        .args(["--print-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let toolchain_base = format!(
        "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
        developer_dir
    );

    // Find the clang version directory (it varies by Xcode version)
    let clang_dir = std::fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        let clang_rt_lib = darwin_path.join("libclang_rt.osx.a");
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

fn build_and_link_mlx_c() -> PathBuf {
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");
    config.define("MLX_BUILD_GGUF", "OFF");
    config.define("MLX_METAL_JIT", "ON");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }

    dst
}

#[cfg(feature = "metal")]
fn embed_mlx_metallib(dst: &Path, out_path: &Path) {
    let metallib_path = dst.join("build/lib/mlx.metallib");
    let metallib_size = fs::metadata(&metallib_path)
        .unwrap_or_else(|error| {
            panic!(
                "failed to read generated MLX metallib {}: {error}",
                metallib_path.display()
            )
        })
        .len();
    assert!(
        metallib_size > 0,
        "generated MLX metallib is empty: {}",
        metallib_path.display()
    );
    assert!(
        metallib_size <= MAX_JIT_METALLIB_SIZE,
        "generated MLX metallib is {metallib_size} bytes, expected a JIT metallib no larger than {MAX_JIT_METALLIB_SIZE} bytes: {}",
        metallib_path.display()
    );

    let escaped_metallib_path = metallib_path
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    let assembly_path = out_path.join("embedded_metallib.S");
    let assembly = format!(
        ".section __TEXT,__mlx_metal\n\
         .balign 16\n\
         .globl _mlx_embedded_metallib_start\n\
         _mlx_embedded_metallib_start:\n\
         .incbin \"{escaped_metallib_path}\"\n\
         .globl _mlx_embedded_metallib_end\n\
         _mlx_embedded_metallib_end:\n\
         .subsections_via_symbols\n\
         .no_dead_strip _mlx_embedded_metallib_start\n"
    );
    fs::write(&assembly_path, assembly).unwrap_or_else(|error| {
        panic!(
            "failed to write embedded metallib assembly {}: {error}",
            assembly_path.display()
        )
    });
    cc::Build::new()
        .cargo_metadata(false)
        .file(&assembly_path)
        .compile("mlx_metallib");
    println!("cargo:rustc-link-search=native={}", out_path.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=mlx_metallib");
}

fn main() {
    // The CMake project is nested beneath this Rust crate, so Cargo does not
    // discover these native-source dependencies automatically.
    println!("cargo:rerun-if-changed=src/mlx-c/CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/mlx-c/mlx/c/fast.cpp");
    println!("cargo:rerun-if-changed=src/mlx-c/mlx/c/fast.h");
    println!("cargo:rerun-if-changed=src/mlx-c/patches/apply-patches.sh");
    println!("cargo:rerun-if-changed=src/mlx-c/patches/mlx-sdpa-head-dim-72.patch");
    println!("cargo:rerun-if-changed=src/mlx-c/patches/mlx-shapeless-split-output-shapes.patch");
    println!("cargo:rerun-if-changed=src/mlx-c/patches/mlx-addmm-gelu-experiment.patch");
    println!("cargo:rerun-if-changed=src/mlx-c/patches/mlx-addmm-gelu-api.patch");
    println!("cargo:rerun-if-changed=src/mlx-c/patches/mlx-precise-power.patch");
    let dst = build_and_link_mlx_c();

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    #[cfg(feature = "metal")]
    embed_mlx_metallib(&dst, &out_path);

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
