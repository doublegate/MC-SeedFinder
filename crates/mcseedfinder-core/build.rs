//! Build script: compiles the vendored cubiomes C library + FFI shim when the
//! `biomes` feature is enabled. No bindgen — the FFI is hand-declared in
//! `src/biomes.rs`, so the build only needs a C compiler (cc crate), making it
//! reproducible across clang/libclang versions and CI.

use std::path::Path;

fn main() {
    // Pure-Rust build (no cubiomes) when the feature is off.
    if std::env::var_os("CARGO_FEATURE_BIOMES").is_none() {
        return;
    }

    let cubiomes = "vendor/cubiomes";
    let shim = "csrc/shim.c";

    // Fail with an actionable message if the submodule wasn't checked out.
    if !Path::new(cubiomes).join("generator.h").exists() {
        panic!(
            "cubiomes sources missing at {cubiomes}/. Initialize the submodule:\n\
             \tgit submodule update --init --recursive\n\
             (or build with --no-default-features to skip the exact biome backend)."
        );
    }

    // cubiomes library translation units (exclude tests.c — it defines main()).
    let cubiomes_srcs = [
        "noise.c",
        "biomenoise.c",
        "biomes.c",
        "layers.c",
        "generator.c",
        "finders.c",
        "util.c",
        "quadbase.c",
    ];

    let mut build = cc::Build::new();
    build.include(cubiomes).include("csrc").opt_level(3);
    // cubiomes assumes two's-complement wraparound on signed overflow.
    build.flag_if_supported("-fwrapv");
    // The vendored C is upstream code; don't fail our build on its warnings.
    build.warnings(false);
    // Opt-in native build: pairs with `CARGO_BUILD_RUSTFLAGS="-C target-cpu=native"`
    // for the Rust side. Distributed wheels stay at the baseline.
    println!("cargo:rerun-if-env-changed=MCSF_NATIVE_CPU");
    if std::env::var("MCSF_NATIVE_CPU").as_deref() == Ok("1") {
        build.flag_if_supported("-march=native");
    }

    for src in cubiomes_srcs {
        let path = format!("{cubiomes}/{src}");
        println!("cargo:rerun-if-changed={path}");
        build.file(path);
    }
    println!("cargo:rerun-if-changed={shim}");
    build.file(shim);

    build.compile("cubiomes_shim");

    // System libs cubiomes needs at final link (libm always; pthread for the
    // parallel search helpers in quadbase.c). Unix-only; revisit for Windows.
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }
}
