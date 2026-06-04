#![cfg_attr(target_arch = "wasm32", feature(stdarch_wasm_atomic_wait))]

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
