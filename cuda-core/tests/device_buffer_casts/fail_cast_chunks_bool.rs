// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{DeviceBuffer, DeviceCopy};

// The length-adjusting cast has the same bound as `cast_elem`: `u32` words
// must not be regrouped into `bool` bytes.
fn forge<T: DeviceCopy>(words: DeviceBuffer<u32>) -> Result<DeviceBuffer<T>, DeviceBuffer<u32>> {
    words.cast_chunks()
}

include!("../support/assert_not_impl.rs");

fn main() {
    assert_not_impl!(bool: cuda_core::bytemuck::AnyBitPattern);
    let _ = forge::<bool>;
}
