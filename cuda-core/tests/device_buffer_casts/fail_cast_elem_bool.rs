// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{DeviceBuffer, DeviceCopy};

// Only 0 and 1 are valid `bool` values, so a byte buffer must not become a
// `DeviceBuffer<bool>` through the safe reinterpreting cast.
fn forge<T: DeviceCopy>(bytes: DeviceBuffer<u8>) -> DeviceBuffer<T> {
    bytes.cast_elem()
}

include!("../support/assert_not_impl.rs");

fn main() {
    assert_not_impl!(bool: cuda_core::bytemuck::AnyBitPattern);
    let _ = forge::<bool>;
}
