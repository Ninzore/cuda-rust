// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::mem::MaybeUninit;

use cuda_core::{DeviceBuffer, DeviceCopy};

#[repr(C)]
#[derive(Clone, Copy)]
struct Padded {
    tag: u8,
    value: u32,
}

// SAFETY: the fields are byte-copyable, zero is valid, and the type owns no
// resources. DeviceCopy permits padding; it does not imply NoUninit.
unsafe impl DeviceCopy for Padded {}

fn uninitialized_element<T: DeviceCopy>(buffer: DeviceBuffer<T>) -> DeviceBuffer<u32> {
    buffer.cast_elem()
}

fn uninitialized_chunks<T: DeviceCopy>(
    buffer: DeviceBuffer<T>,
) -> Result<DeviceBuffer<u8>, DeviceBuffer<T>> {
    buffer.cast_chunks()
}

fn padded_element<T: DeviceCopy>(buffer: DeviceBuffer<T>) -> DeviceBuffer<[u32; 2]> {
    buffer.cast_elem()
}

fn padded_chunks<T: DeviceCopy>(
    buffer: DeviceBuffer<T>,
) -> Result<DeviceBuffer<u8>, DeviceBuffer<T>> {
    buffer.cast_chunks()
}

include!("../support/assert_not_impl.rs");

fn main() {
    assert_not_impl!(MaybeUninit<u32>: cuda_core::bytemuck::NoUninit);
    assert_not_impl!(Padded: cuda_core::bytemuck::NoUninit);
    let _ = (
        uninitialized_element::<MaybeUninit<u32>>,
        uninitialized_chunks::<MaybeUninit<u32>>,
        padded_element::<Padded>,
        padded_chunks::<Padded>,
    );
    fn is_device_copy<T: DeviceCopy>() {}
    is_device_copy::<MaybeUninit<u32>>();
    is_device_copy::<Padded>();
}
