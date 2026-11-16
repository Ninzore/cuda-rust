// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Only 0 and 1 are valid `bool` values, so a byte tensor must not become a
// `Tensor<bool>` through the safe reinterpreting cast: the target type has to
// implement `bytemuck::AnyBitPattern`, and `bool` does not.
use cuda_core::DType;
use cutile::tensor::Tensor;
use std::sync::Arc;

fn forge<T: DType>(bytes: Arc<Tensor<u8>>) -> Arc<Tensor<T>> {
    bytes.reinterpret::<T>(&[4]).unwrap()
}

fn main() {
    // An unexpected AnyBitPattern impl for bool makes `_` ambiguous.
    trait AmbiguousIfAnyBitPattern<A> {
        fn check() {}
    }
    impl<T> AmbiguousIfAnyBitPattern<()> for T {}
    impl<T: cuda_core::bytemuck::AnyBitPattern> AmbiguousIfAnyBitPattern<u8> for T {}
    let _ = <bool as AmbiguousIfAnyBitPattern<_>>::check;
    let _ = forge::<bool>;
}
