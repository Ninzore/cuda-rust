/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A forbidden impl makes `_` ambiguous, adding an unexpected compiler error.
// Unlike a concrete missing-bound error, this does not list Pod implementors.
macro_rules! assert_not_impl {
    ($ty:ty: $bound:path) => {{
        trait AmbiguousIfImpl<A> {
            fn check() {}
        }
        impl<T> AmbiguousIfImpl<()> for T {}
        impl<T: $bound> AmbiguousIfImpl<u8> for T {}
        let _ = <$ty as AmbiguousIfImpl<_>>::check;
    }};
}
