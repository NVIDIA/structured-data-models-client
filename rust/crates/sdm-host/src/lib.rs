// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The host layer every binding shares: control codecs, the 64 KB control cap, Arrow admission, panic containment and run ownership.

#![forbid(unsafe_code)]
