# KernelArrays.jl

KernelArrays.jl provides array-like types that are convenient to use inside kernel functions.

When you work with raw buffers in GPU or other low-level code, a small slice of data is often easier to treat as a vector or matrix than as plain indices. This package wraps that pattern in a few lightweight static-array types.
