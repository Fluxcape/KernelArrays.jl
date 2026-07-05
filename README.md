# KernelArrays.jl

KernelArrays.jl provides array-like types that are convenient to use inside kernel functions.

When you work with raw buffers in GPU or other low-level code, a small slice of data is often easier to treat as a vector or matrix than as plain indices. This package wraps that pattern in a few lightweight static-array types.

## Features

- `KernelStaticArray`: an abstract static-array interface for kernel-friendly arrays
- `KS2Array`: a wrapper around a 2D array with a movable row and column offset
- `KS2Scalar`, `KS2Vector`, `KS2Matrix`, `KS2SquareMatrix`: convenient aliases for common shapes
- `Tuple(a)` conversion for static arrays
- `getindex` and `setindex!` support for wrapped 2D data

## Installation

Add the package with Julia's package manager:

```julia
using Pkg
Pkg.develop(path = ".")
```

## Usage

```julia
using KernelArrays

data = reshape(1:12, 3, 4)

v = KS2Vector{3, Int}(1, 2, data)
@show v[1]      # 4
@show v[2]      # 5
@show Tuple(v)   # (4, 5, 6)

v[2] = 99
@show data[1, 3] # 99
```

You can also create static arrays directly:

```julia
zero_vec = KS2Array{Tuple{3}, Int, 1}()
from_tuple = KS2Array{Tuple{2, 2}, Int, 2}((1, 2, 3, 4))
```

## Type Summary

- `KernelStaticScalar{T}`: scalar-like static array
- `KernelStaticVector{N, T}`: length-$N$ static vector
- `KernelStaticMatrix{M, N, T}`: $M \times N$ static matrix
- `KernelStaticSquareMatrix{N, T}`: $N \times N$ static matrix
- `KernelStaticVecOrMat{T}`: union of static vectors and matrices

- `KS2Scalar{T}`: scalar view into 2D data
- `KS2Vector{N, T}`: vector view into 2D data
- `KS2Matrix{M, N, T}`: matrix view into 2D data
- `KS2SquareMatrix{N, T}`: square-matrix view into 2D data
- `KS2VecOrMat{T}`: union of vector and matrix views

## Notes

`KS2Array` stores a reference to the source array, so updates made through the wrapper are reflected in the original data.
