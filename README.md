# KernelArrays.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxcape.github.io/KernelArrays.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxcape.github.io/KernelArrays.jl/dev/)
[![Build Status](https://github.com/Fluxcape/KernelArrays.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Fluxcape/KernelArrays.jl/actions/workflows/CI.yml)

Lightweight array-like static views for slicing chunks of a flat buffer inside
GPU kernels.

`KernelArrays` provides `KS1Array` and `KS2Array`: fixed-size,
`StaticArray`-compatible view types that reference a slice of an existing
backing buffer (a `Vector`, `CuArray`, or a `oneAPI` device vector) instead of
owning their own storage. Because the backing buffer type is carried in the
type system, a view can be built inside a kernel with no allocation and no
dynamic dispatch.

## Why

In a structure-of-arrays (SoA) kernel, each record's fields are stored in
separate flat buffers, and a kernel thread rebuilds a small vector/matrix over
the slice belonging to one record:

```julia
using KernelArrays
using KernelAbstractions

# x, y, z are device arrays (CuArray, oneArray, …) of Float32
@kernel function kernel(z, x, y)
    i = @index(Global)              # one record per work item (1-based)
    xi = KS1Vector{3}(3*(i-1)+1, x) # view of x[3(i-1)+1 : 3i]
    yi = KS1Vector{3}(3*(i-1)+1, y)
    zi = KS1Vector{3}(3*(i-1)+1, z)
    zi .= xi .+ yi                  # elementwise through the views
    zi[1] = xi' * yi                # dot product, allocation-free
end
```

Because `KS1Vector{3}` is a `StaticArray`, it supports the usual array algebra
(`.+`, `.-`, `.*`, `*`, `dot`, …), and `x' * y` is specialized to a single
fused reduction with no `Adjoint` allocation.

## The two view types

* `KS1Array{S, T, P, A}` — views a contiguous slice of a **1-D** buffer
  (SoA layout). `KS1Vector{N}(idx, data)` is the window `data[idx : idx+N-1]`.
  Build with `KS1Vector{N}(idx, data)`, `KS1Matrix{M, N}(idx, data)`,
  `KS1Scalar(idx, data)`, or `KS1SquareMatrix{N}(idx, data)`.
* `KS2Array{S, T, P, A}` — views a contiguous slice of a **2-D** buffer.
  `KS2Vector{N}(row, col, data)` is the window `data[row, col : col+N-1]`:
  `N` consecutive elements in a fixed `row`. Build with
  `KS2Vector{N}(row, col, data)`, `KS2Matrix{M, N}(row, col, data)`,
  `KS2Scalar(row, col, data)`, or `KS2SquareMatrix{N}(row, col, data)`.

`KS1` vs `KS2`: `KS1` takes a single offset into a flat buffer (SoA layout),
while `KS2` takes a `(row, col)` position and slices along a row — use it when
the backing buffer is naturally two-dimensional (e.g. one row per record).

Both store their index (`idx`, or `row`/`col`) and the buffer reference in
mutable `RefValue` fields, so a view can be re-pointed at another slice
in-place via `idx!` / `row!` / `col!` without reallocating.

## Example (CPU)

```julia
using KernelArrays

buf = randn(Float32, 8)

v = KS1Vector{3}(2, buf)   # view of buf[2:4]
v .= 0
v[1] = 1
buf[2] == 1                # true — writes go through to the buffer

w = KS1Vector{3}(5, buf)   # view of buf[5:7]
v' * w                     # dot product, no allocation

buf2 = randn(Float32, 3, 8)

a = KS2Vector{3}(2, 4, buf2)  # view of buf2[2, 4:6] — fixed row 2, columns 4..6
a .= 0
buf2[2, 5] == 0               # true — writes go through to the buffer
```

## GPU support

The views are designed to be constructed **inside** a GPU kernel. The backing
buffer type `A` is a type parameter and the fields are concrete `RefValue{...}`
(not the abstract `Ref{...}`), so the compiler can fully resolve the buffer
type and eliminate dynamic dispatch. The views are *not* `isbits` (they wrap
mutable references), so keep their construction and use within a single inlined
kernel body rather than passing them across a non-inlined function boundary.

## Design notes

* `KernelArrays` deliberately extends
  `Base.:*(u::Adjoint{<:Number,<:StaticVector}, v::StaticVector)` to return
  `dot(u.parent, v)`. This is type piracy on `StaticArrays`' types — intentional,
  so `x' * y` works uniformly for any `StaticVector` (including `MVector` and the
  kernel views) without allocating on the device.
* The abstract type `KernelStaticArray{S, T, P}` subtypes
  `StaticArraysCore.StaticArray`, so the views inherit the full `StaticArrays`
  API for free.

## Installation

```julia
import Pkg; Pkg.add("KernelArrays")
```
