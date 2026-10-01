# KernelArrays.jl

```@docs
KernelArrays
```

## Installation

```julia
import Pkg; Pkg.add("KernelArrays")
```

## Quick start

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

* [`KS1Array`](@ref) — views a contiguous slice of a **1-D** buffer (SoA
  layout). `KS1Vector{N}(idx, data)` is the window `data[idx : idx+N-1]`.
* [`KS2Array`](@ref) — views a contiguous slice of a **2-D** buffer.
  `KS2Vector{N}(row, col, data)` is `N` consecutive elements in a fixed `row`.

`KS1` vs `KS2`: `KS1` takes a single offset into a flat buffer (SoA layout),
while `KS2` takes a `(row, col)` position and slices along a row — use it when
the backing buffer is naturally two-dimensional (e.g. one row per record).

Both store their index (`idx`, or `row`/`col`) and the buffer reference in
mutable `RefValue` fields, so a view can be re-pointed at another slice
in-place via `idx!` / `row!` / `col!` without reallocating.

See the [API](@ref) page for the full list of types and aliases.
