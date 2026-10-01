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

## Recommended: KernelAbstractions.jl

`KernelArrays` is backend-agnostic — the same view code runs over a `Vector`, a
`CuArray`, or a `oneAPI` device vector, because the buffer type is only a type
parameter. To write a kernel **once** and run it on any of those backends, use
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl):

```julia
using KernelArrays
using KernelAbstractions

@kernel function add_kernel!(z, x, y)
    i = @index(Global)
    xi = KS1Vector{3}(3*(i-1)+1, x)   # view of x[3(i-1)+1 : 3i]
    yi = KS1Vector{3}(3*(i-1)+1, y)
    zi = KS1Vector{3}(3*(i-1)+1, z)
    zi .= xi .+ yi
    zi[1] = xi' * yi
end

backend = KernelAbstractions.CPU()   # or CUDABackend() / ROCBackend() / oneAPIBackend()
add_kernel!(backend, 32)(z, x, y, ndrange=round(Int, length(z) / 3))
```

`KernelArrays` itself does not depend on `KernelAbstractions` — it only needs
`StaticArrays` — so it also plugs into other frameworks such as CUDA.jl or
oneAPI.jl. KernelAbstractions is simply the recommended way to write a kernel
that is portable across backends.

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

## KS2: two-dimensional buffers

`KS2*` views slice a single row of a 2-D buffer — handy when the backing buffer
is laid out one row per record:

```julia
using KernelArrays

buf = randn(Float32, 4, 8)        # 4 records × 8 columns

r = KS2Vector{3}(2, 4, buf)       # record 2: view of buf[2, 4:6]
r .= 0
buf[2, 4:6] == zeros(Float32, 3)  # true — writes go through the view
```

## Operations return mutable `MVector`s

The views do not own their storage. Combining two views with an elementwise
operation therefore produces a fresh **mutable** static array (`MVector` /
`MArray`) that owns its data — not another view:

```julia
using KernelArrays

x = randn(Float32, 6)
y = randn(Float32, 6)

u = KS1Vector{3}(1, x)   # view of x[1:3]
v = KS1Vector{3}(4, y)   # view of y[4:6]

w = u .+ v               # w isa MVector{3, Float32} — it owns its storage
w = u + v                # + behaves the same as .+ here
w = u .* v               # .- , ./ , etc. likewise

w[1] = 0.0f0             # safe: x and y are untouched
x[1] == u[1]             # true — the view still reads the original buffer

u .= u .+ v              # to write back through a view, assign into it instead
```

Matrix-valued results behave the same way: the outer product of two views is
an `MMatrix`:

```julia
M = u * v'               # MMatrix{3, 3, Float32} — outer product, owns its storage
```

Because the result type is statically known (`MVector` / `MMatrix`), these
operations compile to type-stable code with no dynamic dispatch — so the same
expressions work unchanged **inside a GPU kernel** as well.

See the [API](@ref) page for the full list of types and aliases.
