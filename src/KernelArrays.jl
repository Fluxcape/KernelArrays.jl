#=
 * @ author: chenyubao <chenyu.bao@outlook.com>
 * @ date: 2026-07-03 20:10:35
 * @ license: MIT
 =#

"""
    KernelArrays

Lightweight array-like static views for slicing chunks of a flat buffer inside
GPU kernels.

[`KS1Array`](@ref) and [`KS2Array`](@ref) are fixed-size,
`StaticArrays.StaticArray`-compatible view types that reference a slice of an
existing backing buffer (a `Vector`, `CuArray`, or a `oneAPI` device vector)
instead of owning their own storage.

These types target structure-of-arrays (SoA) kernels: each record's fields live
in separate flat buffers, and a kernel thread rebuilds a small vector/matrix
view over the slice belonging to one record. The backing buffer type is carried
in the type system (the `A` parameter and concrete `RefValue` fields), so
in-kernel construction is type-stable and allocation-free.

See [`KS1Array`](@ref) and [`KS2Array`](@ref) for usage.
"""
module KernelArrays

include("KernelStaticArrays.jl")

end # module KernelArrays
