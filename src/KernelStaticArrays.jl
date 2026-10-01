#=
 * @ author: chenyubao <chenyu.bao@outlook.com>
 * @ date: 2026-07-03 20:26:08
 * @ license: MIT
 =#

import Base: RefValue
import StaticArrays
import StaticArraysCore
import StaticArraysCore: tuple_prod
import StaticArraysCore: Size
import LinearAlgebra

"""
    *(u::Adjoint{<:Number, <:StaticVector}, v::StaticVector)

Returns `dot(u.parent, v)` — the real dot product of `u` and `v` — without
materializing an `Adjoint` wrapper. This extends `Base.:*` so that `x' * y`
works uniformly for any `StaticVector` and its adjoint (including `MVector` and
the [`KernelStaticArray`](@ref) views), compiling to a single fused reduction
with no allocation on the GPU.
"""
@inline function Base.:*(u::LinearAlgebra.Adjoint{<:Number, <:StaticArrays.StaticVector}, v::StaticArrays.StaticVector)
    return StaticArrays.dot(u.parent, v)
end

export KernelStaticArray
export KernelStaticScalar, KernelStaticVector, KernelStaticMatrix, KernelStaticSquareMatrix, KernelStaticVecOrMat
export KS1Array
export KS1Scalar, KS1Vector, KS1Matrix, KS1SquareMatrix, KS1VecOrMat
export KS2Array
export KS2Scalar, KS2Vector, KS2Matrix, KS2SquareMatrix, KS2VecOrMat

"""
    KernelStaticArray{S, T, P}

Abstract supertype of [`KS1Array`](@ref) and [`KS2Array`](@ref). Subtypes
`StaticArraysCore.StaticArray` and therefore inherit the full `StaticArrays`
API (indexing, broadcasting, `dot`, `*`, …).

Parameters:
  * `S`: a `Tuple` encoding the static size — `Tuple{3}` for a length-3 vector,
    `Tuple{2, 4}` for a `2×4` matrix, `Tuple{}` for a scalar.
  * `T <: Real`: element type of the backing buffer.
  * `P`: view dimensionality — `0` (scalar), `1` (vector), or `2` (matrix).
"""
abstract type KernelStaticArray{S, T <: Real, P} <: StaticArraysCore.StaticArray{S, T, P} end

"""
    KernelStaticScalar{T}

Alias for `KernelStaticArray{Tuple{}, T, 0}` — a scalar kernel static view.
"""
const KernelStaticScalar{T} = KernelStaticArray{Tuple{}, T, 0}

"""
    KernelStaticVector{N, T}

Alias for `KernelStaticArray{Tuple{N}, T, 1}` — a length-`N` kernel static vector.
"""
const KernelStaticVector{N, T} = KernelStaticArray{Tuple{N}, T, 1}

"""
    KernelStaticMatrix{M, N, T}

Alias for `KernelStaticArray{Tuple{M, N}, T, 2}` — an `M×N` kernel static matrix.
"""
const KernelStaticMatrix{M, N, T} = KernelStaticArray{Tuple{M, N}, T, 2}

"""
    KernelStaticSquareMatrix{N, T}

Alias for `KernelStaticArray{Tuple{N, N}, T, 2}` — an `N×N` kernel static square
matrix.
"""
const KernelStaticSquareMatrix{N, T} = KernelStaticArray{Tuple{N, N}, T, 2}

"""
    KernelStaticVecOrMat{T}

Union of [`KernelStaticVector`](@ref) and [`KernelStaticMatrix`](@ref) with
element type `T`. Useful for dispatch on any kernel static vector or matrix.
"""
const KernelStaticVecOrMat{T} = Union{KernelStaticVector{<:Any, T}, KernelStaticMatrix{<:Any, <:Any, T}}

@inline function Base.Tuple(a::KernelStaticArray{S, T, P})::NTuple{tuple_prod(S), T} where {S <: Tuple, T <: Real, P}
    L = tuple_prod(S)
    return ntuple(i -> getindex(a, i), L)
end

@inline function Base.strides(a::KernelStaticArray{S, T, P}) where {S <: Tuple, T <: Real, P}
    return Base.size_to_strides(1, size(a)...)
end

@inline function StaticArrays.similar_type(::Type{SA}, ::Type{T}, s::Size{S}) where {SA <: KernelStaticArray, T, S}
    return StaticArrays.mutable_similar_type(T, s, StaticArrays.length_val(s))
end

# * KS1Array
#
# `A` carries the concrete array type of the backing buffer (e.g. `Vector{Float32}`,
# `CuArray{Float32,1}`, `oneDeviceVector{Float32,1}`) and the fields use the concrete
# `RefValue{...}` rather than the abstract `Ref{...}`, so the buffer type stays known in
# the type system instead of relying on compiler partial-struct inference.

"""
    KS1Array{S, T, P, A} <: KernelStaticArray{S, T, P}

A static view of a contiguous slice of a 1-D backing buffer
(structure-of-arrays layout). The view covers `tuple_prod(S)` consecutive
elements of `data`, starting at `data[idx]`.

Parameters:
  * `S`: static size tuple (`Tuple{3}` vector, `Tuple{M, N}` matrix, …).
  * `T <: Real`: element type of the backing buffer.
  * `P`: view dimensionality — `0`, `1`, or `2`.
  * `A <: AbstractArray{T, 1}`: concrete type of the backing buffer.

The convenience aliases [`KS1Scalar`](@ref), [`KS1Vector`](@ref),
[`KS1Matrix`](@ref), and [`KS1SquareMatrix`](@ref) infer `S`, `T`, and `P`:

```julia
buf = randn(Float32, 8)
v = KS1Vector{3}(2, buf)   # view of buf[2], buf[3], buf[4]
v .= 0                     # zero the slice through the view
v[1] = 1                   # buf[2] is now 1
```

The view holds `idx` and the buffer reference in mutable `RefValue` fields, so
it can be re-pointed at another slice in-place with `idx!` without
reallocating.
"""
struct KS1Array{S, T <: Real, P, A <: AbstractArray{T, 1}} <: KernelStaticArray{S, T, P}
    idx_::RefValue{Int}
    data_::RefValue{A}
end

"""
    KS1Scalar{T, A}

Alias for `KS1Array{Tuple{}, T, 0, A}` — a scalar view of a 1-D backing buffer.
"""
const KS1Scalar{T <: Real, A <: AbstractArray{T, 1}} = KS1Array{Tuple{}, T, 0, A}

"""
    KS1Vector{N, T, A}

Alias for `KS1Array{Tuple{N}, T, 1, A}` — a length-`N` static vector view of a
1-D backing buffer. Construct with `KS1Vector{N}(idx, data)`.
"""
const KS1Vector{N, T <: Real, A <: AbstractArray{T, 1}} = KS1Array{Tuple{N}, T, 1, A}

"""
    KS1Matrix{M, N, T, A}

Alias for `KS1Array{Tuple{M, N}, T, 2, A}` — an `M×N` static matrix view of a
1-D backing buffer. Construct with `KS1Matrix{M, N}(idx, data)`.
"""
const KS1Matrix{M, N, T <: Real, A <: AbstractArray{T, 1}} = KS1Array{Tuple{M, N}, T, 2, A}

"""
    KS1SquareMatrix{N, T, A}

Alias for `KS1Array{Tuple{N, N}, T, 2, A}` — an `N×N` static square matrix view
of a 1-D backing buffer. Construct with `KS1SquareMatrix{N}(idx, data)`.
"""
const KS1SquareMatrix{N, T <: Real, A <: AbstractArray{T, 1}} = KS1Array{Tuple{N, N}, T, 2, A}

"""
    KS1VecOrMat{T, A}

Union of [`KS1Vector`](@ref) and [`KS1Matrix`](@ref) over a 1-D backing buffer
with element type `T`.
"""
const KS1VecOrMat{T <: Real, A <: AbstractArray{T, 1}} = Union{KS1Vector{<:Any, T, A}, KS1Matrix{<:Any, <:Any, T, A}}

@inline function _idx(a::KS1Array{S, T, P, A})::Int where {S <: Tuple, T <: Real, P, A}
    return a.idx_.x
end

@inline function _data(a::KS1Array{S, T, P, A}) where {S <: Tuple, T <: Real, P, A}
    return a.data_.x
end

@inline function Base.getindex(a::KS1Array{S, T, P, A}, i::Int) where {S <: Tuple, T <: Real, P, A}
    return @inbounds _data(a)[_idx(a) + i - 1]
end

@inline function Base.setindex!(a::KS1Array{S, T, P, A}, v::Real, i::Int) where {S <: Tuple, T <: Real, P, A}
    @inbounds _data(a)[_idx(a) + i - 1] = T(v)
end

@inline function idx!(a::KS1Array{S, T, P, A}, idx::Integer)::Int where {S <: Tuple, T <: Real, P, A}
    return a.idx_.x = Int(idx)
end

# * Constructors for KS1Array

@inline function KS1Array{S, T, P}(
    idx::Integer,
    data::A,
)::KS1Array{S, T, P, A} where {S <: Tuple, T <: Real, P, A <: AbstractArray{T, 1}}
    return KS1Array{S, T, P, A}(RefValue{Int}(Int(idx)), RefValue{A}(data))
end

@inline function KS1Array{S}(
    idx::Integer,
    data::A,
)::KS1Array{S, eltype(A), length(S.parameters), A} where {S <: Tuple, A <: AbstractArray{<:Real, 1}}
    return KS1Array{S, eltype(A), length(S.parameters), A}(RefValue{Int}(Int(idx)), RefValue{A}(data))
end

@inline function KS1Array{S, T}(
    idx::Integer,
    data::A,
)::KS1Array{S, T, length(S.parameters), A} where {S <: Tuple, T <: Real, A <: AbstractArray{T, 1}}
    return KS1Array{S, T, length(S.parameters), A}(RefValue{Int}(Int(idx)), RefValue{A}(data))
end

@inline function KS1Array{S, T, P}()::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> zero(T), tuple_prod(S)))
end

@inline function KS1Array{S, T, P}(
    x::NTuple{L, T},
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P, L}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> x[i], tuple_prod(S)))
end

@inline function KS1Array{S, T, P}(
    ::UndefInitializer,
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> zero(T), tuple_prod(S)))
end

@inline function KS1Array{S, T, P}(
    x::Base.Tuple,
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(x)
end

# * Constructors for KS1Scalar, KS1Vector, KS1Matrix, KS1SquareMatrix

@inline function KS1Scalar(idx::Integer, data::A) where {A <: AbstractArray{<:Real, 1}}
    return KS1Array{Tuple{}, eltype(A), 0}(idx, data)
end

@inline function KS1Vector{N}(idx::Integer, data::A) where {N, A <: AbstractArray{<:Real, 1}}
    return KS1Array{Tuple{N}, eltype(A), 1}(idx, data)
end

@inline function KS1Matrix{M, N}(idx::Integer, data::A) where {M, N, A <: AbstractArray{<:Real, 1}}
    return KS1Array{Tuple{M, N}, eltype(A), 2}(idx, data)
end

@inline function KS1SquareMatrix{N}(idx::Integer, data::A) where {N, A <: AbstractArray{<:Real, 1}}
    return KS1Array{Tuple{N, N}, eltype(A), 2}(idx, data)
end

# * KS2Array

"""
    KS2Array{S, T, P, A} <: KernelStaticArray{S, T, P}

A static view of a contiguous slice of a 2-D backing buffer. The view covers
`tuple_prod(S)` consecutive elements of `data` along one dimension, starting at
`data[row, col]`:

  * `P == 1` — a row slice: `data[row, col]`, `data[row, col+1]`, …;
  * `P == 2` — element `i` maps to `data[row, col + i - 1]`.

Parameters:
  * `S`: static size tuple.
  * `T <: Real`: element type of the backing buffer.
  * `P`: view dimensionality — `0`, `1`, or `2`.
  * `A <: AbstractArray{T, 2}`: concrete type of the backing buffer.

The convenience aliases [`KS2Scalar`](@ref), [`KS2Vector`](@ref),
[`KS2Matrix`](@ref), and [`KS2SquareMatrix`](@ref) infer `S`, `T`, and `P`:

```julia
buf = randn(Float32, 3, 8)
v = KS2Vector{3}(1, 2, buf)   # view of buf[1,2], buf[1,3], buf[1,4]
```

The view holds `row`, `col`, and the buffer reference in mutable `RefValue`
fields, so it can be re-pointed in-place with `row!`/`col!` without
reallocating.
"""
struct KS2Array{S, T <: Real, P, A <: AbstractArray{T, 2}} <: KernelStaticArray{S, T, P}
    row_::RefValue{Int}
    col_::RefValue{Int}
    data_::RefValue{A}
end

"""
    KS2Scalar{T, A}

Alias for `KS2Array{Tuple{}, T, 0, A}` — a scalar view of a 2-D backing buffer.
"""
const KS2Scalar{T <: Real, A <: AbstractArray{T, 2}} = KS2Array{Tuple{}, T, 0, A}

"""
    KS2Vector{N, T, A}

Alias for `KS2Array{Tuple{N}, T, 1, A}` — a length-`N` static vector view of a
2-D backing buffer. Construct with `KS2Vector{N}(row, col, data)`.
"""
const KS2Vector{N, T <: Real, A <: AbstractArray{T, 2}} = KS2Array{Tuple{N}, T, 1, A}

"""
    KS2Matrix{M, N, T, A}

Alias for `KS2Array{Tuple{M, N}, T, 2, A}` — an `M×N` static matrix view of a
2-D backing buffer. Construct with `KS2Matrix{M, N}(row, col, data)`.
"""
const KS2Matrix{M, N, T <: Real, A <: AbstractArray{T, 2}} = KS2Array{Tuple{M, N}, T, 2, A}

"""
    KS2SquareMatrix{N, T, A}

Alias for `KS2Array{Tuple{N, N}, T, 2, A}` — an `N×N` static square matrix view
of a 2-D backing buffer. Construct with `KS2SquareMatrix{N}(row, col, data)`.
"""
const KS2SquareMatrix{N, T <: Real, A <: AbstractArray{T, 2}} = KS2Array{Tuple{N, N}, T, 2, A}

"""
    KS2VecOrMat{T, A}

Union of [`KS2Vector`](@ref) and [`KS2Matrix`](@ref) over a 2-D backing buffer
with element type `T`.
"""
const KS2VecOrMat{T <: Real, A <: AbstractArray{T, 2}} = Union{KS2Vector{<:Any, T, A}, KS2Matrix{<:Any, <:Any, T, A}}

@inline function _row(a::KS2Array{S, T, P, A})::Int where {S <: Tuple, T <: Real, P, A}
    return a.row_.x
end

@inline function _col(a::KS2Array{S, T, P, A})::Int where {S <: Tuple, T <: Real, P, A}
    return a.col_.x
end

@inline function _data(a::KS2Array{S, T, P, A}) where {S <: Tuple, T <: Real, P, A}
    return a.data_.x
end

@inline function Base.getindex(a::KS2Array{S, T, P, A}, i::Int) where {S <: Tuple, T <: Real, P, A}
    return @inbounds _data(a)[_row(a), _col(a) + i - 1]
end

@inline function Base.setindex!(a::KS2Array{S, T, P, A}, v::Real, i::Int) where {S <: Tuple, T <: Real, P, A}
    @inbounds _data(a)[_row(a), _col(a) + i - 1] = T(v)
end

@inline function row!(a::KS2Array{S, T, P, A}, r::Integer)::Int where {S <: Tuple, T <: Real, P, A}
    return a.row_.x = Int(r)
end

@inline function col!(a::KS2Array{S, T, P, A}, c::Integer)::Int where {S <: Tuple, T <: Real, P, A}
    return a.col_.x = Int(c)
end

# * Constructors for KS2Array

@inline function KS2Array{S, T, P}(
    row::Integer,
    col::Integer,
    data::A,
)::KS2Array{S, T, P, A} where {S <: Tuple, T <: Real, P, A <: AbstractArray{T, 2}}
    return KS2Array{S, T, P, A}(RefValue{Int}(Int(row)), RefValue{Int}(Int(col)), RefValue{A}(data))
end

@inline function KS2Array{S}(
    row::Integer,
    col::Integer,
    data::A,
)::KS2Array{S, eltype(A), length(S.parameters), A} where {S <: Tuple, A <: AbstractArray{<:Real, 2}}
    return KS2Array{S, eltype(A), length(S.parameters), A}(
        RefValue{Int}(Int(row)),
        RefValue{Int}(Int(col)),
        RefValue{A}(data),
    )
end

@inline function KS2Array{S, T}(
    row::Integer,
    col::Integer,
    data::A,
)::KS2Array{S, T, length(S.parameters), A} where {S <: Tuple, T <: Real, A <: AbstractArray{T, 2}}
    return KS2Array{S, T, length(S.parameters), A}(RefValue{Int}(Int(row)), RefValue{Int}(Int(col)), RefValue{A}(data))
end

@inline function KS2Array{S, T, P}()::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> zero(T), tuple_prod(S)))
end

@inline function KS2Array{S, T, P}(
    x::NTuple{L, T},
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P, L}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> x[i], tuple_prod(S)))
end

@inline function KS2Array{S, T, P}(
    ::UndefInitializer,
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> zero(T), tuple_prod(S)))
end

@inline function KS2Array{S, T, P}(
    x::Base.Tuple,
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(x)
end

# * Constructors for KS2Scalar, KS2Vector, KS2Matrix, KS2SquareMatrix

@inline function KS2Scalar(row::Integer, col::Integer, data::A) where {A <: AbstractArray{<:Real, 2}}
    return KS2Array{Tuple{}, eltype(A), 0}(row, col, data)
end

@inline function KS2Vector{N}(row::Integer, col::Integer, data::A) where {N, A <: AbstractArray{<:Real, 2}}
    return KS2Array{Tuple{N}, eltype(A), 1}(row, col, data)
end

@inline function KS2Matrix{M, N}(row::Integer, col::Integer, data::A) where {M, N, A <: AbstractArray{<:Real, 2}}
    return KS2Array{Tuple{M, N}, eltype(A), 2}(row, col, data)
end

@inline function KS2SquareMatrix{N}(row::Integer, col::Integer, data::A) where {N, A <: AbstractArray{<:Real, 2}}
    return KS2Array{Tuple{N, N}, eltype(A), 2}(row, col, data)
end
