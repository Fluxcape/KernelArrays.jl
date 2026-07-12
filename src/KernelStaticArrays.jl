#=
 * @ author: chenyubao <chenyu.bao@outlook.com>
 * @ date: 2026-07-03 20:26:08
 * @ license: MIT
 =#

import StaticArrays
import StaticArraysCore
import StaticArraysCore: tuple_prod
import StaticArraysCore: Size

export KernelStaticArray
export KernelStaticScalar, KernelStaticVector, KernelStaticMatrix, KernelStaticSquareMatrix, KernelStaticVecOrMat
export KS1Array
export KS1Scalar, KS1Vector, KS1Matrix, KS1SquareMatrix, KS1VecOrMat
export KS2Array
export KS2Scalar, KS2Vector, KS2Matrix, KS2SquareMatrix, KS2VecOrMat

abstract type KernelStaticArray{S, T <: Real, P} <: StaticArraysCore.StaticArray{S, T, P} end

const KernelStaticScalar{T} = KernelStaticArray{Tuple{}, T, 0}
const KernelStaticVector{N, T} = KernelStaticArray{Tuple{N}, T, 1}
const KernelStaticMatrix{M, N, T} = KernelStaticArray{Tuple{M, N}, T, 2}
const KernelStaticSquareMatrix{N, T} = KernelStaticArray{Tuple{N, N}, T, 2}
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

struct KS1Array{S, T <: Real, P} <: KernelStaticArray{S, T, P}
    idx_::Ref{Int}
    data_::Ref
end

const KS1Scalar{T <: Real} = KS1Array{Tuple{}, T, 0}
const KS1Vector{N, T <: Real} = KS1Array{Tuple{N}, T, 1}
const KS1Matrix{M, N, T <: Real} = KS1Array{Tuple{M, N}, T, 2}
const KS1SquareMatrix{N, T <: Real} = KS1Array{Tuple{N, N}, T, 2}
const KS1VecOrMat{T <: Real} = Union{KS1Vector{<:Any, T}, KS1Matrix{<:Any, <:Any, T}}

@inline function _idx(a::KS1Array{S, T, P})::Int where {S <: Tuple, T <: Real, P}
    return getfield(a, :idx_).x
end

@inline function _data(a::KS1Array{S, T, P}) where {S <: Tuple, T <: Real, P}
    return getfield(a, :data_).x
end

@inline function Base.getindex(a::KS1Array{S, T, P}, i::Int) where {S <: Tuple, T <: Real, P}
    return @inbounds _data(a)[_idx(a) + i - 1]
end

@inline function Base.setindex!(a::KS1Array{S, T, P}, v::Real, i::Int) where {S <: Tuple, T <: Real, P}
    @inbounds _data(a)[_idx(a) + i - 1] = T(v)
end

@inline function idx!(a::KS1Array{S, T, P}, idx::Integer)::Int where {S <: Tuple, T <: Real, P}
    return a.idx_.x = Int(idx)
end

# * Constructors for KS1Array

@inline function KS1Array{S, T, P}(
    idx::Integer,
    data::AbstractArray{T, 1},
)::KS1Array{S, T, P} where {S <: Tuple, T <: Real, P}
    return KS1Array{S, T, P}(Ref{Int}(Int(idx)), Ref{typeof(data)}(data))
end

@inline function KS1Array{S}(
    idx::Integer,
    data::AbstractArray{T, 1},
)::KS1Array{S, T, length(S.parameters)} where {S <: Tuple, T <: Real}
    return KS1Array{S, T, length(S.parameters)}(Ref{Int}(Int(idx)), Ref{typeof(data)}(data))
end

@inline function KS1Array{S, T}(
    idx::Integer,
    data::AbstractArray{T, 1},
)::KS1Array{S, T, length(S.parameters)} where {S <: Tuple, T <: Real}
    return KS1Array{S, T, length(S.parameters)}(Ref{Int}(Int(idx)), Ref{typeof(data)}(data))
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

@inline function KS1Scalar(idx::Integer, data::AbstractArray{T, 1}) where {T <: Real}
    return KS1Scalar{T}(idx, data)
end

@inline function KS1Vector{N}(idx::Integer, data::AbstractArray{T, 1}) where {N, T <: Real}
    return KS1Vector{N, T}(idx, data)
end

@inline function KS1Matrix{M, N}(idx::Integer, data::AbstractArray{T, 1}) where {M, N, T <: Real}
    return KS1Matrix{M, N, T}(idx, data)
end

@inline function KS1SquareMatrix{N}(idx::Integer, data::AbstractArray{T, 1}) where {N, T <: Real}
    return KS1SquareMatrix{N, T}(idx, data)
end

# * KS2Array

struct KS2Array{S, T <: Real, P} <: KernelStaticArray{S, T, P}
    row_::Ref{Int}
    col_::Ref{Int}
    data_::Ref
end

const KS2Scalar{T <: Real} = KS2Array{Tuple{}, T, 0}
const KS2Vector{N, T <: Real} = KS2Array{Tuple{N}, T, 1}
const KS2Matrix{M, N, T <: Real} = KS2Array{Tuple{M, N}, T, 2}
const KS2SquareMatrix{N, T <: Real} = KS2Array{Tuple{N, N}, T, 2}
const KS2VecOrMat{T <: Real} = Union{KS2Vector{<:Any, T}, KS2Matrix{<:Any, <:Any, T}}

@inline function _row(a::KS2Array{S, T, P})::Int where {S <: Tuple, T <: Real, P}
    return getfield(a, :row_).x
end

@inline function _col(a::KS2Array{S, T, P})::Int where {S <: Tuple, T <: Real, P}
    return getfield(a, :col_).x
end

@inline function _data(a::KS2Array{S, T, P}) where {S <: Tuple, T <: Real, P}
    return getfield(a, :data_).x
end

@inline function Base.getindex(a::KS2Array{S, T, P}, i::Int) where {S <: Tuple, T <: Real, P}
    return @inbounds _data(a)[_row(a), _col(a) + i - 1]
end

@inline function Base.setindex!(a::KS2Array{S, T, P}, v::Real, i::Int) where {S <: Tuple, T <: Real, P}
    @inbounds _data(a)[_row(a), _col(a) + i - 1] = T(v)
end

@inline function row!(a::KS2Array{S, T, P}, r::Integer)::Int where {S <: Tuple, T <: Real, P}
    return a.row_.x = Int(r)
end

@inline function col!(a::KS2Array{S, T, P}, c::Integer)::Int where {S <: Tuple, T <: Real, P}
    return a.col_.x = Int(c)
end

# * Constructors for KS2Array

@inline function KS2Array{S, T, P}(
    row::Integer,
    col::Integer,
    data::AbstractArray{T, 2},
)::KS2Array{S, T, P} where {S <: Tuple, T <: Real, P}
    return KS2Array{S, T, P}(Ref{Int}(Int(row)), Ref{Int}(Int(col)), Ref{typeof(data)}(data))
end

@inline function KS2Array{S}(
    row::Integer,
    col::Integer,
    data::AbstractArray{T, 2},
)::KS2Array{S, T, length(S.parameters)} where {S <: Tuple, T <: Real}
    return KS2Array{S, T, length(S.parameters)}(Ref{Int}(Int(row)), Ref{Int}(Int(col)), Ref{typeof(data)}(data))
end

@inline function KS2Array{S, T}(
    row::Integer,
    col::Integer,
    data::AbstractArray{T, 2},
)::KS2Array{S, T, length(S.parameters)} where {S <: Tuple, T <: Real}
    return KS2Array{S, T, length(S.parameters)}(Ref{Int}(Int(row)), Ref{Int}(Int(col)), Ref{typeof(data)}(data))
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

@inline function KS2Scalar(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {T <: Real}
    return KS2Scalar{T}(row, col, data)
end

@inline function KS2Vector{N}(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {N, T <: Real}
    return KS2Vector{N, T}(row, col, data)
end

@inline function KS2Matrix{M, N}(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {M, N, T <: Real}
    return KS2Matrix{M, N, T}(row, col, data)
end

@inline function KS2SquareMatrix{N}(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {N, T <: Real}
    return KS2SquareMatrix{N, T}(row, col, data)
end
