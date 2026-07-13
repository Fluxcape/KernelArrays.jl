#=
 * @ author: chenyubao <chenyu.bao@outlook.com>
 * @ date: 2026-07-04 13:25:46
 * @ license: MIT
 =#

using Test
using KernelArrays
using StaticArrays

@testset "abstract type aliases" begin
    @test KernelStaticScalar{Float64} === KernelStaticArray{Tuple{}, Float64, 0}
    @test KernelStaticVector{3, Float64} === KernelStaticArray{Tuple{3}, Float64, 1}
    @test KernelStaticMatrix{2, 4, Float64} === KernelStaticArray{Tuple{2, 4}, Float64, 2}
    @test KernelStaticSquareMatrix{5, Float64} === KernelStaticArray{Tuple{5, 5}, Float64, 2}
end

@testset "KS1Array" begin
    data1 = randn(12)
    @test KS1Scalar(1, data1) isa KernelArrays.KS1Array{Tuple{}, Float64, 0}
    @test KS1Vector{3}(1, data1) isa KernelArrays.KS1Array{Tuple{3}, Float64, 1}
    @test KS1Matrix{2, 4}(1, data1) isa KernelArrays.KS1Array{Tuple{2, 4}, Float64, 2}
    @test KS1SquareMatrix{2}(1, data1) isa KernelArrays.KS1Array{Tuple{2, 2}, Float64, 2}
    v1 = KS1Vector{2}(1, data1)
    v2 = KS1Vector{2}(3, data1)
    v3 = KS1Vector{2}(5, data1)
    v4 = KS1Vector{2}(7, data1)
    m1 = KS1SquareMatrix{2}(9, data1)
    v1 .= 1
    v2 .= 2 .* v1
    v3 .= v1 .+ v2 .* 2
    v4 .= v1' * v2
    m1 .= v1 * v2'
    @test all(v1 .≈ 1)
    @test all(v2 .≈ 2)
    @test all(v3 .≈ 5)
    @test all(v4 .≈ 4)
    @test all((Tuple(m1) .- (2, 2, 2, 2) .≈ 0))
end

@testset "KS2Array" begin
    data2 = randn(3, 8)
    @test KS2Scalar(1, 1, data2) isa KS2Array{Tuple{}, Float64, 0}
    @test KS2Vector{3}(1, 1, data2) isa KS2Array{Tuple{3}, Float64, 1}
    @test KS2Matrix{2, 4}(1, 1, data2) isa KS2Array{Tuple{2, 4}, Float64, 2}
    @test KS2SquareMatrix{2}(1, 1, data2) isa KS2Array{Tuple{2, 2}, Float64, 2}
    v1 = KS2Vector{2}(1, 1, data2)
    v2 = KS2Vector{2}(2, 1, data2)
    v3 = KS2Vector{2}(3, 1, data2)
    v4 = KS2Vector{2}(1, 3, data2)
    m1 = KS2SquareMatrix{2}(1, 5, data2)
    v1 .= 1
    v2 .= 2 .* v1
    v3 .= v1 .+ v2 .* 2
    v4 .= v1' * v2
    m1 .= v1 * v2'
    @test all(v1 .≈ 1)
    @test all(v2 .≈ 2)
    @test all(v3 .≈ 5)
    @test all(v4 .≈ 4)
    @test all((Tuple(m1) .- (2, 2, 2, 2) .≈ 0))
end
