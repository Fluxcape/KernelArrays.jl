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
    data1 = randn(8)
    @test KernelArrays.KS1Scalar(1, data1) isa KernelArrays.KS1Array{Tuple{}, Float64, 0}
    @test KernelArrays.KS1Vector{3}(1, data1) isa KernelArrays.KS1Array{Tuple{3}, Float64, 1}
    @test KernelArrays.KS1Matrix{2, 4}(1, data1) isa KernelArrays.KS1Array{Tuple{2, 4}, Float64, 2}
    @test KernelArrays.KS1SquareMatrix{2}(1, data1) isa KernelArrays.KS1Array{Tuple{2, 2}, Float64, 2}
end

@testset "KS2Array" begin
    data2 = randn(3, 8)
    @test KS2Scalar(1, 1, data2) isa KS2Array{Tuple{}, Float64, 0}
    @test KS2Vector{3}(1, 1, data2) isa KS2Array{Tuple{3}, Float64, 1}
    @test KS2Matrix{2, 4}(1, 1, data2) isa KS2Array{Tuple{2, 4}, Float64, 2}
    @test KS2SquareMatrix{2}(1, 1, data2) isa KS2Array{Tuple{2, 2}, Float64, 2}
end

@testset "sizes" begin
    data2 = randn(3, 8)
    @test size(KS2Scalar(1, 1, data2)) == ()
    @test size(KS2Vector{3}(1, 1, data2)) == (3,)
    @test size(KS2Matrix{2, 4}(1, 1, data2)) == (2, 4)
    @test size(KS2SquareMatrix{2}(1, 1, data2)) == (2, 2)
end

@testset "getindex / setindex!" begin
    data1 = collect(1.0:10.0)
    a = KernelArrays.KS1Array{Tuple{3}, Float64, 1}(3, data1)
    @test a[1] == 3.0
    @test a[2] == 4.0
    @test a[3] == 5.0
    a[2] = 99.0
    @test a[2] == 99.0
    @test data1[4] == 99.0

    data2 = randn(5, 6)
    b = KS2Vector{3}(2, 3, data2)
    @test b[1] == data2[2, 3]
    @test b[2] == data2[2, 4]
    @test b[3] == data2[2, 5]
end

@testset "Tuple conversion" begin
    data1 = collect(1.0:10.0)
    a = KernelArrays.KS1Array{Tuple{3}, Float64, 1}(3, data1)
    @test Tuple(a) == (3.0, 4.0, 5.0)

    data2 = randn(5, 8)
    b = KS2Matrix{2, 3}(2, 3, data2)
    @test Tuple(b) == ntuple(i -> data2[2, 3 + i - 1], 6)
end

@testset "MArray constructors" begin
    z = KS2Array{Tuple{3}, Float64, 1}()
    @test z isa MArray{Tuple{3}, Float64, 1, 3}
    @test all(==(0.0), z)

    t = KS2Array{Tuple{3}, Float64, 1}((1.0, 2.0, 3.0))
    @test t isa MArray{Tuple{3}, Float64, 1, 3}
    @test Tuple(t) == (1.0, 2.0, 3.0)
end

@testset "repositioning" begin
    data1 = collect(1.0:10.0)
    a = KernelArrays.KS1Array{Tuple{3}, Float64, 1}(3, data1)
    @test a[1] == 3.0
    KernelArrays.idx!(a, 6)
    @test a[1] == 6.0
    @test a[3] == 8.0

    data2 = randn(5, 6)
    b = KS2Vector{3}(2, 3, data2)
    @test b[1] == data2[2, 3]
    KernelArrays.row!(b, 4)
    @test b[1] == data2[4, 3]

    KernelArrays.col!(b, 1)
    @test b[1] == data2[4, 1]
end
