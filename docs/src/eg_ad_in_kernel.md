# Autograd in Kernel

Autograd is a technique used in machine learning and deep learning to compute gradients of functions with respect to their inputs.

In the Julia ecosystem, [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) is a popular package that provides automatic differentiation capabilities. It allows users to compute derivatives of functions efficiently and accurately.

Inside the GPU kernel, memory allocation is limited. After our trials, we found only [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) can work properly in the kernel. Here's a simple example to demonstrate how to use [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) in a kernel to compute the Jacobian of a function.

```julia
using ForwardDiff
using oneAPI
using KernelAbstractions
using KernelArrays

@inline function f(x)
    u = similar(x)
    u[1] = x[1] + x[2]
    u[2] = x[1] * x[2]
    return u
end

@kernel function kernel_g!(arr)
    idx = @index(Global)
    x = KS2Vector{2}(idx, 1, arr)
    jac = ForwardDiff.jacobian(f, x)
    res = KS2SquareMatrix{2}(idx, 3, arr)
    res .= jac
end

arr = randn(Float32, 3, 6) |> oneArray
arr
# 3×6 oneArray{Float32, 2, oneAPI.oneL0.DeviceBuffer}:
#  -1.75271   -1.50263   -0.707247  -0.526534  -0.694657  0.11173
#  -0.651412   0.269179  -1.07794    0.70608   -0.461094  1.35437
#   0.305119   0.472155  -0.607594  -0.629629  -0.286176  0.168033
kernel_g!(oneAPIBackend(), 3)(arr, ndrange=(3,))
arr
# 3×6 oneArray{Float32, 2, oneAPI.oneL0.DeviceBuffer}:
#  -1.75271   -1.50263   1.0  -1.50263   1.0  -1.75271
#  -0.651412   0.269179  1.0   0.269179  1.0  -0.651412
#   0.305119   0.472155  1.0   0.472155  1.0   0.305119
```

> You can even run AD inside GPU kernel! Although currently I have not figured out what's the purpose of this feature ... LOL, but it's interesting, isn't it?