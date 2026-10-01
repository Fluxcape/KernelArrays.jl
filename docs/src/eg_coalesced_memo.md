# Coalesced Memory Access on GPU

In GPU programming, coalesced memory access is a technique that allows multiple threads to access contiguous memory locations in a single memory transaction. This can significantly improve performance by reducing the number of memory transactions and increasing memory bandwidth utilization.

To vividly explain coalesced memory access, we introduce 2 memory access patterns: Structure of Arrays (**SoA**) and Array of Structures (**AoS**).

Let's consider a 2D particle contains features of:

- position $X = (x, y)$
- velocity $V = (u, v)$
- mass $m$
- strain $s = \sigma_{ij}$ (a $2 \times 2$ tensor, i.e. 4 components $\sigma_{11}, \sigma_{12}, \sigma_{21}, \sigma_{22}$)

To store the data of $N$ particles, we can use either **SoA** or **AoS** layout. It's natural to store the data into a 2D array. In **AoS** layout, all features of one particle are stored contiguously. Since arrays in Julia are column-major, the data will look like:

```txt
x1 x2 x3 ...
y1 y2 y3 ...
u1 u2 u3 ...
v1 v2 v3 ...
m1 m2 m3 ...
σ11 σ11 σ11 ...
σ12 σ12 σ12 ...
σ21 σ21 σ21 ...
σ22 σ22 σ22 ...
```

In **SoA** layout, one feature's values across all particles are stored contiguously, so the data will look like:

```txt
x1 y1 u1 v1 m1 σ11 σ12 σ21 σ22 ...
x2 y2 u2 v2 m2 σ11 σ12 σ21 σ22 ...
......
```

On GPU, **SoA** layout is preferred because it allows threads to access contiguous memory locations, resulting in coalesced memory access. `KS2Array` is designed to support **SoA** layout, which is more efficient for GPU computations.

We can use `KS2Vector` to view a particle's vector features, and `KS2SquareMatrix` to view a particle's strain tensor. The following example demonstrates how to use `KS2Vector` and `KS2SquareMatrix` to access particle data in a coalesced manner.

```julia
using KernelAbstractions
using KernelArrays

const FeatureNT = (
    x=1,
    u=3,
    m=5,
    s=6
)

@kernel function particle_action!(particles, dt)
    idx = @index(Global)
    x = KS2Vector{2}(idx, FeatureNT.x, particles)
    u = KS2Vector{2}(idx, FeatureNT.u, particles)
    mass = particles[idx, FeatureNT.m]
    sigma = KS2SquareMatrix{2}(idx, FeatureNT.s, particles)

    x .+= u .* dt # move the particles
    # other operations on mass and sigma...
end

particles = randn(100, 9)
dt = 0.01
backend = KernelAbstractions.CPU()   # or CUDA / ROC / oneAPI / Metal
particle_action!(backend, 128)(particles, dt, ndrange=(size(particles, 1),))
KernelAbstractions.synchronize(backend)
```
