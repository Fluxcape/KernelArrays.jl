using Documenter
using KernelArrays

DocMeta.setdocmeta!(KernelArrays, :DocTestSetup, :(using KernelArrays); recursive=true)

makedocs(;
    modules=[KernelArrays],
    checkdocs=:exports,
    authors="chenyubao and contributors",
    repo="https://github.com/Fluxcape/KernelArrays.jl/blob/{commit}{path}#{line}",
    sitename="KernelArrays.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Fluxcape.github.io/KernelArrays.jl",
        repolink="https://github.com/Fluxcape/KernelArrays.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/Fluxcape/KernelArrays.jl.git",
    devbranch="main",
)
