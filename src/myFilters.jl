module myFilters
using MATLAB
using LinearAlgebra
using Statistics
using attitudeFunctions
using lightCurveModeling
using Distributions
using Infiltrator
using PyPlot

include("types.jl")
include("utilities.jl")
include("filterFunctions.jl")

export MUKF, RK4, UKFoptions, attFilteringProblem, filteringProblem, filteringResults, simulate, attFiltErrPlot, lightMagFilteringProbGenerator, AttFilteringProbGenerator

end # module
