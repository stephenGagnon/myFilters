module myFilters
using MATLAB
using LinearAlgebra
using Statistics
using attitudeFunctions
using lightCurveModeling
using Distributions
using Infiltrator
using PyPlot
using ForwardDiff
using MATLABfunctions

include("types.jl")
include("utilities.jl")
include("filters.jl")
include("filterUpdateFunctions.jl")
include("filteringProblemGenerators.jl")

export KF, EKF, MEKF, MUKF, GM_MUKF, RK4, _RK4, UKFoptions, EKFoptions, nlFilteringProblem, linearFilteringProblem, filteringProblem, attFilterInit, nlFilterInit, filteringResults, simulate, attFiltErrPlot, lightMagFilteringProbGenerator, AttFilteringProbGenerator, linearFilteringTestProbGen, GM_LM_filteringProbGen

end # module
