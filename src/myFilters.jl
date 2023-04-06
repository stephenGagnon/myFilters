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
using MAT
# using CSV
# using Tables

include("types.jl")
include("utilities.jl")
include("filters.jl")
include("filterUpdateFunctions.jl")
include("filteringProblemGenerators.jl")
include("simulators.jl")

export KF, EKF, MEKF, MUKF, GM_MUKF, attitudeFilteringSimulation, RK4, _RK4, UKFoptions, EKFoptions, nlFilteringProblem, linearFilteringProblem, filteringProblem, attFilterInit, nlFilterInit, filteringResults, simulate, attFiltErrPlot, lightMagFilteringProbGenerator, AttFilteringProbGenerator, linearFilteringTestProbGen, GM_LM_filteringProbGen, attFilterPerformanceEval

end # module
