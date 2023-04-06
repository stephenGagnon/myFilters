struct UKFoptions
    alpha::Float64
    beta::Float64
    kappa::Float64
    integrator::Union{Symbol,Function}
end

function UKFoptions(;alpha=1, beta=2, kappa=NaN, integrator=:RK4)
    return UKFoptions(alpha, beta, kappa, integrator)
end

struct EKFoptions
    integrator::Union{Symbol,Function}
end

struct attFilterInit
    errorState::Vector
    covariance::Matrix
    trueState::Vector
    estimateState::Vector
end

struct nlFilterInit
    state::Vector
    covariance::Matrix
    trueState::Vector
end

struct GMfilterInit
    means::Vector{V} where {T<:Number,V<:Vector{T}}
    covariances::Vector{P} where {T<:Number, P<:Array{T,2}}
    weights :: Vector
    trueState::Vector
end

struct linearFilteringProblem
    initialState::Vector
    initialCovariance::Matrix
    trueInitialState::Vector
    processNoise::Union{Matrix, Number}
    measurementNoise::Union{Matrix, Number}
    measurementMatrix::Matrix
    stateMatrix::Matrix
    timeVector::Vector
    measurementDimension::Int64
    measurementInterval::Float64
    measurementOffset::Int64
    measurementTimes::Vector
    isDiscrete::Bool
    underweightCoeff::Float64
end

struct nlFilteringProblem
    initial::Union{attFilterInit,nlFilterInit,GMfilterInit}
    processNoise::Union{Matrix, Number}
    measurementNoise::Union{Matrix, Number}
    includeNoiseOnMeasurments :: Bool
    dynamics::Function
    measurementModel::Function
    timeStep::Float64
    finalTime::Float64
    measurementDimension::Int64
    measurementInterval::Float64
    initialMeasurementTime::Float64
    measurementTimeFunc::Union{Function,Nothing}
    integratorSteps::Float64
    simulationTimeFactor::Int64
    underweightCoeff::Float64
    filterType::Symbol
end

filteringProblem = Union{linearFilteringProblem,nlFilteringProblem}

struct filteringResults
    stateEstimate
    covariance
    residuals
    residualCovariance
end

struct filteringSimulationResults
    stateEstimate
    covariance
    residuals
    residualCovariance
    stateTrue
    time :: Vector
    measurements
    measurementTimes
end


# struct attFilteringProblem
#     initial :: attFilterInitialization
#     # initialErrorState :: Vector
#     # initialCovariance :: Matrix
#     # trueInitialState :: Vector
#     # estInitialState :: Vector
#     processNoise :: Matrix
#     measurementNoise :: Matrix
#     dynamics :: Function
#     measurementModel :: Function
#     timeStep :: Float64
#     finalTime :: Float64
#     measurementDimension :: Int64
#     measurementInterval :: Float64
#     initialMeasurementTime :: Float64
#     measurementTimeFunc :: Union{Function, Nothing}
#     integratorSteps :: Float64
#     simulationTimeFactor :: Int64
#     underweightCoeff :: Float64
#     filterType :: Symbol
# end
