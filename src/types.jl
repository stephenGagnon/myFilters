struct UKFoptions
    a :: Float64
    f :: Float64
    alpha :: Float64
    beta :: Float64
    kappa :: Float64
    integrator :: Symbol
    integratorFunction :: Union{Function,Nothing}
end

function UKFoptions(;a = 1, f = 2*(a+1), alpha = 1, beta = 2, kappa = NaN, integrator = :RK4, integratorFunction = nothing)
    return UKFoptions(a,f,alpha, beta, kappa, integrator, integratorFunction)
end

struct attFilteringProblem
    initialErrorState :: Vector
    initialCovariance :: Matrix
    trueInitialState :: Vector
    estInitialState :: Vector
    processNoise :: Matrix
    measurementNoise :: Matrix
    timeVector :: Vector
    dynamics :: Function
    measurementModel :: Function
    measurementDimension :: Int64
    measurementInterval :: Float64
    measurementOffset :: Int64
    measurementTimes :: Vector
end

struct filteringProblem
    initialState :: Vector
    initialCovariance :: Matrix
    trueInitialState :: Vector
    processNoise :: Matrix
    measurementNoise :: Matrix
    timeVector :: Vector
    dynamics :: Function
    measurementModel :: Function
    measurementDimension :: Int64
    stateDimension :: Int64
    measurementInterval :: Float64
    measurementOffset :: Int64
    measurementTimes :: Vector
end
