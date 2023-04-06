# generate a strucutre containing all the information necessary for filtering based on light magnitude measurements
function lightMagFilteringProbGenerator(;x0 = zeros(6), P0 = zeros(6,6), x0true = [0;0;0;1;0;0;0], xf0 = x0true, Q = zeros(7,7), R = nothing, unitConversionFactor = 1.0, dt = .1, tf = 1.0, measInt = 1,  meas_t_init = 0, measTimeFunc = nothing, includeNoiseOnMeasurments = true, integratorSteps = 1, simTimeFac = 5, underweight = 1.2, type = :MUKF, scenParams = nothing, objParams = nothing, multiSpectral = false, vectorized = false)

    (sat,_,scen) = customScenarioGenerator(scenParams = scenParams, objParams = objParams, multiSpectral = multiSpectral, vectorized = vectorized)

    binNo = length(sat.Rdiff[1]) # number of frequnecy bins for model

    dynamicsFunc = (t,x) -> attDyn(t,x,sat.J,inv(sat.J),[0;0;0])
    measModel = (x) -> _Fobs(x[1:4], sat.nvecs, sat.uvecs, sat.vvecs,
    sat.Areas, sat.nu, sat.nv, sat.Rdiff, sat.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, qRotate, unitConversionFactor)


    measDim = scen.obsNo*binNo
    if isnothing(R)
        R = zeros(measDim,measDim)
    elseif typeof(R) <: Matrix
        # do nothing
    elseif size(R) == ()
        R = (unitConversionFactor * R)^2*diagm(ones(measDim))
    elseif (!(size(R,1) == measDim) | !(size(R,2) == measDim))
        error("Measurment Noise matrix must be a square matrix of the same dimension as number of observers")
    else
        error("Please Provide a valid measurement Noise matrix")
    end
    
    return nlFilteringProblem(attFilterInit(x0, P0, x0true, xf0), Q, R, includeNoiseOnMeasurments, dynamicsFunc, measModel, dt, tf, measDim, measInt, meas_t_init, measTimeFunc, integratorSteps, simTimeFac, underweight, type)
end

function GM_LM_filteringProbGen(x0, P0; weights = nothing, x0true=[0; 0; 0; 1; 0; 0; 0], Q=zeros(7, 7), R=nothing, unitConversionFactor = 1.0, dt=0.1, tf=1.0, measInt=1, meas_t_init=0, measTimeFunc=nothing, includeNoiseOnMeasurments = true, integratorSteps=1, simTimeFac=5, underweight=1.2, type = :GM_MUKF, scenParams=nothing, objParams=nothing, multiSpectral=false, vectorized=false)

    (sat, _, scen) = customScenarioGenerator(scenParams=scenParams, objParams=objParams, multiSpectral=multiSpectral, vectorized=vectorized)

    binNo = length(sat.Rdiff[1]) # number of frequnecy bins for model

    dynamicsFunc = (t, x) -> attDyn(t, x, sat.J, inv(sat.J), [0; 0; 0])
    measModel = (x) -> unitConversionFactor * _Fobs(x[1:4], sat.nvecs, sat.uvecs, sat.vvecs,
        sat.Areas, sat.nu, sat.nv, sat.Rdiff, sat.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, qRotate)


    measDim = scen.obsNo * binNo
    if isnothing(R)
        R = zeros(measDim, measDim)
    elseif (size(R, 1) == measDim & size(R, 2) == measDim)
        # do nothing
    elseif size(R) == ()
        R = (unitConversionFactor * R)^2 * diagm(ones(measDim))
    elseif (!(size(R, 1) == measDim) | !(size(R, 2) == measDim))
        error("Measurment Noise matrix must be a square matrix of the same dimension as number of observers")
    else
        error("Please Provide a valid measurement Noise matrix")
    end

    if isnothing(weights)
        weights_init = weights
    else
        weights_init = Array{Float64,1}(undef,length(x0))
        weights_init .= 1 / length(x0)
    end

    init = GMfilterInit(x0, P0, weights_init, x0true)

    return nlFilteringProblem(init, Q, R, includeNoiseOnMeasurments, dynamicsFunc, measModel, dt, tf, measDim, measInt, meas_t_init, measTimeFunc, integratorSteps, simTimeFac, underweight, type)

end

####### not updated
function AttFilteringProbGenerator(;x0 = zeros(6), P0 = zeros(6,6), x0true = [0;0;0;1;0;0;0], xf0 = x0true, Q = zeros(7,7), R = nothing, measInt = 1, measOffset = 0, tvec = [1:.1:10...], measTimes = nothing, scenParams = nothing, objParams = nothing, multiSpectral = false, vectorized = false)

    (sat,_,scen) = customScenarioGenerator(scenParams = scenParams, objParams = objParams, multiSpectral = multiSpectral, vectorized = vectorized)

    binNo = length(sat.Rdiff[1]) # number of frequnecy bins for model

    dynamicsFunc = (t,x) -> attDyn(t,x,sat.J,inv(sat.J),[0;0;0])
    measModel = (x) -> x[1:4]

    if isnothing(R)
        R = zeros(4,4)
    elseif (size(R,1) == 4 & size(R,2) == 4)
        # do nothing
    elseif (!(size(R,1) == 4) | !(size(R,2) == 4))
        error("Measurment Noise matrix must have same dimension as measurement")
    else
        error("Please Provide a valid measurement Noise matrix")
    end

    R_true = R

    if isnothing(measTimes)
        measTimes = Array{Bool,1}(undef,length(tvec))
        measTimes .= false
    elseif length(measTimes) !== length(tvec)
        error("length of time span and measurement time vector must match")
    else
        error("please provide valid measurement times")
    end

    return attFilteringProblem(x0, P0, x0true, xf0, Q, R, R_true, tvec, dynamicsFunc, measModel, scen.obsNo*binNo, measInt, measOffset, measTimes)
end

function linearFilteringTestProbGen()
    phi = [9.9985E-1 9.8510E-3;-2.9953E-2 9.7030E-1]
    gam = [4.9502E-5;9.8510E-3]
    H = [1 0;0 1;1 1]
    R = diagm([.01, .01, .01])
    dt = .01
    n = 1001
    tvec = [0:dt:dt*(n-1)...]
    q = 1
    Q = gam*q*gam'
    x0_true = [1, 1]
    P0 = (2/3)^2*eye(2)
    x0 = x0_true + gam*sqrt(q)*randn(1)[1]

    m = 3
    mTimes = Array{Bool,1}(undef,length(tvec))
    mTimes .= false
    m_int = 1
    m_off = 0
    isDiscrete = true
    underweightCoeff = 1


    return linearFilteringProblem(x0, P0, x0_true, Q, R, tvec, H, phi, m, m_int, m_off, mTimes, isDiscrete, underweightCoeff)
end