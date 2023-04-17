# function to simulate a dynamic system and generate measurements
function simulate(fp :: nlFilteringProblem, integrator :: Function)

    # create time vector for simulation
    # assume start time is 0 and use provided final time
    # time step is the time step divided by the simulationTimeFactor. The factor should be an integer equal to or greater than one. Values greater than one allow for multiple integration steps between recorded state values for higher fidelity
    timeVector = [0:fp.timeStep:fp.finalTime...]

    # integrate the state dynamics over the time period specified
    (xhist,_) = integrator(fp.dynamics, timeVector, fp.initial.trueState)
    # extract the states associated with the true times, and create the associated time vector
    # xhist = xhist_full[1:fp.simulationTimeFactor:end]
    # timeVector = timeVectorSim[1:fp.simulationTimeFactor:end]
    # initialize an array to hold measurements
    yvec = Array{Array{Float64,1},1}(undef, length(timeVector))
    # yvec[:] .= [zeros(fp.measurementDimension)]



    # generate array which contains booleans indicating if a measurement is generated at each time
    measurementTimes = Array{Bool,1}(undef,length(timeVector))
    customTimes = !isnothing(fp.measurementTimeFunc)

    # create multi variable normal distribution from measurement noise parameters
    d = MvNormal(zeros(fp.measurementDimension),fp.measurementNoise)

    # loop through time vector and generate measurements
    for i = eachindex(timeVector)

        # if the user specified a funciton that determines if a measurement should be generated
        if customTimes
            if fp.includeNoiseOnMeasurments
                noise = rand(d,1)[:]
            else
                noise = zeros(fp.measurementDimension)
            end
            yvec[i] = fp.measurementModel(xhist[i]) + noise
            # generate measurements at appropriate times
            if fp.measurementTimeFunc(timeVector[i])
                measurementTimes[i] = true
            else
                measurementTimes[i] = false
            end
        else
            if fp.includeNoiseOnMeasurments
                noise = rand(d,1)[:]
            else
                noise = zeros(fp.measurementDimension)
            end
            yvec[i] = fp.measurementModel(xhist[i]) + noise
            
            # otherwise generate measurements at the specified intervales after the specified start time
            if timeVector[i] >= fp.initialMeasurementTime
                if mod(i,fp.measurementInterval) == 0
                    measurementTimes[i] = true
                else
                    measurementTimes[i] = false
                end
            else
                measurementTimes[i] = false
            end
        end

    end

    xHist_m = Array{Float64,2}(undef,length(xhist[1]),length(xhist))
    for i = eachindex(xhist)
        xHist_m[:,i] = xhist[i]
    end
    return xHist_m, yvec, timeVector, measurementTimes
end

function simulate(fp :: linearFilteringProblem, phi :: Matrix)

    # initialize true state history
    x = Array{Array{Float64,1},1}(undef, length(fp.timeVector))

    # intialize measurement history
    y = Array{Array{Float64,1},1}(undef, length(fp.timeVector))

    measurementTimes = similar(fp.measurementTimes)
    customMeasurementTimes = false

    if any(fp.measurementTimes)
        customMeasurementTimes = true
        measurementTimes = fp.measurementTimes
    end

    d = MvNormal(zeros(fp.measurementDimension),fp.measurementNoise)

    for i = 1:length(fp.timeVector)
        if i == 1
            x[i] = fp.trueInitialState
        else
            x[i] = phi*x[i-1]
        end

        if customMeasurementTimes
            if measurementTimes[i]
                y[i] = fp.measurementMatrix*x[i] + rand(d,1)[:]
            end
        elseif mod(i-2-fp.measurementOffset,fp.measurementInterval) == 0
            measurementTimes[i] = true
            y[i] = fp.measurementMatrix*x[i] + rand(d,1)[:]
        else
            measurementTimes[i] = false
        end
    end

    return x, y, measurementTimes
end

function attitudeFilteringSimulation(Prob :: nlFilteringProblem, options::UKFoptions)

    # define integration function used to propagate the state dynamics
    if options.integrator == :RK4
        integrator = RK4
    elseif typeof(options.integrator) == Function
        integrator = options.integrator
    else
        error("unsupported integrator")
    end

    # compute the measurements over the simulation period
    (xTrue, measurements, timeVector, measurementTimes) = simulate(Prob, integrator)

    results = MUKF(Prob, options, timeVector, measurements, measurementTimes, xTrue)

    residuals = Array{eltype(results.residuals[1]),2}(undef, length(results.residuals[1]), length(results.residuals))
    for i = eachindex(results.residuals)
        residuals[:,i] = results.residuals[i]
    end

    return filteringSimulationResults(results.stateEstimate, results.covariance, residuals, results.residualCovariance, xTrue, timeVector, measurements, measurementTimes)


end

function GMattitudeFilteringSimulation(Prob, options)
        # define integration function used to propagate the state dynamics
        if options.integrator == :RK4
            integrator = RK4
        elseif typeof(options.integrator) == Function
            integrator = options.integrator
        else
            error("unsupported integrator")
        end
    
        # compute the measurements over the simulation period
        (xTrue, measurements, timeVector, measurementTimes) = simulate(Prob, integrator)

        results, data = GM_MUKF(Prob, options, timeVector, measurements, measurementTimes, xTrue)
    
        # residuals = Array{eltype(results.residuals[1]),2}(undef, length(results.residuals[1]), length(results.residuals))
        # for i = eachindex(results.residuals)
        #     residuals[:,i] = results.residuals[:,i]
        # end
    
        return filteringSimulationResults(results.stateEstimate, results.covariance, results.residuals, results.residualCovariance, xTrue, timeVector, measurements, measurementTimes), data
    
end