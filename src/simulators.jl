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

    results = MUKF(Prob, options, timeVector, integrator,  measurements, measurementTimes, xTrue)

    residuals = Array{eltype(results.residuals[1]),2}(undef, length(results.residuals[1]), length(results.residuals))
    for i = eachindex(results.residuals)
        residuals[:,i] = results.residuals[i]
    end

    return filteringSimulationResults(results.stateEstimate, results.covariance, residuals, results.residualCovariance, xTrue, timeVector, measurements, measurementTimes)


end