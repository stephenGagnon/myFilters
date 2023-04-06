# Kalman gain equation
function KalmanGain(P,H,R, underweightCoeff = 1)
    K = P*H'*inv(underweightCoeff*H*P*H' + R)
end

# compute sigma points from mean and covariance
function sigmaPoints(X, x, P, Psq, gamma)

    # try to take the cholesky decomposition of the covariance
    if isposdef(P)

        Psq[:] = cholesky(P).U'
        exitflag = 0

    elseif any(isnan.(P))

        exitflag = 4
        return X, exitflag

    elseif P != P'

        exitflag = 1
        return X, exitflag

    else
        # [v, d]=eigvals(pcov);
        # P_old = copy(P)
        v = eigvecs(P)
        d = eigvals(P)
        #set any eigenvalues that are not positive to a small positive number
        for ii=eachindex(d)
            if d[ii]<=1e-14
                if abs(d[ii]/maximum(d))>1e-3 #large negative eigenvalues
                    exitflag = 2
                    return X, exitflag
                end
                d[ii] = 1e-14;
            end
        end

        # Reconstruct the covariance matrix, which should now be positive definite
        # @infiltrate
        P = real(v*diagm(d)*v');
        P = (P + P')./2
        # perform the cholesky decomposition and record that a correction occured (errorflag 3)
        Psq[:] = cholesky(P).U'
        exitflag = 3
    end

    # generate sigma points
    X[1] = x
    for i = axes(Psq,2)
        X[i+1] = x - (gamma * Psq[:, i])
        X[i+7] = x + (gamma * Psq[:, i])
    end

    return X, exitflag
end

# multivariate gaussian PDF
function pdf_gaussian(x, m ,P)
    temp = det(2 * pi * P)^(-1 / 2) * exp(-(1 / 2) * (x - m)' * inv(P) * (x - m))
    # if isnan(temp)
    #    error("Covariance is singular")
    # end
    return temp
end

# method of moments for mixtures of attitude distributions
function MoM_att(w, x, P)
    # initialize mean state
    x_m = similar(x[1])
    # initialize mean covariance
    P_m = zeros(size(P[1]))

    # compute mean angular velocity as simple weighted sum
    ω_mean = zeros(3)
    for i = 1:lastindex(x)
        ω_mean = ω_mean + w[i] * x[i][5:7]
    end
    x_m[5:7] = ω_mean

    #compute mean quaternion using quaternion averaging
    q_mean = quaternionAverage([v[1:4] for v in x], w)

    x_m[1:4] = q_mean

    d = zeros(6)
    for i = 1:lastindex(x)
        d[4:6] = x[i][5:7] - ω_mean
        d[1:3] = 2 * (qprod(x[i][1:4], qinv(q_mean)))[1:3]
        P_m += w[i] * (P[i] + d * d')
    end

    return x_m, P_m
end

# method of moments for mixtures of attitude distributions
function MoM(w, x, P)

    # compute mean state as weighted sum
    x_mean = sum(w.*x)

    P_mean = zeros(size(P[1]))
    for i = eachindex(P)
        P_mean += w[i]*(P[i] + (x[i]-x_mean)*(x[i]-x_mean)')
    end

    return x_mean, P_mean
end

# 4th order constant time- step runge cutta integrator
function RK4(f, tvec, x0)

    # fixed time step 4th order runge kutta integrator

    # f is a discrete dynamics function that takes two arguments, a state value and a time step and propagates the state forward by the time step
    # tvec is a vector of time values. It should have a uniform time step.
    # x0 is the initial state

    # time step (assume constant time step)
    dt = tvec[2]-tvec[1];

    # intialize state history
    # xt = zeros(length(x0),length(tvec))
    xt = Array{Array{Float64,1},1}(undef, length(tvec))
    xt[1] = x0

    # integrate
    for i = 1:length(tvec)-1
        xt[i+1] = _RK4(f, tvec[i], dt, xt[i])
    end

    return xt, tvec
end

# single iteration of RK4
function _RK4(f, t, dt, x0)
    f1 = dt*f(t, x0)
    f2 = dt*f(t + dt/2, x0 + .5*f1)
    f3 = dt*f(t + dt/2, x0 + .5*f2)
    f4 = dt*f(t + dt, x0 + f3)
    return x0 + 1/6*(f1 + 2*f2 + 2*f3 + f4)
end

# compute outer product of two vectors
function outerProduct(a :: AbstractArray{T,1}, b :: AbstractArray{T,1}) where T
    out = Array{T,2}(undef,length(a),length(b))
    out .= 0
    return outerProduct!(a,b,out)
end

function outerProduct(a)
    return outerProduct(a,a)
end

function outerProduct!(a,b,out)
    for i = eachindex(a)
        for j = eachindex(b)
            out[i,j] += a[i]*b[j]
        end
    end
    return out
end

# compute sum of outer products of the collumns of two matricies
function outerProductSum(a, b)

    out = outerProduct(a[1],b[1])
    for i = 2:lastindex(a)
        outerProduct!(a[i],b[i],out)
    end
    return out
end

function outerProductSum(a)
    return outerProductSum(a,a)
end

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

# generate plots of errors and 3 sigma limits from the results of filtering
function attFiltErrPlot(results :: filteringSimulationResults)
    tf = size(results.stateEstimate,2)
    _attFiltErrPlot(results.time[1:tf], results.stateTrue[:,1:tf], results.stateEstimate, results.covariance)        
end

function measurementResidualPlot(results :: filteringSimulationResults, n_cols = 1)

    error = results.residuals
    covariance = results.residualCovariance

    t = results.time[1:length(covariance)]

    measurementDimension = size(error,1)

    sig = similar(error)
    mult = 3 .* ones(measurementDimension)
    for j = eachindex(covariance)
        sig[:, j] = sqrt.(diag(covariance[j])) .* mult
    end

    if mod(measurementDimension,n_cols) == 0
        n_rows = Int(measurementDimension/n_cols)
    else
        n_rows = 1
        n_cols = measurementDimension
        print("Cannot evenly divide measurement dimension by provided number of columns \n")
        print("Default to one column of plots")
    end

    t_array = Array{Array{Float64,1},2}(undef, n_rows, n_cols)
    t_array .= [t]

    x_array = Array{Array{Array{Float64,1},1},2}(undef, n_rows, n_cols)
    linespec = Array{Array{Array{Any,1},1},2}(undef, n_rows, n_cols)
    strings = [["k", "linewidth", 1.2], ["--r"], ["--r"]]

    xlabel_strings = Array{String,2}(undef, n_rows, n_cols)
    xlabel_strings .= "time (s)"
    ylabel_strings = Array{String,2}(undef, n_rows, n_cols)
    ylabel_strings .= "Residual Error pW/m^2"

    for i = 1:n_rows
        for j = 1:n_cols
            temp = [error[(i-1)*n_cols+j, :], sig[(i-1)*n_cols+j, :], -sig[(i-1)*n_cols+j, :]]
            x_array[i, j] = temp
            linespec[i, j] = strings
        end
    end

    subplot_MATLAB(t_array, x_array, linespec, xlabel_string = xlabel_strings, ylabel_string = ylabel_strings)

end

function _attFiltErrPlot(t, stateTrue, stateEstimate, covariance)

    rad2deg = 180/pi
    s2hr = 1/3600

    xError = similar(stateTrue, 6, length(t))
    xError[1:3, :] = rad2deg .* attitudeErrors(stateTrue[1:4, :], stateEstimate[1:4, :])
    xError[4:6, :] = rad2deg/s2hr .* (stateTrue[5:7, :] .- stateEstimate[5:7, :])

    sig = similar(xError)
    mult = [rad2deg * 6 .* ones(3); rad2deg/s2hr * 3 .* ones(3)]
    for j = 1:length(t)
        sig[:, j] = sqrt.(diag(covariance[j])) .* mult
    end

    t_array = Array{Array{Float64,1},2}(undef, 2, 3)
    t_array .= [t]

    x_array = Array{Array{Array{Float64,1},1},2}(undef, 2, 3)
    linespec = Array{Array{Array{Any,1},1},2}(undef, 2, 3)
    strings = [["k", "linewidth", 1.2], ["--r"], ["--r"]]

    xlabel_strings = Array{String,2}(undef, 2, 3)
    xlabel_strings .= "time (s)"
    ylabel_strings = Array{String,2}(undef, 2, 3)


    for i = 1:2
        for j = 1:3
            temp = [xError[(i-1)*3+j, :], sig[(i-1)*3+j, :], -sig[(i-1)*3+j, :]]
            x_array[i, j] = temp
            linespec[i, j] = strings
            if i == 1
                ylabel_strings[i, j] = "Attitude Error (degrees)"
            else
                ylabel_strings[i, j] = "Angular Velocity Error (degrees/hr)"
            end
        end
    end


    subplot_MATLAB(t_array, x_array, linespec, xlabel_string = xlabel_strings, ylabel_string = ylabel_strings)
end

function attFilterPerformanceEval(results, averagingWindowSize, attThreshold=5, avThreshold=0.01)

    tf = size(stateEstimate,2)
    time = results.time[1:tf]
    xTrue = results.stateTrue[:,1:tf]
    xEst = results.stateEstimate

    mult = [6; 6; 6; 3; 3; 3]
    sig3Violation = 0
    attErrSum = 0
    avErrSum = 0
    for i = eachindex(time)
        attErr = attitudeErrors(xTrue[1:4,i], xEst[1:4,i])
        avErr = xTrue[end-2:end,i] - xEst[end-2:end,i]
        fullErr = vcat(attErr,avErr)

        if i > (tf-averagingWindowSize)
            attErrSum += norm(attErr)
            avErrSum += norm(avErr)
        end

        sig = sqrt.(diag(results.covariance[i])) .* mult
        for j = 1:6
            if abs(sig[j]) < abs(fullErr[j])
                sig3Violation += abs(fullErr[j]) - abs(sig[j])
            end
        end
    end

    attErr = attErrSum / (averagingWindowSize + 1)
    avErr = avErrSum / (averagingWindowSize + 1)

    conv = false
    if (attErr < attThreshold) & (avErr < avThreshold)
        conv = true
    end

    return conv, sig3Violation
end