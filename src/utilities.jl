# Kalman gain equation
function KalmanGain(P,H,R, underweightCoeff = 1)
    K = P*H'*inv(underweightCoeff*H*P*H' + R)
end

# compute sigma points from mean and covariance
function sigmaPoints(X, x, P, Psq, gamma)

    # try to take the cholesky decomposition of the covariance
    if isposdef(P)
        Psq[:] = cholesky(P).U
        exitflag = false
    elseif all(diag(P) .> 0)
        try
            Psq[:] = cholesky(Hermitian(P + diagm(max(diag(P)...) * 8 * ones(6)))).U
            exitflag = false
        catch
            exitflag = true
        end
    else
        exitflag = true
    end

    # generate sigma points
    X[1] = x
    for i = 1:6
        X[i+1] = x + gamma * Psq[:, i]
        X[i+7] = x - gamma * Psq[:, i]
    end

    return X, exitflag
end

# multivariate gaussian PDF
function pdf_gaussian(x, m ,P)
    temp = det(2 * pi * P)^(-1 / 2) * exp(-(1 / 2) * (x - m)' * inv(P) * (x - m))
    if isnan(temp)
       error("Covariance is singular")
    end
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

# 4th order constant time- step runge cutta integrator
function RK4(f, tvec, x0)

    # fixed time step 4th order runge kutta integrator

    # f is a discrete dynamics function that takes two arguments, a state value and a time step and propagates the state forward by the time step
    # tvec is a vector of time values. It should have a uniform time step.
    # x0 is the initial state

    # time step
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
    f1 = f(t, x0)
    f2 = f(t + dt/2, x0+.5*f1)
    f3 = f(t + dt/2, x0+.5*f2)
    f4 = f(t + dt, x0+f3)
    return x0 + dt/6*(f1+2*f2+2*f3+f4)
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
    for i = 1:length(a)
        for j = 1:length(b)
            out[i,j] += a[i]*b[j]
        end
    end
    return out
end

# compute sum of outer products of the collumns of two matricies
function outerProductSum(a, b)

    out = outerProduct(a[1],b[1])
    for i = 2:length(a)
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
    timeVectorSim = [0:fp.timeStep/fp.simulationTimeFactor:fp.finalTime...]

    # integrate the state dynamics over the time period specified
    (xhist_full,_) = integrator(fp.dynamics, timeVectorSim, fp.initial.trueState)
    # extract the states associated with the true times, and create the associated time vector
    xhist = xhist_full[1:fp.simulationTimeFactor:end]
    timeVector = timeVectorSim[1:fp.simulationTimeFactor:end]
    # initialize an array to hold measurements
    if fp.measurementDimension > 1
        yvec = Array{Array{Float64,1},1}(undef, length(timeVector))
        yvec[:] .= [zeros(fp.measurementDimension)]
    elseif fp.measurementDimension == 1
        yvec = Array{Float64,1}(undef, length(timeVector))
        yvec .= 0
    end

    # generate array which contains booleans indicating if a measurement is generated at each time
    measurementTimes = Array{Bool,1}(undef,length(timeVector))
    customTimes = !isnothing(fp.measurementTimeFunc)

    # create multi variable normal distribution from measurement noise parameters
    d = MvNormal(zeros(fp.measurementDimension),fp.measurementNoise)

    # loop through time vector and generate measurements
    for i = 1:length(timeVector)

        # if the user specified a funciton that determines if a measurement should be generated
        if customTimes
            # generate measurements at appropriate times
            if fp.measurementTimeFunc(timeVector[i])
                yvec[i] = fp.measurementModel(xhist[i]) + rand(d,1)[:]
                measurementTimes[i] = true
            else
                measurementTimes[i] = false
            end
        else
            # otherwise generate measurements at the specified intervales after the specified start time
            if timeVector[i] > fp.initialMeasurementTime
                if mod(i,fp.measurementInterval) == 0
                    yvec[i] = fp.measurementModel(xhist[i]) + rand(d,1)[:]
                    measurementTimes[i] = true
                else
                    measurementTimes[i] = false
                end
            else
                measurementTimes[i] = false
            end
        end

    end
    return xhist, yvec, timeVector, measurementTimes
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
function attFiltErrPlot(results :: filteringResults)

    t = results.time
    xError = similar(results.stateTrue, 6, length(t))
    xError[1:3, :] = attitudeErrors(results.stateTrue[1:4, :], results.stateEstimate[1:4, :])
    xError[4:6, :] = results.stateTrue[5:7, :] .- results.stateEstimate[5:7, :]

    sig = similar(xError)
    mult = [6; 6; 6; 3; 3; 3]
    for j = 1:length(t)
        sig[:, j] = sqrt.(diag(results.covariance[j])) .* mult
    end

    t_array = Array{Array{Float64,1},2}(undef, 2, 3)
    t_array .= [t]

    x_array = Array{Array{Array{Float64,1},1},2}(undef, 2, 3)
    linespec = Array{Array{Array{Any,1},1},2}(undef, 2, 3)
    strings = [["k", "linewidth", 2], ["--r"], ["--r"]]
    for i = 1:2
        for j = 1:3
            temp = [xError[(i-1)*3+j, :], sig[(i-1)*3+j, :], -sig[(i-1)*3+j, :]]
            x_array[i, j] = temp
            linespec[i, j] = strings
        end
    end

    subplot_MATLAB(t_array, x_array, linespec)

    # pygui(true)
    # figure()
    # for i = 1:size(xError,1)
    #   subplot(2,3,i)
    #   plot(t,xError[i,:])
    #   plot(t,sig[i,:],"--r",t,-sig[i,:],"--r")
    # end
end
