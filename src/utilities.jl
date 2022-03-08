function RK4(f,tvec,x0,symbol :: Symbol)

    # fixed time step 4th order runge kutta integrator

    # f is a discrete dynamics function that takes two arguments, a state value and a time step and propagates the state forward by the time step
    # tvec is a vector of time values. It should have a uniform time step.
    # x0 is the initial state

    # time step
    dt = tvec[2]-tvec[1];

    # intialize state history
    xt = zeros(length(x0),length(tvec))
    xt[:,1] = x0

    # integrate
    for i = 1:length(tvec)-1
        # t = tvec[i]
        f1 = f(tvec[i], view(xt,:,i))
        f2 = f(tvec[i] + dt/2, view(xt,:,i)+.5*f1)
        f3 = f(tvec[i] + dt/2, view(xt,:,i)+.5*f2)
        f4 = f(tvec[i] + dt, view(xt,:,i)+f3)
        xt[:,i+1] = view(xt,:,i) + 1/6*(f1+2*f2+2*f3+f4);
    end

    return xt, tvec
end

function RK4(f,tvec,x0)

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
        # t = tvec[i]
        f1 = f(tvec[i], xt[i])
        f2 = f(tvec[i] + dt/2, xt[i]+.5*f1)
        f3 = f(tvec[i] + dt/2, xt[i]+.5*f2)
        f4 = f(tvec[i] + dt, xt[i]+f3)
        xt[i+1] = xt[i] + 1/6*(f1+2*f2+2*f3+f4);
    end

    return xt, tvec
end

function outerProduct(a, b)
    out = Array{Float64,2}(undef,length(a),length(b))
    out .= 0
    return outerProduct!(a,b,out)
end

function outerProduct(a :: Vector{BigFloat}, b :: Vector{BigFloat})
    out = Array{BigFloat,2}(undef,length(a),length(b))
    out .= 0
    return outerProduct!(a,b,out)
end

function outerProduct(a :: Vector{BigFloat}, b)
    out = Array{BigFloat,2}(undef,length(a),length(b))
    out .= 0
    return outerProduct!(a,b,out)
end

function outerProduct(a, b :: Vector{BigFloat})
    out = Array{BigFloat,2}(undef,length(a),length(b))
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

function simulate(fp :: attFilteringProblem, integrator :: Function)

    (xhist,_) = integrator(fp.dynamics,fp.timeVector,fp.trueInitialState)

    yvec = Array{Array{Float64,1},1}(undef, length(fp.timeVector))
    yvec[:] .= [zeros(fp.measurementDimension)]

    measurementTimes = similar(fp.measurementTimes)
    customMeasurementTimes = false

    if any(fp.measurementTimes)
        customMeasurementTimes = true
        measurementTimes = fp.measurementTimes
    end

    d = MvNormal(zeros(fp.measurementDimension),fp.measurementNoise)

    for i = 1:length(fp.timeVector)
        if customMeasurementTimes
            if measurementTimes[i]
                yvec[i] = fp.measurementModel(xhist[i]) + rand(d,1)[:]
            end
        elseif mod(i-2-fp.measurementOffset,fp.measurementInterval) == 0
            measurementTimes[i] = true
            yvec[i] = fp.measurementModel(xhist[i]) + rand(d,1)[:]
        else
            measurementTimes[i] = false
        end
    end
    return xhist,yvec,measurementTimes
end

function attFiltErrPlot(results :: filteringResults)

    t = results.time
    xError = similar(results.stateTrue,6,length(t))
    xError[1:3,:] = attitudeErrors(results.stateTrue[1:4,:],results.stateEstimate[1:4,:])
    xError[4:6,:] = results.stateTrue[5:7,:] .- results.stateEstimate[5:7,:]

    sig = similar(xError)
    mult = [6;6;6;3;3;3]
    for j = 1:length(t)
        sig[:,j] = sqrt.(diag(results.covariance[j])).*mult
    end

    pygui(true)
    figure()
    for i = 1:size(xError,1)
      subplot(2,3,i)
      plot(t,xError[i,:])
      plot(t,sig[i,:],"--r",t,-sig[i,:],"--r")
    end
end
