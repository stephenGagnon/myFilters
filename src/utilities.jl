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


# function outerProductSum(a :: Vector{Vector{BigFloat}}, b :: Vector{Vector{BigFloat}})
#
#     out = Array{BigFloat,2}(undef,length(a[1]),length(b[1]))
#     for i = 1:size(out,1)
#         for j = 1:size(out,2)
#             out[i,j] = BigFloat(0)
#         end
#     end
#
#     return outerProductSum!(a,b,out)
# end
#
# function outerProductSum(a :: Vector{Vector{BigFloat}}, b)
#
#     out = Array{BigFloat,2}(undef,length(a[1]),length(b[1]))
#     for i = 1:size(out,1)
#         for j = 1:size(out,2)
#             out[i,j] = BigFloat(0)
#         end
#     end
#
#     return outerProductSum!(a,b,out)
# end
#
# function outerProductSum(a, b :: Vector{Vector{BigFloat}})
#
#     out = Array{BigFloat,2}(undef,length(a[1]),length(b[1]))
#     for i = 1:size(out,1)
#         for j = 1:size(out,2)
#             out[i,j] = BigFloat(0)
#         end
#     end
#
#     return outerProductSum!(a,b,out)
# end
#
# function outerProductSum(a :: Vector{Vector{Float64}}, b :: Vector{Vector{Float64}})
#     out = zeros(length(a[1]),length(b[1]))
#     return outerProductSum!(a,b,out)
# end
#
# function outerProductSum!(a, b, out)
#
#     for k = 1:length(a)
#         for i = 1:length(a[1])
#             for j = 1:length(b[1])
#                 out[i,j] += a[k][i]*b[k][j]
#             end
#         end
#     end
#     return out
# end
# function outerProductSum(a)
#
#     out = Array{typeof(a[1][1]),2}(undef,length(a[1]),length(a[1]))
#     for i = 1:size(out,1)
#         for j = 1:size(out,2)
#             out[i,j] = typeof(a[1][1])(0)
#         end
#     end
#
#     for k = 1:length(a)
#         for i = 1:length(a[1])
#             for j = 1:length(a[1])
#                 out[i,j] += a[k][i]*a[k][j]
#             end
#         end
#     end
#     return out
# end
