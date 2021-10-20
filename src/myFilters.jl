module myFilters

using LinearAlgebra
using Statistics
using attitudeFunctions
using Distributions
using Infiltrator

include("types.jl")
include("utilities.jl")
include("filterFunctions.jl")

export MUKF, RK4, outerProduct, UKFoptions, attFilteringProblem, filteringProblem, simulate, MUKFstep

function MUKF(model :: attFilteringProblem, options :: UKFoptions)

      # define integration function used to propagate the state dynamics
      if options.integrator == :RK4
            integrator = RK4
      elseif options.integrator == :custom
            integrator = options.integratorFunction
      else
            error("unsupported integrator")
      end

      # compute the measurements over the simulation period
      (_,measurements,measurementTimes) = simulate(model,integrator)

      # get length of simulation
      m = length(model.timeVector)

      ## compute UKF parameters based on used defined alpha and beta

      # deafult kappa value, specified by inputting a kappa value of NaN
      # user specified kappa values will override the default
      if isnan(options.kappa)
            #Minimizes mean square error up to 4th order, need to be careful with negative kappa.
            # kappa = 3-model.errorStateDimension
            kappa = -3
      end

      #scaling parameter
      # lambda = options.alpha^2*(model.errorStateDimension+kappa)-model.errorStateDimension
      lambda = options.alpha^2*(6+kappa)-6

      # Define the weights:
      # W0_mean = lambda/(model.errorStateDimension+lambda)
      W0_mean = lambda/(6+lambda)
      W0_cov = W0_mean+(1-options.alpha^2+options.beta)
      # Wi_mean = 1/(2*(model.errorStateDimension+lambda))
      Wi_mean = 1/(2*(6+lambda))
      Wi_cov = Wi_mean
      W = [W0_mean; W0_cov; Wi_mean; Wi_cov]

      # define gamma
      # gamma = sqrt(model.errorStateDimension+lambda)
      gamma = sqrt(6+lambda)

      # initalize state and covariance
      xvec = Array{Array{Float64,1},1}(undef,m)
      qvec = Array{Array{Float64,1},1}(undef,m)
      Pvec = Array{Array{Float64,2},1}(undef,m)
      xvec[1] = model.initialErrorState
      Pvec[1] = model.initialCovariance
      qvec[1] = model.estInitialState
      x = deepcopy(model.initialErrorState)
      P = deepcopy(model.initialCovariance)
      q = deepcopy(model.estInitialState[1:4])

      # main loop
      for i = 2:m

            t = [model.timeVector[i-1]; model.timeVector[i]]

            x,P,q,exitflag = MUKFstep(x, P, q, 6, t, measurements[i], measurementTimes[i], model.dynamics, model.measurementModel, model.processNoise, model.measurementNoise, options.a, options.f, gamma, W, integrator)

            # store state and covariance
            if exitflag
                  xvec[i] = deepcopy(x)
                  Pvec[i] = deepcopy(P)
                  qvec[i] = deepcopy(q)
            else
                  xvec[i] = deepcopy(x)
                  Pvec[i] = deepcopy(P)
                  qvec[i] = deepcopy(q)
                  xvec = xvec[1:i]
                  Pvec = Pvec[1:i]
                  qvec = qvec[1:i]
                  measurementsOut = measurements[1:i]
                  return xvec,qvec,Pvec,measurementsOut
            end
      end

      return xvec,qvec,Pvec,measurements
end

function MUKFstep(x, P, q, n, t, y, mtime, dynamics, measModel, Q, R, a, f, gamma, W, integrator)
      # Covariance Decomposition
      exitflag = true
      if isposdef(P)
            Psq = cholesky(P).U
      elseif isposdef(P+diagm(max(diag(P)...)*8*ones(6)))
            Psq = cholesky(P+diagm(max(diag(P)...)*8*ones(6))).U
      else
            @infiltrate
            exitflag = false
            return x, P, q, exitflag
      end

      # Error sigma points:
      # sig = hcat(-gamma*Psq, gamma*Psq)
      sig = vcat([-gamma*Psq[:,i] for i = 1:size(Psq,2)], [gamma*Psq[:,i] for i = 1:size(Psq,2)])
      x[1:3] = zeros(3)
      X = vcat([x], sig .+ [x])

      # @infiltrate
      # propogate
      X,x,P,q = MUKFprop(X,q,n,t,W,Q,dynamics,integrator,a,f)

      # @infiltrate

      # Update
      if mtime
            # @infiltrate
            x,P = MUKFupdate(X,x,P,q,n,W,R,Q,y,measModel,a,f)
      end

      # update quaternion
      # mrp error
      dp = x[1:3]
      # convert to error quaternions
      dq = p2q(dp,a,f)
      q = qprod(dq,q)
      q = q./norm(q)

      return x,P,q,exitflag
end

end # module
