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
      (xTrue,measurements,measurementTimes) = simulate(model,integrator)

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
      qvec[1] = model.estInitialState[1:4]
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
                  # xvec = xvec[1:i]
                  # Pvec = Pvec[1:i]
                  # qvec = qvec[1:i]

                  stateEstimate = Array{Float64,2}(undef,7,i)
                  stateTrue = Array{Float64,2}(undef,7,i)
                  for j = 1:i
                        stateEstimate[1:4,j] = qvec[j]
                        stateEstimate[5:7,j] = xvec[j][4:6]
                        stateTrue[:,j] = xTrue[j]
                  end

                  return filteringResults(stateEstimate, stateTrue, Pvec[1:i], measurements[1:i], measurementTimes[1:i], model.timeVector[1:i])
            end
      end


      stateEstimate = Array{Float64,2}(undef,7,m)
      stateTrue = Array{Float64,2}(undef,7,m)
      for j = 1:m
            stateEstimate[1:4,j] = qvec[j]
            stateEstimate[5:7,j] = xvec[j][4:6]
            stateTrue[:,j] = xTrue[j]
      end

      return filteringResults(stateEstimate, stateTrue, Pvec, measurements, measurementTimes, model.timeVector)
end

function MUKFstep(x, P, q, n, t, y, mtime, dynamics, measModel, Q, R, a, f, gamma, W, integrator)
      # Covariance Decomposition
      exitflag = true
      # if isposdef(P)
      #       Psq = cholesky(P).U
      # elseif isposdef(P+diagm(max(diag(P)...)*8*ones(6)))
      #       Psq = cholesky(P+diagm(max(diag(P)...)*8*ones(6))).U
      # else
      #       exitflag = false
      #       @infiltrate
      #       return x, P, q, exitflag
      # end
      Psq = similar(P)

      if isposdef(P)
            Psq[:] = cholesky(P).U
      elseif all(diag(P).>0)
            if isposdef(Hermitian(P))
                  Psq[:] = cholesky(Hermitian(P)).U
            else
                  try Psq[:] = cholesky(Hermitian(P+diagm(max(diag(P)...)*8*ones(6)))).U
                  catch
                        exitflag = false
                        return x, P, q, exitflag
                  end
            end
      elseif all(diag(P+diagm(max(diag(P)...)*8*ones(6))).>0)
            try Psq[:] = cholesky(Hermitian(P+diagm(max(diag(P)...)*8*ones(6)))).U
            catch
                  exitflag = false
                  return x, P, q, exitflag
            end
      else
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
      P1 = P
      X1 = X
      x1 = x
      q1 = q
      X,x,P,q = MUKFprop(X,q,n,t,W,Q,dynamics,integrator,a,f)

      P2 = P
      x2 = x
      # Update
      if mtime
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

function MUKFprop(X,q,n,tvec,W,Qb,dynamics,integrator,a,f)

    # preallocate varriables so that they are available in the entire loop
    qu = similar(q)
    qui = similar(q)

    ## propogate sigma points
    for j = 1:2*n+1
        dp = X[j][1:3]
        ws = X[j][4:6]
        # convert to error quaternions
        dq = p2q(dp,a,f)
        # npd2 = norm(dp)^2
        # dq4 = (-a*npd2+f*sqrt(f^2 + (1-a^2)*npd2))/(f^2 + npd2)
        # dqv = (a + dq4)*dp/f
        # dq = [dqvdq4]

        # calculate total quaternion
        qs = qprod(dq,q)

        # propogate quaternions
        (dx,_) = integrator(dynamics,tvec,[qs;ws])
        qp = dx[end][1:4]
        qp = qp./norm(qp)
        wp = dx[end][5:7]

        if j == 1
            qu = qp
            qui = qinv(qp)
            dpp = zeros(3)
        else
            # get updated error quaternion
            dqp = qprod(qp,qui)

            # convert to MRP (sigma points)
            dpp = q2p(dqp,a,f)
        end

        X[j] = [dpp;wp]
    end

    # calculate propogated state and covariance estimate
    x = W[1].*X[1] + W[3].*sum(view(X,2:length(X)))

    Xtemp = X .- [x]

    P = W[2].*outerProduct(Xtemp[1]) + W[4].*outerProductSum(view(Xtemp,2:length(Xtemp))) + .5*Qb

    return X,x,P,qu
end

function MUKFupdate(X,x,P,q,n,W,R,Q,yvec,measModel,a,f)

    # calculate the mean observation
    # gam = zeros(length(yvec),2*n+1)

    gam = Array{Array{Float64,1},1}(undef,2*n+1)
    for j = 1:2*n+1
        qx = qprod(p2q(X[j][1:3]),q)
        gam[j] = measModel(qx)
    end
    y = W[1].*gam[1] + W[3].*sum(view(gam,2:length(gam)))

    Ytemp = gam .- [y]
    Xtemp = X .- [x]

    # calcualte output covariance
    Pyy = W[2].*outerProduct(Ytemp[1]) + W[4].*outerProductSum(view(Ytemp,2:length(Ytemp)))

    # calculate innovation covariance
    Pvv = Pyy + R

    # calculate cross correlation
    Pxy = W[2].*outerProduct(Xtemp[1],Ytemp[1]) + W[4].*outerProductSum(view(Xtemp,2:length(Xtemp)), view(Ytemp,2:length(Ytemp)))

    # calculate the gain
    K = Pxy*inv(Pvv)

    # update the state and covariance
    x = x + K*(yvec - y)
    P = P - K*Pvv*K' + .5*Q

    return x,P
end
