function MUKF(model::nlFilteringProblem, options::UKFoptions)

      # define intermediate variables for convinience
      dynamics = model.dynamics
      measModel = model.measurementModel
      Q = model.processNoise
      R = model.measurementNoise
      uwCoeff = model.underweightCoeff

      # define integration function used to propagate the state dynamics
      if options.integrator == :RK4
            integrator = RK4
      elseif typeof(options.integrator) == Function
            integrator = options.integrator
      else
            error("unsupported integrator")
      end

      # compute the measurements over the simulation period
      (xTrue, measurements, timeVector, measurementTimes) = simulate(model, integrator)

      # @infiltrate

      # get length of simulation
      m = length(timeVector)

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
      lambda = options.alpha^2 * (6 + kappa) - 6

      # Define the weights:
      # W0_mean = lambda/(model.errorStateDimension+lambda)
      W0_mean = lambda / (6 + lambda)
      W0_cov = W0_mean + (1 - options.alpha^2 + options.beta)
      # Wi_mean = 1/(2*(model.errorStateDimension+lambda))
      Wi_mean = 1 / (2 * (6 + lambda))
      Wi_cov = Wi_mean
      W = [W0_mean; W0_cov; Wi_mean; Wi_cov]

      # define gamma
      # gamma = sqrt(model.errorStateDimension+lambda)
      gamma = sqrt(6 + lambda)

      # initalize state and covariance
      Pvec = Array{Array{Float64,2},1}(undef, m)
      Pvec[1] = model.initial.covariance
      x = deepcopy(model.initial.errorState)
      P = deepcopy(model.initial.covariance)
      q = deepcopy(model.initial.estimateState[1:4])

      # pre-allocate arrays 
      Psq = similar(P)
      X = Array{Array{Float64,1},1}(undef, 13)

      stateEstimate = Array{Float64,2}(undef, 7, m)
      stateTrue = Array{Float64,2}(undef, 7, m)
      stateEstimate[:, 1] = deepcopy(model.initial.estimateState)
      stateTrue[:, 1] = xTrue[1]
      tf = m
      exitflag = false


      # main loop
      for i = 2:m   
    
            t = view(timeVector, i-1:i)

            # compute filtering updates

            # reset attitude error state to zero
            x[1:3] = zeros(3)
            # for numberical stability, average off-diagonal element of covariance
            P = (P + P') ./ 2

            # Covariance Decomposition
            X, exitflag = sigmaPoints(X, x, P, Psq, gamma)

            # check exit condition
            if exitflag
                  tf = i - 1
                  break
            end

            # propogate
            X, x, P, q = MUKFprop(X, q, 6, t, W, Q, dynamics, integrator, options.a, options.f)

            # Update
            if measurementTimes[i]
                  x, P = MUKFupdate(X, x, P, q, 6, W, R, uwCoeff, measurements[i], measModel, options.a, options.f)
            end

            # update quaternion
            # mrp error
            dp = x[1:3]
            # convert to error quaternions
            dq = p2q(dp, options.a, options.f)
            q = qprod(dq, q)
            q = q ./ norm(q)

            # store state and covariance
            stateEstimate[1:4, i] = deepcopy(q)
            stateEstimate[5:7, i] = deepcopy(x[4:6])
            Pvec[i] = deepcopy(P)
            stateTrue[:, i] = xTrue[i]
    
      end


      return filteringResults(stateEstimate[:, 1:tf], stateTrue[:, 1:tf], Pvec[1:tf], measurements[1:tf], measurementTimes[1:tf], timeVector[1:tf])
end

function GM_MUKF(model::nlFilteringProblem, options::UKFoptions)

      # define intermediate variables for convinience
      dynamics = model.dynamics
      measModel = model.measurementModel
      Q = model.processNoise
      R = model.measurementNoise
      uwCoeff = model.underweightCoeff

      # define integration function used to propagate the state dynamics
      if options.integrator == :RK4
            integrator = RK4
      elseif typeof(options.integrator) == Function
            integrator = options.integrator
      else
            error("unsupported integrator")
      end

      # compute the measurements over the simulation period
      (xTrue, measurements, timeVector, measurementTimes) = simulate(model, integrator)

      # get length of simulation
      m = length(timeVector)
      tf = m

      # number of gaussian components
      n = length(model.initial.covariances)

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
      lambda = options.alpha^2 * (6 + kappa) - 6

      # Define the weights:
      # W0_mean = lambda/(model.errorStateDimension+lambda)
      W0_mean = lambda / (6 + lambda)
      W0_cov = W0_mean + (1 - options.alpha^2 + options.beta)
      # Wi_mean = 1/(2*(model.errorStateDimension+lambda))
      Wi_mean = 1 / (2 * (6 + lambda))
      Wi_cov = Wi_mean
      W = [W0_mean; W0_cov; Wi_mean; Wi_cov]

      # define gamma
      # gamma = sqrt(model.errorStateDimension+lambda)
      gamma = sqrt(6 + lambda)

      # initialize gaussian components and weights
      GM_means = deepcopy(model.initial.means)
      GM_covariances = deepcopy(model.initial.covariances)
      GM_w = deepcopy(model.initial.weights)
      β = similar(GM_w)

      # initalize state and covariance
      stateEstimate = Array{Float64,2}(undef, 7, m)
      stateTrue = Array{Float64,2}(undef, 7, m)
      Pvec = Array{Array{Float64,2},1}(undef, m)

      # initialize arrays for sigma point calculations
      Psq = similar(GM_covariances[1])
      X = Array{Array{Float64,1},1}(undef, 13)
      exitflag = false


      # main loop
      for i = 2:m

            t = view(timeVector, i-1:i)

            for j = 1:n

                  # error and attitude states for the current GM
                  x = vcat(zeros(3), GM_means[j][5:7])
                  q = GM_means[j][1:4]

                  # covariance
                  P = GM_covariances[j]

                  # for numberical stability, average off-diagonal element of covariance
                  P = (P + P') ./ 2

                  # Covariance Decomposition
                  X, exitflag = sigmaPoints(X, x, P, Psq, gamma)

                  # check exit condition
                  if exitflag
                        tf = i - 1
                        break
                  end

                  # propogate
                  X, x, P, q = MUKFprop(X, q, 6, t, W, Q, dynamics, integrator, options.a, options.f)

                  # Update
                  if measurementTimes[i]
                        x, P, y, Py = MUKFupdate(X, x, P, q, 6, W, R, uwCoeff, measurements[i], measModel, options.a, options.f)
                  end

                  # update quaternion
                  # mrp error
                  dp = x[1:3]
                  # convert to error quaternions
                  dq = p2q(dp, options.a, options.f)
                  q = qprod(dq, q)
                  q = q ./ norm(q)
                  
                  # store updated mean, covariance, and weights for each gaussian mixture
                  GM_means[j][1:4] = deepcopy(q)
                  GM_means[j][5:7] = deepcopy(x[4:6])
                  GM_covariances[j] = deepcopy(P)

                  if measurementTimes[i]
                        try
                              β[j] = pdf_gaussian(measurements[i], y, Py)
                        catch
                              β[j] = β[j] 
                        end
                  end
            
            end

            if exitflag
                  tf = i - 1
                  break
            end
      
            if measurementTimes[i]
                  # update gaussian mixture weights
                  GM_w = (GM_w .* β) ./ dot(GM_w, β)
            end

            # calculate the mean state and covariance using the method of moments (adapted for attitudes)
            # store in state and covariance history
            stateEstimate[:, i], Pvec[i] = MoM_att(GM_w, GM_means, GM_covariances) #MoM_att(w, x, P)

            # store true state
            stateTrue[:, i] = xTrue[i]
      end

      return filteringResults(stateEstimate[:, 1:tf], stateTrue[:, 1:tf], Pvec[1:tf], measurements[1:tf], measurementTimes[1:tf], timeVector[1:tf])
end

function MEKF(model :: nlFilteringProblem, options :: EKFoptions)
end

function EKF(model :: nlFilteringProblem, options :: EKFoptions)

      # define integration function used to propagate the state dynamics
      if options.integrator == :RK4
           integrator = RK4
      elseif options.integrator == :custom
           integrator = options.integratorFunction
      else
           error("unsupported integrator")
      end

      # compute the measurements over the simulation period
      (xTrue,measurements, timeVector, measurementTimes) = simulate(model,integrator)

      # get length of simulation
      m = length(timeVector)
      dt = model.timeVector[2] - timeVector[1]

      # initalize state and covariance
      xvec = Array{Array{Float64,1},1}(undef,m)
      Pvec = Array{Array{Float64,2},1}(undef,m)
      xvec[1] = model.initialState
      Pvec[1] = model.initialCovariance
      x = deepcopy(model.initialState)
      P = deepcopy(model.initialCovariance)

      # generate anonymous function that is used to find the state transition matrix phi via integration
      phidot = (F, phi_0) ->
      begin

      end

      # filter loop
      for i = 2:m

            H = ForwardDiff.jacobian(model.measurementModel, x)
            # update the state and covariance based on the measurement
            x_update, P_update = KFupdate(x, P, H, model.measurementNoise, measurements, measurementTimes)

            # propogate state
            x, P = EKFprop(x_update, P_update, model.dynamics, model.processNoise, dt , integrator)

            # store data
            xvec[i] = deepcopy(x)
            Pvec[i] = deepcopy(P)

      end

      return filteringResults(xvec, xTrue, Pvec, measurements, measurementTimes, timeVector)
end

function KF(model :: linearFilteringProblem)

      # handle discrete and continuous models
      if model.isDiscrete
            H = model.measurementMatrix
            phi = model.stateMatrix
      else
            H = model.measurementMatrix
            dt = model.timeVector[2] - model.timeVector[1]
            phi = exp(model.stateMatrix.*dt)
      end

      # compute the measurements over the simulation period
      (xTrue,measurements,measurementTimes) = simulate(model, phi)

      # get length of simulation
      m = length(model.timeVector)

      # initalize state and covariance
      xvec = Array{Array{Float64,1},1}(undef,m)
      Pvec = Array{Array{Float64,2},1}(undef,m)
      xvec[1] = model.initialState
      Pvec[1] = model.initialCovariance
      x = deepcopy(model.initialState)
      P = deepcopy(model.initialCovariance)

      # filter loop
      for i = 2:m

            # update the state and covariance based on the measurement
            x_update, P_update = KFupdate(x, P, H, model.measurementNoise, measurements[i], measurementTimes[i])

            # propogate state
            x, P = KFprop(x_update, P_update, phi, model.processNoise)

            # store data
            xvec[i] = deepcopy(x)
            Pvec[i] = deepcopy(P)

      end

      return filteringResults(xvec, xTrue, Pvec, measurements, measurementTimes, model.timeVector)
end
