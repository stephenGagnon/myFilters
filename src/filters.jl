function MUKF(model::nlFilteringProblem, options::UKFoptions, timeVector, integrator,  measurements, measurementTimes, xTrue = nothing)

      # define intermediate variables for convinience
      dynamics = model.dynamics
      measModel = model.measurementModel
      Q = model.processNoise
      R = model.measurementNoise
      uwCoeff = model.underweightCoeff

      # parameter for error handling, gets incremented when the covariance is modified to deal with numerical issues
      error_occurences = 0

      # @infiltrate

      # length of simulation
      m = length(timeVector)

      ## compute UKF parameters based on used defined alpha and beta

      # deafult kappa value, specified by inputting a kappa value of NaN
      # user specified kappa values will override the default
      if isnan(options.kappa)
            #Minimizes mean square error up to 4th order for a 6D system, need to be careful with negative kappa.
            # kappa = 3-model.errorStateDimension
            kappa = -3
      else
            kappa = options.kappa
      end

      #scaling parameter
      # lambda = options.alpha^2*(model.errorStateDimension+kappa)-model.errorStateDimension
      lambda = options.alpha^2 * (6 + kappa) - 6

      # weights for unscented transform first two are for mean sigma points, second two are for other sigma points:
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
      Pvec[1] = copy(model.initial.covariance)

      x = deepcopy(model.initial.errorState)
      P = deepcopy(model.initial.covariance)
      q = deepcopy(model.initial.estimateState[1:4])

      # pre-allocate arrays 
      Psq = similar(P)
      X = Array{Array{Float64,1},1}(undef, 13)

      stateEstimate = Array{Float64,2}(undef, 7, m)
      # stateTrue = Array{Float64,2}(undef, 7, m)
      stateEstimate[:, 1] = deepcopy(model.initial.estimateState)
      # stateTrue[:, 1] = xTrue[1]
      tf = m
      exitflag = false

      # initialize measurement residual and covariance
      Pyyvec = Array{Array{Float64,2},1}(undef, m)
      residual_history = similar(measurements)

      # compute initial residuals
      X, _ = sigmaPoints(X, x, P, Psq, gamma)
      # @infiltrate
      residual, Pyy = computeUKFResidual(measurements[1],X,W,q,6,measModel)

      Pyyvec[1] = copy(Pyy)
      residual_history[1] = copy(residual)


      # main loop
      for i = 2:m   
    
            t = timeVector[i-1:i]

            # compute filtering updates
            # @infiltrate
            # reset attitude error state to zero
            x[1:3] = zeros(3)
            # for numberical stability, average off-diagonal element of covariance
            P = (P + P') ./ 2

            # Covariance Decomposition
            X, exitflag = sigmaPoints(X, x, P, Psq, gamma)

            # error handling for cholesky decomposition issues
            if exitflag == 1
                  tf = i - 2
                  print("cholesky decomposition failed, Covariance is non-symmetric \n")
                  print("covariance = ")
                  display(P) 
                  print("\n")
                  print("iteration #", i-1, "\n")
                  measNo = sum(measurementTimes[1:i])
                  print("number of measurements = ", measNo, "\n")
                  @infiltrate
                  print("eigenvalues of the covariance matrix: \n")
                  display(eigvals(P))
                  break
            elseif exitflag == 2
                  tf = i - 2
                  print("cholesky decomposition failed, Covariance eigenvalues are negative \n")
                  print("covariance = ")
                  display(P) 
                  display(eigvals(P))
                  print("\n")
                  print("iteration #", i-1, "\n")
                  measNo = sum(measurementTimes[1:i])
                  print("number of measurements = ", measNo, "\n")
                  break
            elseif exitflag == 3
                  error_occurences += 1
            elseif exitflag == 4
                  tf = i - 2
                  print("cholesky decomposition failed, Covariance contains NaN values \n")
                  print("covariance = ")
                  display(P) 
                  display(eigvals(P))
                  print("\n")
                  print("iteration #", i-1, "\n")
                  measNo = sum(measurementTimes[1:i])
                  print("number of measurements = ", measNo, "\n")
                  break
            end

            # propogate
            X, x, P, q = MUKFprop(X, q, 6, t, W, Q, dynamics)

            # Update
            if measurementTimes[i]
                  # @infiltrate
                  x, P, Pyy, residual = MUKFupdate(X, x, P, q, 6, W, R, uwCoeff, measurements[i], measModel)
                  
                  # update quaternion
                  # convert to error quaternions
                  q = qprod(p2q(x[1:3],1,4), q)
                  q = q ./ norm(q)

            else
                  residual, Pyy = computeUKFResidual(measurements[i],X,W,q,6,measModel)
                  # update quaternion
                  # convert to error quaternions
                  q = qprod(p2q(x[1:3],1,4), q)
                  q = q ./ norm(q)
                  # @infiltrate
            end
            residual_history[i] = residual 
            Pyyvec[i] = Pyy

            # store state and covariance
            stateEstimate[1:4, i] = deepcopy(q)
            stateEstimate[5:7, i] = deepcopy(x[4:6])
            Pvec[i] = deepcopy(P)
            # stateTrue[:, i] = xTrue[i]
    
      end

      if error_occurences > 0
            print("Finished \n")
            print("Covariance modified ", error_occurences, " times due to small negative eigenvalues \n")
      else 
            print("Finished \n")
      end

      return filteringResults(stateEstimate[:, 1:tf], Pvec[1:tf], residual_history[1:tf], Pyyvec[1:tf])
end

# need to update to separate sim code from filtering 
function GM_MUKF(model::nlFilteringProblem, options::UKFoptions)

      save_GM_data = true

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

      # number of measurements
      n_measurements = length(measurements[1])

      # number of gaussian components
      n_mixtures = length(model.initial.covariances)

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

      GM_residuals = Array{Array{Float64,1},1}(undef,m)
      GM_residual_covariances = Array{Array{Float64,2},1}(undef,m)

      # if selected, save GM data for analysis purposes
      if save_GM_data
            GM_m_hist = Array{typeof(GM_means),1}(undef, m)
            GM_m_hist[1] = deepcopy(GM_means)
      
            GM_c_hist = Array{typeof(GM_covariances),1}(undef, m)
            GM_c_hist[1] = deepcopy(GM_covariances)
      
            GM_w_hist = Array{typeof(GM_w),1}(undef, m)
            GM_w_hist[1] = deepcopy(GM_w)
      else
            GM_m_hist = nothing
            GM_c_hist = nothing 
            GM_w_hist = nothing
      end

      # initalize state and covariance 
      stateEstimate = Array{Float64,2}(undef, 7, m)
      stateTrue = Array{Float64,2}(undef, 7, m)
      Pvec = Array{Array{Float64,2},1}(undef, m)
      residuals = Array{Float64,2}(udnef,n_measurements,m)
      residual_Covariances = Array{Array{Float64,2},1}(undef, m)

      # initial state
      stateEstimate[:, 1], Pvec[1] = MoM_att(GM_w, GM_means, GM_covariances) #MoM_att(w, x, P)
      stateTrue[:, 1] = xTrue[1]

      # initialize arrays for sigma point calculations
      Psq = similar(GM_covariances[1])
      X = Array{Array{Float64,1},1}(undef, 13)
      exitflag = false


      # main loop
      for i = 2:m

            t = view(timeVector, i-1:i)

            for j = 1: n_mixtures

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
                        print("cholesky decomposition failed")
                        break
                  end

                  # propogate
                  X, x, P, q = MUKFprop(X, q, 6, t, W, Q, dynamics)

                  # Update
                  if measurementTimes[i]
                        x, P, Pvv, residual = MUKFupdate(X, x, P, q, 6, W, R, uwCoeff, measurements[i], measModel)
                  
                        # update quaternion
                        # convert to error quaternions
                        q = qprod(p2q(x[1:3],1,4), q)
                        q = q ./ norm(q)
      
                  else
                        residual, Pvv = computeUKFResidual(measurements[i],X,W,q,6,measModel)
                        # update quaternion
                        # convert to error quaternions
                        q = qprod(p2q(x[1:3],1,4), q)
                        q = q ./ norm(q)
                        # @infiltrate
                  end

                  # store updated mean, covariance, and weights for each gaussian mixture
                  GM_means[j][1:4] = deepcopy(q)
                  GM_means[j][5:7] = deepcopy(x[4:6])
                  GM_covariances[j] = deepcopy(P)
                  GM_residuals[j] = deepcopy(residual)
                  GM_residual_covariances[j] = deepcopy(Pvv)

                  if measurementTimes[i]
                        β[j] = pdf_gaussian(measurements[i], y, Py)
                        # if β[j] == 0
                        #       @infiltrate
                        # end
                        if isnan(β[j])
                              exitflag = true
                              tf = i-1
                              print("gaussian PDF computation failed (liekly due to inverse of covaraince)")
                              break
                        end
                  end

            end

            if exitflag
                  tf = i - 1
                  print("error")
                  break
            end

            if measurementTimes[i]
                  # update gaussian mixture weights
                  s = sum(GM_w .* β)
                  # print("β =", β,"\n")
                  # print("GM_w =", GM_w,"\n")
                  if s == 0
                        tf = i - 1
                        print("sum of GM weigts is zero \n")
                        print("β =", β,"\n")
                        print("GM_w =", GM_w,"\n")
                        break
                  end
                  for k = 1:lastindex(β)
                        GM_w[k] = (GM_w[k] * β[k]) / s
                  end
            end

            # calculate the mean state and covariance using the method of moments (adapted for attitudes)
            # store in state and covariance history
            if any(isnan.(GM_w)) | any(any.([isnan.(v) for v in GM_means]))
                  # @infiltrate
                  tf = i-1
                  print("nan values in GM weights or means")
                  break 
            end

            if save_GM_data
                  GM_m_hist[i] = deepcopy(GM_means)
                  GM_c_hist[i] = deepcopy(GM_covariances)
                  GM_w_hist[i] = deepcopy(GM_w)
            end

            stateEstimate[:, i], Pvec[i] = MoM_att(GM_w, GM_means, GM_covariances) #MoM_att(w, x, P)
            residuals[:,i], residual_Covariances[i] = MoM(GM_w, GM_residuals, GM_residual_covariances)

      end

      # return filteringResults(stateEstimate[:, 1:tf], stateTrue[:, 1:tf], Pvec[1:tf], measurements[1:tf], measurementTimes[1:tf], timeVector[1:tf]), (GM_w_hist[1:tf], GM_m_hist[1:tf], GM_c_hist[1:tf])
      return filteringResults(stateEstimate[:, 1:tf], Pvec[1:tf], residuals, residual_Covariances), (GM_w_hist[1:tf], GM_m_hist[1:tf], GM_c_hist[1:tf])
end

# not finished
# Multiplicative Extended Kalman Filter for Attitude Estimation
function MEKF(model :: nlFilteringProblem, options :: EKFoptions)
end

#not finished
# need to update to separate sim code from filtering 
#Extended Kalman Filter
function EKF(model :: nlFilteringProblem, options :: EKFoptions)

      computeResidual = true

      # define integration function used to propagate the state dynamics
      if options.integrator == :RK4
           integrator = RK4
      elseif options.integrator == :custom
           integrator = options.integratorFunction
      else
           error("unsupported integrator")
      end

      # compute the measurements over the simulation period
      (xTrue, measurements, timeVector, measurementTimes) = simulate(model,integrator)

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

      residuals = Array{Array{Float64,1},1}(undef,m)
      P_residual = Array{Array{Float64,2},1}(undef,m)

      if computeResidual
            residuals[1], P_residual[1] = computeKFReidual(measurements[1], x, P, ForwardDiff.jacobian(model.measurementModel, x), model.measurementNoise, computeResidual)
      end

      # generate anonymous function that is used to find the state transition matrix phi via integration
      phidot = (F, phi_0) ->
      begin

      end

      # filter loop
      for i = 2:m

            H = ForwardDiff.jacobian(model.measurementModel, x)

            # update the state and covariance based on the measurement
            if measurementTimes[i]
                  x, P, residual, residualCovariance = KFupdate(x, P, H, model.measurementNoise, measurements[i], computeResidual)
            else 
                  residual, residualCovariance = computeKFReidual(meas, x, P, H, model.measurementNoise, computeResidual)
            end

            # propogate state
            x, P = EKFprop(x, P, model.dynamics, model.processNoise, dt , integrator)

            # store data
            xvec[i] = deepcopy(x)
            Pvec[i] = deepcopy(P)

            if computeResidual
                  residuals[i] = residual
                  P_residual[i] = residualCovariance
            end

      end

      return filteringResults(xvec, Pvec, residuals, P_residual)
end

# need to update to separate sim code from filtering 
# linear Kalman Fitler
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


      residuals = Array{Array{Float64,1},1}(undef,m)
      P_residual = Array{Array{Float64,2},1}(undef,m)

      if computeResidual
            residuals[1], P_residual[1] = computeKFReidual(measurements[1], x, P, ForwardDiff.jacobian(model.measurementModel, x), model.measurementNoise, computeResidual)
      end

      # filter loop
      for i = 2:m

            # propogate state
            x, P = KFprop(x, P, phi, model.processNoise)

            # update the state and covariance based on the measurement
            if measurementTimes[i]
                  x, P, residual, residualCovariance = KFupdate(x, P, H, model.measurementNoise, measurements[i], computeResidual)
            else 
                  residual, residualCovariance = computeKFReidual(meas, x, P, H, model.measurementNoise, computeResidual)
            end

            # store data
            xvec[i] = deepcopy(x)
            Pvec[i] = deepcopy(P)

            if computeResidual
                  residuals[i] = residual
                  P_residual[i] = residualCovariance
            end

      end

      return filteringResults(xvec, Pvec, residuals, P_residual)
end
