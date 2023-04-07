function MUKFprop(X,q,n,t,W,Qb,dynamics;data = nothing, diagnostics = false)

    # preallocate varriables so that they are available in the entire loop
    # qui holds the inverse of the updated mean quaternion to avoid recomputing the inverse for each sigma point
    qui = similar(q)
    # q_update holds the updated mean quaternion, and is necessary because the un-updated quaternion is used to compute the errror for each sigma point, so it cannot be updated immediately
    q_update = similar(q)
    
    ## propogate sigma points
    for j = eachindex(X)
        dp = X[j][1:3]
        ws = X[j][4:6]

        # calculate quaternion including error for current sigma point
        qe = qprod(p2q(dp,1,4),q)

        # propogate quaternions
        xp = _RK4(dynamics, t[1], t[2]-t[1], [qe;ws])


        qp = xp[1:4]
        qp = qp./norm(qp)
        wp = xp[5:7]

        if j == 1
            # only for the first sigma point, store the propagated mean sigma point, and compute the inverse for future use
            q_update[:] = copy(qp)
            qui[:] = qinv(qp)
            # the mean sigma point propagation error is set to zero
            dpp = zeros(3)
        else
            # compute error MRP for current sigma point wrt mean sigma point
            dpp = q2p(qprod(qp,qui),1,4)
        end

        X[j] = [dpp;wp]
    end

    # calculate propogated state and covariance estimate
    x = W[1].*X[1] + W[3].*sum(X[2:end])

    X_deviations = X .- [x]
    P = W[2].*outerProduct(X_deviations[1]) + W[4].*outerProductSum(X_deviations[2:length(X_deviations)]) + Qb

    q[:] = q_update

    return X,x,P,q
end

function MUKFupdate(X,x,P,q,n,W,R,underweightCoeff,yvec,measModel;diagnostics=false, trueState = [0], data = nothing)

    # calculate the mean observation
    gam = Array{Array{Float64,1},1}(undef,2*n+1)
    for j = 1:2*n+1
        gam[j] = measModel(qprod(p2q(X[j][1:3],1,4),q))
    end
    y = W[1].*gam[1] + W[3].*sum(gam[2:end])

    Y_deviations = gam .- [y]
    X_deviations = X .- [x]

    # calculate output covariance
    Pyy = W[2].*outerProduct(Y_deviations[1]) + W[4].*outerProductSum(Y_deviations[2:end])

    # calculate innovation covariance
    Pvv = underweightCoeff*Pyy + R

    # calculate cross correlation
    Pxy = W[2].*outerProduct(X_deviations[1],Y_deviations[1]) + W[4].*outerProductSum(X_deviations[2:end], Y_deviations[2:end])

    # calculate the gain
    K = Pxy*pinv(Pvv)

    # update the state and covariance
    x = x + K*(yvec - y)
    P = P - K*Pvv*K'

    return x, P, Pvv, yvec - y
end

function computeUKFResidual(yvec, X, W, q, n, measModel, underweightCoeff, R)
    # calculate the mean observation
    gam = Array{Array{Float64,1},1}(undef,2*n+1)
    for j = 1:2*n+1
        gam[j] = measModel(qprod(p2q(X[j][1:3],1,4),q))
    end
    
    y = W[1].*gam[1] + W[3].*sum(gam[2:end])

    Y_deviations = Array{Float64,2}(undef,length(gam[1]),length(gam))
    for i = 1:length(gam)
        Y_deviations[:,i] = gam[i] - y
    end

    Pyy = W[2] .* Y_deviations[:,1]*Y_deviations[:,1]' + W[4] .* Y_deviations[:,2:end]*Y_deviations[:,2:end]'
    Pvv = underweightCoeff*Pyy + R

    return yvec-y, Pvv
end

function KFupdate(x, P, H, R, meas, calcResid)
 
    # calculate residual and covariance
    residual = meas - H*x
    if calcResidual
        residualCovariance = H*P*H' + R
    else 
        residualCovariance = NaN
    end

    # measurement update
    K = KalmanGain(P, H, R)
    x_up = x + K*()

    Pup = (eye(length(x)) - K*H)*P

    return x_up, Pup, residual, residualCovariance

end

function computeKFReidual(meas, x, P, H, R, computeResidual)

    residual = meas - H*x

    if computeResidual
        residualCovariance = H*P*H' + R
    else
        residualCovariance = NaN
    end

    return residual, residualCovariance
end

function KFprop(x_in, P_in, phi, Q)
    # propogate state
    x = phi*x_in

    # covariance update and propogation
    P = phi*P_in*phi' + Q

    return x,P
end

function EKFprop(x_in, P_in, dynamics, Q, dt, integrator)

    (dx,_) = integrator(dynamics, [0, dt] ,x_in)

    F = ForwardDiff.jacobian(dynamics, x_in)

    # calculate phi MISSING

    phi_0 = eye(length(x_in))


    P = phi*P_in*phi' + Q

    return x, P
end
