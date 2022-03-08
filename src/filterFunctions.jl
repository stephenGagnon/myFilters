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
