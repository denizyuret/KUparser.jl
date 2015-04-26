typealias Model Vector{Net}     # The 3-net model

function KUnet.predict(p::Parser, model::Model, x, y=similar(x, p.nmove, size(x,2)); batch=0)
    @assert length(model) == 3
    @assert size(y) == (p.nmove, size(x,2))
    (mnet, lnet, rnet) = model    # mnet: 4 moves, lnet: left deprels, rnet: right deprels
    ym = predict(mnet, x; batch=batch) # SHIFT=1, LEFT=2, RIGHT=3, [REDUCE=4]
    @assert size(ym) == ((isa(p,ArcHybridR1) ? 3 : 4), size(x,2))
    yl = predict(lnet, x; batch=batch)
    @assert size(yl) == (p.ndeps, size(x,2))
    yr = predict(rnet, x; batch=batch)
    @assert size(yr) == (p.ndeps, size(x,2))
    for i=1:size(y,2)
        for m=1:size(y,1)       # y[m,i]: move m for instance i
            if m == shiftmove(p)
                y[m,i] = ym[1,i]
            elseif m == reducemove(p)
                y[m,i] = ym[4,i]
            elseif in(m, leftmoves(p))
                y[m,i] = ym[2,i] + yl[label(p,m),i]
            elseif in(m, rightmoves(p))
                y[m,i] = ym[3,i] + yr[label(p,m),i]
            else
                error("Move $m is not supported")
            end
        end
    end
    return y
end

# this may need parser as arg
function KUnet.train(p::Parser, model::Model, xx, yy; batch=128, iters=0, loss=softmaxloss, shuffle=false)
    @assert length(model) == 3  # move, left, right
    shuffle && KUnet.shufflexy!(xx,yy)
    (batch == 0) && (batch = size(xx,2))
    (x3,y3) = trainsplit(p, xx, yy)
    for imodel=1:3
        net = model[imodel]
        x = x3[imodel]
        y = y3[imodel]
        xrows,xcols = size(x)
        yrows,ycols = size(y)
        buf = KUnet.inittrain(net, x, y, batch)
        for b = 1:batch:xcols
            e = b + batch - 1
            if (e > xcols)
                e = xcols
                KUnet.chksize(buf, :x, net[1].w, (xrows, e-b+1))
                KUnet.chksize(buf, :y, net[end].w, (yrows, e-b+1))
            end
            copy!(buf.x, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
            copy!(buf.y, (1:yrows,1:e-b+1), y, (1:yrows,b:e))
            KUnet.backprop(net, buf.x, buf.y, loss)
            for l in net
                isdefined(l,:w) && KUnet.update(l.w, l.dw, l.pw)
                isdefined(l,:b) && KUnet.update(l.b, l.db, l.pb)
            end
            iters > 0 && e/batch >= iters && break
        end
        KUnet.free(buf.x); KUnet.free(buf.y) # this should not be necessary now that gc() works...
    end
end

function trainsplit(p::Parser, xx, yy)
    (m3,) = findn(yy)           # all the moves in original encoding
    @assert length(m3)==size(yy,2)
    m4 = similar(m3)            # converting to unlabeled move encoding
    for i=1:length(m3)          # SHIFT=1, LEFT=2, RIGHT=3, [REDUCE=4]
        m4[i] = ((m3[i] == shiftmove(p)) ? 1 :
                 (m3[i] == reducemove(p)) ? 4 :
                 in(m3[i], leftmoves(p)) ? 2 : 3)
    end

    x3 = Array(typeof(xx), 3)
    y3 = Array(typeof(yy), 3)

    x3[1] = xx                  # the first (move type) training set has all instances
    y3[1] = zeros(eltype(yy), (isa(p, ArcHybridR1) ? 3 : 4), size(yy, 2))
    y3[1][sub2ind(size(y3[1]), m4, 1:size(y3[1],2))] = one(eltype(yy))

    x3[2] = xx[:,m4.==2]        # second dataset is for the left labels
    y3[2] = zeros(eltype(yy), p.ndeps, size(x3[2],2))
    idx = 0
    for i=1:length(m3)
        (m4[i] == 2) || continue
        idx += 1
        y3[2][label(p,m3[i]),idx] = one(eltype(yy))
    end
    @assert idx == size(y3[2],2)

    x3[3] = xx[:,m4.==3]        # third dataset is the right labels
    y3[3] = zeros(eltype(yy), p.ndeps, size(x3[3],2))
    idx = 0
    for i=1:length(m3)
        (m4[i] == 3) || continue
        idx += 1
        y3[3][label(p,m3[i]),idx] = one(eltype(yy))
    end
    @assert idx == size(y3[3],2)

    return (x3,y3)
end
