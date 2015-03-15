# The greedy transition based parser parses the sentence using the
# following steps:

function gparse(s::Sentence, n::Net, f::Features; pred::Bool=true, feat::Bool=true)
    (p, x, y, score) = initgparse(s,n,f)
    nx = 0
    while (v = valid(p); any(v))
        nx += 1; xx = sub(x,:,nx:nx)
        y[indmin(cost(p, s.head)), nx] = one(eltype(y))
        feat ? features(p, s, f, xx) : rand!(xx)
        pred ? predict(n, xx, score) : rand!(score)
        score[!v,:] = -Inf
        move!(p, indmax(score))
    end
    (p.head, sub(x,:,1:nx), sub(y,:,1:nx))
end

function initgparse(s::Sentence, n::Net, f::Features)
    (ndims, nword) = size(s.wvec)
    xtype = eltype(n[1].w)
    xrows = flen(ndims, f)
    xcols = 2 * (nword - 1)
    p = ArcHybrid(nword)
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, p.nmove, xcols)
    score = Array(xtype, p.nmove, 1)
    (p, x, y, score)
end

# We parse a corpus using a for loop or map:

function gparse(c::Corpus, n::Net, f::Features; args...)
    p = map(s->gparse(s,n,f;args...), c)
    h = map(z->z[1], p)
    x = hcat(map(z->z[2], p)...)
    y = hcat(map(z->z[3], p)...)
    (h, x, y)
end

# There are two opportunities for parallelism:
# 1. We process multiple sentences to minibatch net input.
#    This speeds up predict.

function gparse(corpus::Corpus, net::Net, feats::Features, batch::Integer; feat::Bool=true)
    # determine dimensions
    (batch > length(corpus)) && (batch = length(corpus))
    nsent = length(corpus)
    nword = 0; for s in corpus; nword += wcnt(s); end
    xcols = 2 * (nword - nsent)
    wvec1 = corpus[1].wvec
    wdims = size(wvec1,1)
    xrows = flen(wdims, feats)
    xtype = eltype(wvec1)
    yrows = ArcHybrid(1).nmove

    # initialize arrays
    p = Array(ArcHybrid, nsent) 	# parsers
    v = Array(Bool, yrows, nsent)       # valid move arrays
    x = Array(xtype, xrows, xcols)      # feature vectors
    y = Array(xtype, yrows, xcols)      # predicted move scores
    z = zeros(xtype, yrows, xcols)      # mincost moves, 1-of-k encoding
    svalid = Array(Int, batch)          # indices of valid sentences in current batch
    idx = 0                             # index of last used column in x, y, z
    for s=1:nsent; p[s] = ArcHybrid(wcnt(corpus[s])); end
    xxcols = batch
    xx = similar(net[1].w, (xrows, xxcols)) # device array

    # parse corpus[b:e] in parallel
    for b = 1:batch:nsent
        e = b + batch - 1
        (e > nsent) && (e = nsent; batch = e - b + 1)
        for s=b:e
            p[s] = ArcHybrid(wcnt(corpus[s]))
            svalid[s-b+1] = s
        end
        nvalid = batch
        while true
            # Update svalid and nvalid
            nv = 0
            for i=1:nvalid
                s = svalid[i]
                vs = sub(v, :, s)
                valid(p[s], vs)
                any(vs) && (nv += 1; svalid[nv] = s)
            end
            (nv == 0) && break
            nvalid = nv

            # svalid[1:nvalid] are the indices of still valid sentences in current batch
            # Take the next move with them
            # First calculate features x[:,idx+1:idx+nvalid]
            for i=1:nvalid
                s = svalid[i]
                feat ? features(p[s], corpus[s], feats, sub(x, :, idx + i)) : rand!(sub(x,:,idx+i))
            end

            KUnet.predict(net, sub(x, 1:xrows, idx+1:idx+nvalid), sub(y, 1:yrows, idx+1:idx+nvalid))

            # Finally find best moves and execute max score valid moves
            for i=1:nvalid
                s = svalid[i]
                bestmove = indmin(cost(p[s], corpus[s].head))
                z[bestmove, idx+i] = one(xtype)
                maxmove, maxscore = 0, -Inf
                for j=1:yrows
                    yj = y[j,idx+i]
                    v[j,s] && (yj > maxscore) && ((maxmove, maxscore) = (j, yj))
                end
                move!(p[s], maxmove)
            end # for i=1:nvalid
            idx = idx + nvalid
        end # while true
    end # for b = 1:batch:nsent
    KUnet.free(xx)
    h = Array(Pvec, nsent) 	# predicted heads
    for s=1:nsent; h[s] = p[s].head; end
    return (h, sub(x,:,1:idx), sub(z,:,1:idx))
end

# 2. We do multiple batches in parallel to utilize CPU cores.
#    This speeds up features.
#
# nworkers() gives the number of processes available
# distribute() splits the array based on number of workers
#
# TODO: find a way to share a CudaArray among processes.
# TODO: find a way to make this work inside gparse:
#   (nworkers() < ncpu) && (addprocs(ncpu - nprocs() + 1))
#   require("CUDArt")
#   @everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
#   require("KUparser")

function gparse(corpus::Corpus, net::Net, feats::Features, batch::Integer, ncpu::Integer)
    assert(nworkers() >= ncpu)
    d = distproc(corpus, workers()[1:ncpu])
    n = testnet(net)
    @everywhere gc()
    p = pmap(procs(d)) do x
        gparse(localpart(d), copy(n, :gpu), feats, batch)
    end
    # n=d=nothing; @everywhere gc()
    # h = vcat(map(z->z[1], p)...)
    # x = hcat(map(z->z[2], p)...)
    # y = hcat(map(z->z[3], p)...)
    # p=nothing; @everywhere gc()
    # (h, x, y)
end

function distproc(a::AbstractArray, procs)
    owner = myid()
    rr = RemoteRef()
    put!(rr, a)
    d = DArray(size(a), procs) do I
        remotecall_fetch(owner, ()->fetch(rr)[I...])
    end
    # Ensure that all workers have fetched their localparts.
    # Else a gc in between can recover the RemoteRef rr
    for chunk in d.chunks
        wait(chunk)
    end
    d
end

using CUDArt

function testnet(l::Layer)
    # Only copy what is needed for predict
    ll = Layer()
    isdefined(l,:w) && (ll.w = to_host(l.w))
    isdefined(l,:b) && (ll.b = to_host(l.b))
    isdefined(l,:f) && (ll.f = l.f)
    return ll
end

testnet(net::Net)=map(l->testnet(l), net)

