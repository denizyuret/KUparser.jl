# The greedy transition based parser parses the sentence using the
# following steps:

function gparse(s::Sentence, n::Net, f::Features; pred::Bool=true, feat::Bool=true)
    (ndims, nword) = size(s.wvec)
    p = ArcHybrid(nword)
    x = Array(eltype(s.wvec), flen(ndims, f), 1)
    y = Array(eltype(x), p.nmove, 1)
    while (v = valid(p); any(v))
        feat ? features(p, s, f, x) : rand!(x)
        pred ? predict(n, x, y) : rand!(y)
        y[!v,:] = -Inf
        move!(p, indmax(y))
    end
    p.head
end

# We parse a corpus using a for loop or map:

function gparse(c::Corpus, n::Net, f::Features; args...)
    map(s->gparse(s,n,f;args...), c)
    # for s in c; gparse(s,n,f); end # not faster than map
end

# There are two opportunities for parallelism:
# 1. We process multiple sentences to minibatch net input.
#    This speeds up predict.

function gparse(corpus::Corpus, net::Net, fmat::Features, batch::Integer; feat::Bool=true)
    # determine dimensions
    nsent = length(corpus)
    nword = 0; for s in corpus; nword += wcnt(s); end
    xcols = 2 * (nword - nsent)
    wvec1 = corpus[1].wvec
    wdims = size(wvec1,1)
    xrows = flen(wdims, fmat)
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
                feat ? features(p[s], corpus[s], fmat, sub(x, :, idx + i)) : rand!(sub(x,:,idx+i))
            end

            # Next predict y in bulk
            (xxcols != nvalid) && (xxcols = nvalid; KUnet.free(xx); xx = similar(net[1].w, (xrows, xxcols)))
            copy!(xx, (1:xrows, 1:nvalid), x, (1:xrows, idx+1:idx+nvalid))
            yy = KUnet.forw(net, xx, false)
            copy!(y, (1:yrows, idx+1:idx+nvalid), yy, (1:yrows, 1:nvalid))

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
    return (h, x, y, z)
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

function gparse(corpus::Corpus, net::Net, fmat::Features, batch::Integer, ncpu::Integer)
    assert(nworkers() >= ncpu)
    d = distribute(corpus, workers()[1:ncpu])
    n = copy(net, :cpu)
    @everywhere gc()
    @time p = pmap(procs(d)) do x
        gparse(localpart(d), copy(n, :gpu), fmat, batch)
    end
    pmerge(p)
end

function pmerge(p)
    (h, x, y, z) = p[1]
    for i=2:length(p)
        (h2,x2,y2,z2) = p[i]
        h = append!(h, h2)
        x = [x x2]
        y = [y y2]
        z = [z z2]
    end
    (h, x, y, z)
end

import Base.distribute

function distribute(a::AbstractArray, procs)
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
