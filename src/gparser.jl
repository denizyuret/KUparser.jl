# The greedy transition based parser parses the sentence using the
# following steps:

function gparse(sent::Sentence, net::Net, feats::Features, ndeps::Integer)
    (parser,x,y,cost,score) = initgparse(sent,net,feats,ndeps)
    nx = 0
    while anyvalidmoves(parser)
        nx += 1; xx = sub(x,:,nx:nx)
        movecosts(parser, sent.head, sent.deprel, cost)
        y[indmin(cost), nx] = one(eltype(y))
        features(parser, sent, feats, xx)
        predict(net, xx, score)
        score[cost .== Pinf,:] = -Inf
        move!(parser, indmax(score))
    end
    @assert nx == size(x,2)
    (parser, x, y)
end

# We can parse a corpus using map:

function gparse(corpus::Corpus, net::Net, feats::Features, ndeps::Integer)
    pxy = map(s->gparse(s,net,feats,ndeps), corpus)
    p = map(z->z[1], pxy)
    x = hcat(map(z->z[2], pxy)...)
    y = hcat(map(z->z[3], pxy)...)
    (p, x, y)
end

# Most of the time gets spent on predict and features.
# There are two opportunities for parallelism:
# 1. We process multiple sentences to minibatch net input.
#    This speeds up predict.

function gparse(corpus::Corpus, net::Net, feats::Features, ndeps::Integer, nbatch::Integer)
    (nbatch == 0 || nbatch > length(corpus)) && (nbatch = length(corpus))
    (p,x,y,cost,score) = initgparse(corpus, net, feats, ndeps, nbatch)
    nx = 0
    for s1 = 1:nbatch:length(corpus)
        s2 = min(length(corpus), s1+nbatch-1)
        while true
            nx2 = nx
            for s=s1:s2
                anyvalidmoves(p[s]) || continue
                features(p[s], corpus[s], feats, sub(x, :, (nx2+=1)))
            end
            (nx2 == nx) && break
            KUnet.predict(net, sub(x, :, nx+1:nx2), sub(score, :, nx+1:nx2))
            for s=s1:s2
                anyvalidmoves(p[s]) || continue
                nx += 1
                movecosts(p[s], corpus[s].head, corpus[s].deprel, cost)
                y[indmin(cost), nx] = one(eltype(y))
                score[cost .== Pinf, nx] = -Inf
                move!(p[s], indmax(sub(score,:,nx)))
            end
            @assert nx == nx2
        end # while true
    end # for s1 = 1:nbatch:length(corpus)
    return (p, sub(x,:,1:nx), sub(y,:,1:nx))
end

# 2. We do multiple batches in parallel to utilize CPU cores.
#    This speeds up features.

function gparse(corpus::Corpus, net::Net, feats::Features, ndeps::Integer, nbatch::Integer, ncpu::Integer)
    Main.resetworkers(ncpu)
    d = distproc(corpus, workers()[1:ncpu])
    net = testnet(net)
    pxy = pmap(procs(d)) do x
        gparse(localpart(d), copy(net, :gpu), feats, ndeps, nbatch)
    end
    rmprocs(workers()); sleep(2)
    p = vcat(map(z->z[1], pxy)...)
    x = hcat(map(z->z[2], pxy)...)
    y = hcat(map(z->z[3], pxy)...)
    (p, x, y)
end

function initgparse(sent::Sentence, net::Net, feats::Features, ndeps::Integer)
    (ndims, nword) = size(sent.wvec)
    xtype = eltype(net[1].w)
    xrows = flen(ndims, feats)
    xcols = 2 * (nword - 1)
    p = ArcHybrid(nword,ndeps)
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, p.nmove, xcols)
    cost = Array(Pval,p.nmove)
    score = Array(xtype, p.nmove, 1)
    (p, x, y, cost, score)
end

function initgparse(corpus::Corpus, net::Net, feats::Features, ndeps::Integer, nbatch::Integer)
    p = map(s->ArcHybrid(wcnt(s),ndeps), corpus)
    nsent = length(corpus)
    nword = sum(wcnt, corpus)
    xcols = 2 * (nword - nsent)
    wvec1 = corpus[1].wvec
    wdims = size(wvec1,1)
    xrows = flen(wdims, feats)  # TODO: flen and other features.jl fns may need ndeps!
    xtype = eltype(wvec1)
    yrows = p[1].nmove
    x = Array(xtype, xrows, xcols)      # feature vectors
    y = zeros(xtype, yrows, xcols)      # mincost moves, 1-of-k encoding
    cost = Array(Pval, yrows)
    score = Array(xtype, yrows, xcols)  # predicted move scores
    (p,x,y,cost,score)
end