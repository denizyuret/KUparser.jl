# The oracle parsers executes the moves that will lead to the best
# parse:

function oparse(s::Sentence, f::Features)
    (p, x, y, c) = initoparse(s, f)
    nx = 0
    while (v = valid(p); any(v))
        nx += 1
        features(p, s, f, sub(x,:,nx:nx))
        cost(p, s.head, c)
        bestmove = indmin(c)
        y[bestmove, nx] = one(eltype(y))
        move!(p, bestmove)
    end
    (p.head, x, y)
end

function oparse(c::Corpus, f::Features)
    p = map(s->oparse(s,f), c)
    h = map(z->z[1], p)
    x = hcat(map(z->z[2], p)...)
    y = hcat(map(z->z[3], p)...)
    (h, x, y)
end

function oparse(c::Corpus, f::Features, ncpu::Integer)
    assert(nworkers() >= ncpu)
    d = distproc(c, workers()[1:ncpu])
    @everywhere gc()
    @time p = pmap(procs(d)) do x
        oparse(localpart(d), f)
    end
    h = vcat(map(z->z[1], p)...)
    x = hcat(map(z->z[2], p)...)
    y = hcat(map(z->z[3], p)...)
    (h, x, y)
end

function initoparse(s::Sentence, f::Features)
    (ndims, nword) = size(s.wvec)
    xtype = eltype(s.wvec)
    xrows = flen(ndims, f)
    xcols = 2 * (nword - 1)
    p = ArcHybrid(nword)
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, p.nmove, xcols)
    c = Array(Pval,p.nmove)
    (p, x, y, c)
end
