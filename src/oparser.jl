# The oracle parser executes the moves that will lead to the best parse:

function oparse(pt::ParserType, s::Sentence, f::Features, ndeps::Integer)
    (p, x, y, c) = initoparse(pt, s, f, ndeps)
    nx = 0; cost = 0
    while anyvalidmoves(p)
        nx += 1
        features(p, s, f, sub(x,:,nx:nx))
        movecosts(p, s.head, s.deprel, c)
        (bestcost,bestmove) = findmin(c)
        cost += bestcost
        y[bestmove, nx] = one(eltype(y))
        move!(p, bestmove)
    end
    @assert nx == size(x,2)
    @assert cost == sum((p.head .!= s.head) | (p.deprel .!= s.deprel))
    (p, x, y)
end

function oparse(pt::ParserType, c::Corpus, f::Features, ndeps::Integer)
    pxy = map(s->oparse(pt,s,f,ndeps), c)
    p = map(z->z[1], pxy)
    x = hcat(map(z->z[2], pxy)...)
    y = hcat(map(z->z[3], pxy)...)
    (p, x, y)
end

function oparse(pt::ParserType, c::Corpus, f::Features, ndeps::Integer, ncpu::Integer)
    Main.resetworkers(ncpu)
    d = distproc(c, workers()[1:ncpu])
    pxy = pmap(procs(d)) do x
        oparse(pt, localpart(d), f, ndeps)
    end
    Main.rmworkers()
    p = vcat(map(z->z[1], pxy)...)
    x = hcat(map(z->z[2], pxy)...)
    y = hcat(map(z->z[3], pxy)...)
    (p, x, y)
end

function initoparse(pt::ParserType, s::Sentence, f::Features, ndeps::Integer)
    p = Parser{pt}(wcnt(s), ndeps)
    xtype = wtype(s)
    xrows = flen(p, s, f)
    xcols = 2 * (p.nword - 1)
    nmove = nmoves(p)
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, nmove, xcols)
    c = Array(Position,nmove)
    (p, x, y, c)
end
