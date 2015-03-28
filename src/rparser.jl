# The random parser executes random valid moves to test the cost fn.

function rparse(pt::ParserType, s::Sentence, ndeps::Integer)
    (p, c) = initrparse(pt, s, ndeps)
    cost = 0
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        if rand() < 0.1
            f = find(c .< Pinf)
            r = f[rand(1:length(f))]
        else
            r = indmin(c)
        end
        cost += c[r]
        move!(p, r)
    end
    @assert cost == sum((p.head .!= s.head) | (p.deprel .!= s.deprel))
    return p
end

function rparse(pt::ParserType, c::Corpus, ndeps::Integer)
    map(s->rparse(pt,s,ndeps), c)
end

function rparse(pt::ParserType, c::Corpus, ndeps::Integer, ncpu::Integer)
    Main.resetworkers(ncpu)
    d = distproc(c, workers()[1:ncpu])
    @everywhere gc()
    p = pmap(procs(d)) do x
        rparse(pt, localpart(d), ndeps)
    end
    Main.rmworkers()
    return p
end

function initrparse(pt::ParserType, s::Sentence, ndeps::Integer)
    (ndims, nword) = size(s.wvec)
    p = Parser{pt}(nword, ndeps)
    c = Array(Position, p.nmove)
    (p, c)
end
