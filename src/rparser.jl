# The random parser executes random valid moves to test the cost fn.

function rparse(pt::ParserType, s::Sentence, ndeps::Integer)
    (p, c) = initrparse(pt, s, ndeps)
    cost = 0
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        if rand() < 0.1
            r = rand(find(c .< Pinf))
        else
            r = indmin(c)
        end
        cost += c[r]
        move!(p, r)
    end
    @assert cost == sum((p.head .!= s.head) | (p.deprel .!= s.deprel))
    (wcnt(s),cost,p.head, p.deprel)
end

function rparse(pt::ParserType, c::Corpus, ndeps::Integer)
    p = map(s->rparse(pt,s,ndeps), c)
    # TODO: should we merge here?
    # h = map(z->z[1], p)
    # x = hcat(map(z->z[2], p)...)
    # y = hcat(map(z->z[3], p)...)
    # (h, x, y)
end

function rparse(pt::ParserType, c::Corpus, ndeps::Integer, ncpu::Integer)
    Main.resetworkers(ncpu)
    d = distproc(c, workers()[1:ncpu])
    @everywhere gc()
    p = pmap(procs(d)) do x
        rparse(pt, localpart(d), ndeps)
    end
    Main.rmworkers()
    # d=nothing; @everywhere gc()
    # h = vcat(map(z->z[1], p)...)
    # x = hcat(map(z->z[2], p)...)
    # y = hcat(map(z->z[3], p)...)
    # p=nothing; @everywhere gc()
    # (h, x, y)
end

function initrparse(pt::ParserType, s::Sentence, ndeps::Integer)
    (ndims, nword) = size(s.wvec)
    p = Parser{pt}(nword, ndeps)
    c = Array(Position,nmoves(p))
    (p, c)
end
