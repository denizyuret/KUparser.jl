# The random parser executes random valid moves to test the cost fn.

function rparse(s::Sentence, ndeps::Integer)
    (p, c, v) = initrparse(s, ndeps)
    cost = 0
    while any(validmoves(p,v))
        movecosts(p, s.head, s.deprel, c)
        if rand() < 0.1
            r = rand(find(v))
        else
            r = indmin(c)
        end
        cost += c[r]
        move!(p, r)
    end
    @assert cost == sum((p.head .!= s.head) | (p.deps .!= s.deprel))
    (wcnt(s),cost,p.head, p.deps)
end

function rparse(c::Corpus, ndeps::Integer)
    p = map(s->rparse(s,ndeps), c)
    # TODO: should we merge here?
    # h = map(z->z[1], p)
    # x = hcat(map(z->z[2], p)...)
    # y = hcat(map(z->z[3], p)...)
    # (h, x, y)
end

function rparse(c::Corpus, ndeps::Integer, ncpu::Integer)
    # TODO: should we initialize cpus here?
    assert(nworkers() >= ncpu)
    d = distproc(c, workers()[1:ncpu])
    @everywhere gc()
    p = pmap(procs(d)) do x
        rparse(localpart(d), ndeps)
    end
    # d=nothing; @everywhere gc()
    # h = vcat(map(z->z[1], p)...)
    # x = hcat(map(z->z[2], p)...)
    # y = hcat(map(z->z[3], p)...)
    # p=nothing; @everywhere gc()
    # (h, x, y)
end

function initrparse(s::Sentence, ndeps::Integer)
    (ndims, nword) = size(s.wvec)
    p = ArcHybrid(nword, ndeps)
    c = Array(Pval,p.nmove)
    v = Array(Bool,p.nmove)
    (p, c, v)
end
