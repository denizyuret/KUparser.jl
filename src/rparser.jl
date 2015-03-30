# The random parser executes random valid moves to test the cost fn.

function rparse(p::Parser, s::Sentence, ndeps::Integer)
    c = Array(Position, p.nmove)
    totalcost = 0
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        if rand() < 0.1
            f = find(c .< Pinf)
            r = f[rand(1:length(f))]
        else
            r = indmin(c)
        end
        totalcost += c[r]
        move!(p, r)
    end
    @assert totalcost == truecost(p, s)
    return p
end

function rparse{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer)
    p = pt(wcnt(s), ndeps)
    rparse(p, s, ndeps)
end

function rparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer)
    pa = map(s->pt(wcnt(s), ndeps), c)
    rparse(pa, c, ndeps)
end

function rparse{T<:Parser}(pa::Vector{T}, c::Corpus, ndeps::Integer)
    for i=1:length(c); rparse(pa[i], c[i], ndeps); end
    return pa
end

function rparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, ncpu::Integer)
    @date Main.resetworkers(ncpu)
    sa = distribute(c)                                  # distributed sentence array
    pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
    @sync for p in procs(sa)
        @spawnat p oparse(localpart(pa), localpart(sa), f, ndeps)
    end
    pa = convert(Vector{pt}, pa)
    @date Main.rmworkers()
    return pa
end

function rparse_dbg{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer)
    dump(vcat([1:wcnt(s)]', s.head', s.deprel'))
    p = pt(wcnt(s), ndeps)
    c = Array(Position, p.nmove)
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
        println("$((r,int(c[r]),cost)) $(int(p.stack[1:p.sptr])) $(p.wptr) $(int(c))")
    end
    @assert cost == truecost(p,s)
    return p
end

function truecost(p::Parser, s::Sentence)
    cost = 0
    for i=1:p.nword
        ((p.head[i] != s.head[i]) || (p.deprel[i] != s.deprel[i])) && (cost += 1)
    end
    return cost
end

