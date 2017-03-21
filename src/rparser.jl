# The random parser executes random valid moves to test the cost fn.
#
# The public interface for rparse takes the following arguments:
#
# pt::ParserType: ArcHybridR1, ArcEagerR1, ArcHybrid13, ArcEager13
# s::Sentence or c::Corpus: input sentence(s), a single parser is returned for s, a vector of parsers for c
# ndeps::Integer: number of dependency types
# ncpu::Integer: (optional) performs parallel processing

function rparse{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer)
    p = pt(wcnt(s), ndeps)
    rparse(p, s, ndeps)
end

function rparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer)
    pa = map(s->pt(wcnt(s), ndeps), c)
    rparse(pa, c, ndeps)
end

function rparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, ncpu::Integer)
    (ncpu == 1) && return rparse(pt, c, ndeps)
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

function rparse(p::Parser, s::Sentence, ndeps::Integer)
    c = Array(Position, p.nmove)
    totalcost = 0
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        if rand() < 0.1
            f = find(c .< typemax(Cost))
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

function rparse{T<:Parser}(pa::Vector{T}, c::Corpus, ndeps::Integer)
    for i=1:length(c); rparse(pa[i], c[i], ndeps); end
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
            f = find(c .< typemax(Cost))
            r = f[rand(1:length(f))]
        else
            r = indmin(c)
        end
        cost += c[r]
        move!(p, r)
        println("$((r,Int(c[r]),cost)) $(Int(p.stack[1:p.sptr])) $(p.wptr) $(Int(c))")
    end
    @assert cost == truecost(p,s)
    return p
end

