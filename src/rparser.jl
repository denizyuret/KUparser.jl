# The random parser executes random valid moves to test the cost fn.
#
# The public interface for rparse takes the following arguments:
#
# pt::ParserType: ArcHybridR1, ArcEagerR1, ArcHybrid13, ArcEager13
# s::Sentence or c::Corpus: input sentence(s), a single parser is returned for s, a vector of parsers for c
# ncpu::Integer: (optional) performs parallel processing
# oracle: (kwarg) probability of following the oracle

function rparse{T<:Parser}(pt::Type{T}, s::Sentence; o...)
    p = pt(s)
    rparse(p, s; o...)
end

function rparse{T<:Parser}(pt::Type{T}, c::Corpus; o...)
    pa = map(pt, c)
    rparse(pa, c; o...)
end

function rparse(p::Parser, s::Sentence; oracle=0)
    c = Array(Cost, p.nmove)
    totalcost = 0
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        if rand() < oracle
            r = indmin(c)
        else
            f = find(c .< typemax(Cost))
            r = f[rand(1:length(f))]
        end
        totalcost += c[r]
        move!(p, r)
    end
    # truecost only makes sense if s has oracle parse
    if oracle > 0 && totalcost != truecost(p, s)
        error("Bad cost estimate")
    end
    return p
end

function rparse{T<:Parser}(pa::Vector{T}, c::Corpus; o...)
    for i=1:length(c); rparse(pa[i], c[i]; o...); end
    return pa
end

# Not tested
function rparse_dbg{T<:Parser}(pt::Type{T}, s::Sentence)
    dump(vcat([1:wcnt(s)]', s.head', s.deprel'))
    p = pt(s)
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

# Not tested
function rparse{T<:Parser}(pt::Type{T}, c::Corpus, ncpu::Integer; o...)
    (ncpu == 1) && return rparse(pt, c)
    @date Main.resetworkers(ncpu)
    sa = distribute(c)          # distributed sentence array
    pa = map(pt, sa)            # distributed parser array
    @sync for p in procs(sa)
        @spawnat p rparse(localpart(pa), localpart(sa), f; o...)
    end
    pa = convert(Vector{pt}, pa)
    @date Main.rmworkers()
    return pa
end

