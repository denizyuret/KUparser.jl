# We define Parser{T} as a parametric type where T is :ArcHybrid,
# :ArcEager etc. to allow specialization of methods.  Thanks to
# julia-users members Toivo Henningsson and Simon Danisch for suggesting
# this design.

typealias ParserType Symbol

# Parser methods: arc!, copy!
# Parser{T} methods: init!, move!, movecosts, anyvalidmoves

# Some utility types
@compat typealias Position UInt8
@compat typealias DepRel UInt8
@compat typealias Move Integer
typealias Pvec AbstractVector{Position}
typealias Dvec AbstractVector{DepRel}
typealias Pmat AbstractMatrix{Position}
const Pinf=typemax(Position)
Pzeros(n::Integer...)=zeros(Position, n...)
Dzeros(n::Integer...)=zeros(DepRel, n...)

# Each parser has a ParserState field representing the stack, 
# buffer, set of arcs etc. i.e. all the mutable stuff.
type Parser{T}
    nword::Position       # number of words in sentence
    ndeps::DepRel         # number of dependency labels (excluding ROOT)
    nmove::Move           # number of possible moves (set by init!)
    wptr::Position        # index of first word in buffer
    sptr::Position        # index of last word (top) of stack
    stack::Pvec           # 1xn vector for stack of indices
    head::Pvec            # 1xn vector of heads
    deprel::Dvec          # 1xn vector of dependency labels
    lcnt::Pvec            # lcnt(h): number of left deps for h
    rcnt::Pvec            # rcnt(h): number of right deps for h
    ldep::Pmat            # nxn matrix for left dependents
    rdep::Pmat            # nxn matrix for right dependents
    
    function Parser(nword::Integer, ndeps::Integer)
        @assert (nword < typemax(Position)) "nword >= $(typemax(Position))"
        @assert (ndeps < typemax(DepRel)) "ndeps >= $(typemax(DepRel))"
        p = new(nword, ndeps, 0,               # nword, ndeps, nmove
                1, 0, Pzeros(nword),           # wptr, sptr, stack
                Pzeros(nword), Dzeros(nword),  # head, deprel
                Pzeros(nword), Pzeros(nword),  # lcnt, rcnt
                Pzeros(nword,nword), Pzeros(nword,nword)) # ldep, rdep
        init!(p)
        return p
    end
end # ParserState

# arc! sets the head of d as h with label l

function arc!(p::Parser, h::Position, d::Position, l::DepRel)
    p.head[d] = h
    p.deprel[d] = l
    if d < h
        p.lcnt[h] += 1
        p.ldep[h, p.lcnt[h]] = d
    else
        p.rcnt[h] += 1
        p.rdep[h, p.rcnt[h]] = d
    end # if
end # arc!

# Extend copy! to copy ParserStates
import Base.copy!
function copy!(dst::Parser, src::Parser)
    @assert dst.nword == src.nword
    @assert dst.ndeps == src.ndeps
    @assert dst.nmove == src.nmove
    dst.wptr = src.wptr
    dst.sptr = src.sptr
    copy!(dst.stack, src.stack)
    copy!(dst.head, src.head)
    copy!(dst.deprel, src.deprel)
    copy!(dst.lcnt, src.lcnt)
    copy!(dst.rcnt, src.rcnt)
    copy!(dst.ldep, src.ldep)
    copy!(dst.rdep, src.rdep)
    dst
end # copy!

import Base.isequal
function isequal(a::Parser, b::Parser)
    all(map(isequal, map(n->a.(n), fieldnames(a)), map(n->b.(n), fieldnames(a))))
end

