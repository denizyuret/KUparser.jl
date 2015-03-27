# We define ArcHybrid, ArcEager etc. as immutable children of Parser
# methods: move!, movecosts, anyvalidmoves
# fields: state, nword, ndeps, nmove
abstract Parser

# Some utility types
@compat typealias Pval UInt8    # Type representing sentence position
@compat typealias Mval UInt8    # Type representing parser move
@compat typealias Dval UInt8    # Type representing dependency label
typealias Pvec AbstractVector{Pval}
typealias Dvec AbstractVector{Dval}
typealias Pmat AbstractMatrix{Pval}
const Pinf=typemax(Pval)
Pzeros(n::Integer...)=zeros(Pval, n...)
Dzeros(n::Integer...)=zeros(Dval, n...)

# Each parser has a ParserState field representing the stack, 
# buffer, set of arcs etc. i.e. all the mutable stuff.
type ParserState
    wptr::Pval    # index of first word in buffer
    sptr::Pval    # index of last word (top) of stack
    stack::Pvec   # 1xn vector for stack of indices
    head::Pvec    # 1xn vector of heads
    deprel::Dvec  # 1xn vector of dependency labels
    lcnt::Pvec    # lcnt(h): number of left deps for h
    rcnt::Pvec    # rcnt(h): number of right deps for h
    ldep::Pmat    # nxn matrix for left dependents
    rdep::Pmat    # nxn matrix for right dependents
    
    function ParserState(nword::Integer)
        @assert (nword <= (typemax(Pval)-1))    "nword > $(typemax(Pval)-1)"
        p = new(1, 0, Pzeros(nword),           # wptr, sptr, stack
                Pzeros(nword), Dzeros(nword),  # head, deprel
                Pzeros(nword), Pzeros(nword),  # lcnt, rcnt
                Pzeros(nword,nword), Pzeros(nword,nword)) # ldep, rdep
        return p
    end
end # ParserState

# arc! sets the head of d as h with label l

function arc!(p::ParserState, h::Pval, d::Pval, l::Dval)
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
function copy!(dst::ParserState, src::ParserState)
    dst.wptr = src.wptr
    dst.sptr = src.sptr
    copy!(dst.stack, src.stack)
    copy!(dst.head, src.head)
    copy!(dst.deprel, src.deprel)
    copy!(dst.lcnt, src.lcnt)
    copy!(dst.rcnt, src.rcnt)
    copy!(dst.ldep, src.ldep)
    copy!(dst.rdep, src.rdep)
end # copy!

# Moves are represented by integers 1..nmove, 0 is not valid
isshift(p,m)=(m==p.nmove)     # The last move is SHIFT
isreduce(p,m)=(m==p.nmove-1)  # The penultimate move is REDUCE in ArcEager
# The other moves are left/right dependency moves
# Labels are represented by integers 1..ndeps
const LEFT=1                  # Odd moves are LEFT
const RIGHT=0                 # Even moves are RIGHT
midx(d,l)=(l<<1-d)            # Move number from direction and label
mdep(m)=((m+1)>>1)            # Dependency label of a move
mdir(m)=(m&1)                 # Direction of a move

