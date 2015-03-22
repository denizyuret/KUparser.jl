# archybrid.jl, Deniz Yuret, July 7, 2014: Transition based greedy arc-hybrid parser based on:
# http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
# Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# Modified validmoves to output a single root-child.

# TODO: think of other parser types with more moves.
# if we standardize move numbers (REDUCE=4), only NMOVE differs
# nmove can be a function.
# cost, valid, move are different
# arc and constructor and fields are the same:
# how to use that?
# also think of arceasy.

@compat typealias Pval UInt8    # Type representing sentence position
@compat typealias Mval UInt8    # Type representing parser move
@compat typealias Dval UInt8    # Type representing dependency label
typealias Dvec AbstractVector{Dval}
typealias Mvec AbstractVector{Mval}
typealias Pvec AbstractVector{Pval}
typealias Pmat AbstractMatrix{Pval}
const Pinf=typemax(Pval)
pzeros(n::Integer...)=zeros(Pval, n...)
movedep(m::Mval)=(m>>1)         # 1..ndep for m>1
movedir(m::Mval)=(m&1)          # 0=LEFT 1=RIGHT for m>1
const SHIFT=convert(Mval,1)     # m=0 illegal, m=1 shift, m=2..nmove L/R moves
const LEFT=convert(Mval,0)
const RIGHT=convert(Mval,1)

abstract Parser

type ArcHybrid <: Parser
    nword::Pval   # number of words in sentence
    ndeps::Dval   # number of dependency labels
    nmove::Mval   # number of legal moves
    wptr::Pval    # index of first word in buffer
    sptr::Pval    # index of last word (top) of stack
    stack::Pvec   # 1xn vector for stack of indices
    head::Pvec    # 1xn vector of heads
    deps::Dvec    # 1xn vector of dependency labels
    lcnt::Pvec    # lcnt(h): number of left deps for h
    rcnt::Pvec    # rcnt(h): number of right deps for h
    ldep::Pmat    # nxn matrix for left dependents
    rdep::Pmat    # nxn matrix for right dependents
    
    function ArcHybrid(nword::Integer, ndeps::Integer)
        @assert (nword <= (typemax(Pval)-1))    "nword > $(typemax(Pval)-1)"
        @assert (ndeps <= (typemax(Mval)-1)>>1) "ndeps > $((typemax(Mval)-1)>>1)"
        p = new(nword, ndeps, 1+2*ndeps,       # nword, ndeps, nmove
                1, 0, pzeros(nword),           # wptr, sptr, stack
                pzeros(nword), dzeros(nword),  # head, deps
                pzeros(nword), pzeros(nword),  # lcnt, rcnt
                pzeros(nword,nword), pzeros(nword,nword)) # ldep, rdep
        move!(p, SHIFT)
        return p
    end
end # ArcHybrid

import Base.copy!
function copy!(dst::ArcHybrid, src::ArcHybrid)
    assert(dst.nword == src.nword)
    assert(dst.ndeps == src.ndeps)
    assert(dst.nmove == src.nmove)
    dst.wptr = src.wptr
    dst.sptr = src.sptr
    copy!(dst.stack, src.stack)
    copy!(dst.head, src.head)
    copy!(dst.deps, src.deps)
    copy!(dst.lcnt, src.lcnt)
    copy!(dst.rcnt, src.rcnt)
    copy!(dst.ldep, src.ldep)
    copy!(dst.rdep, src.rdep)
end # copy!

# arc! sets the head of d as h with label l

function arc!(p::ArcHybrid, h::Pval, d::Pval, l::Dval)
    p.head[d] = h
    p.deps[d] = l
    if d < h
        p.lcnt[h] += 1
        p.ldep[h, p.lcnt[h]] = d
    else
        p.rcnt[h] += 1
        p.rdep[h, p.rcnt[h]] = d
    end # if
end # arc!

# In the archybrid system:
# A token starts life without any arcs in the buffer.
# It becomes n0 after a number of shifts.
# n0 acquires ldeps using lefts.
# It becomes s0 using shift.
# s0 acquires rdeps using shift+right.
# Finally gets a head with left or right.

function move!(p::ArcHybrid, op::Integer)
    @assert (op > 0 && op <= p.nmove) "Move $op is not supported"
    op = convert(Mval, op)
    if op == SHIFT
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    elseif movedir(op) == LEFT
        arc!(p, p.wptr, p.stack[p.sptr], movedep(op))
        p.sptr -= 1
    else # if movedir(op) == RIGHT
        arc!(p, p.stack[p.sptr-1], p.stack[p.sptr], movedep(op))
        p.sptr -= 1
    end
end # move!

# movecosts() counts gold arcs that become impossible after possible
# transitions.  Tokens start their lifecycle in the buffer without
# links.  They move to the top of the buffer (n0) with SHIFT moves.
# There they acquire left dependents using LEFT moves.  After that a
# single SHIFT moves them to the top of the stack (s0).  There they
# acquire right dependents with SHIFT-RIGHT pairs.  Finally from s0
# they acquire a head with a LEFT or RIGHT move.  Any token from the
# buffer may become the head but only s1 from the stack may become a
# left head.  The parser terminates with a single word at s0 whose
# head is ROOT.
#
# 1. SHIFT moves n0 to s0: n0 cannot acquire left dependents after a
# shift.  Also it can no longer get a head from the stack to the left
# of s0 or get a root head if there is s0: (0+s\s0,n0) + (n0,s)
#
# 2. RIGHT adds (s1,s0): s0 cannot acquire a head or dependent from
# the buffer after right: (s0,b) + (b,s0)
#
# 3. LEFT adds (n0,s0): s0 cannot acquire s1 or 0 (if there is no s1)
# or ni (i>0) as head.  It also cannot acquire any more right
# children: (s0,b) + (b\n0,s0) + (s1 or 0,s0)

function movecosts(p::ArcHybrid, head::AbstractArray, deps::AbstractArray, cost::Pvec=Array(Pval,p.nmove))
    head = convert(Pvec, head)
    deps = convert(Dvec, deps)
    @assert (length(head) == p.nword)
    @assert (length(deps) == p.nword)
    @assert (length(cost) == p.nmove)
    fill!(cost, Pinf)
    n0 = p.wptr                                                 # n0 is top of buffer
    if (n0 <= p.nword)                                          # shift is legal moving n0 to s0
        n0h = head[n0]                                          # n0h is the actual head of n0
        cost[SHIFT] = (sum(head[p.stack[1:p.sptr]] .== n0) +	# no more left dependents for n0
                       sum(p.stack[1:p.sptr-1] .== n0h) +       # no heads to the left of s0 for n0
                       ((n0h == 0) && (p.sptr >= 1)))           # no root head for n0 if there is s0
    end
    if (p.sptr >= 1)                                            # left/right valid if stack nonempty
        s0 = p.stack[p.sptr]                                    # s0 is top of stack
        s0h = head[s0]                                          # s0h is the actual head of s0
        s0b = sum(head[n0:end] .== s0)                          # num buffer words whose head is s0

        if (n0 <= p.nword)                                      # left is legal, making n0 head of s0
            leftcost = (s0b +                                   # no more right children for s0
                        ((s0h > n0) ||                          # no heads to the right of n0 for s0
                         ((p.sptr == 1) && (s0h == 0)) ||       # no root head for s0 if alone
                         ((p.sptr > 1) &&                       # no more s1 for head of s0
                          (s0h == p.stack[p.sptr-1]))))
            if (s0h != n0)
                cost[2:2:p.nmove] = leftcost                    # even numbered moves >= 2 are left moves
            else
                cost[2:2:p.nmove] = leftcost + 1                # +1 for the wrong labels
                cost[deps[s0]<<1] -= 1                          # except for the correct label
            end
        end
        if (p.sptr >= 2)                                        # right is legal making s1 head of s0
            s1 = p.stack[p.sptr-1]                              # s1 is the stack element before s0
            rightcost = s0b + (s0h >= n0)                       # no more right head or dependent for s0
            if (s0h != s1)
                cost[3:2:p.nmove] = rightcost                   # odd numbered moves >= 3 are right moves
            else
                cost[3:2:p.nmove] = rightcost + 1               # +1 for the wrong labels
                cost[deps[s0]<<1+1] -= 1                        # except for the correct label
            end
        end
    end
    @assert (validmoves(p) == (cost .< Pinf))
    return cost
end # movecosts

function validmoves(p::ArcHybrid, v::AbstractVector{Bool}=Array(Bool, p.nmove))
    v[SHIFT] = (p.wptr <= p.nword)
    right_ok = (p.sptr >= 2)
    left_ok = ((p.sptr >= 1) && (p.wptr <= p.nword))
    for m=2:2:p.nmove; v[m] = left_ok; end
    for m=3:2:p.nmove; v[m] = right_ok; end
    return v
end # validmoves

