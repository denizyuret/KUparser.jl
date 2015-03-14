# archybrid.jl, Deniz Yuret, July 7, 2014: Transition based greedy arc-hybrid parser based on:
# http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
# Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# Modified valid_moves to output a single root-child.

@compat typealias Pval UInt16   # Type representing sentence position
typealias Pvec AbstractVector{Pval}
typealias Pmat AbstractMatrix{Pval}
const Pinf=typemax(Pval)
pzeros(n::Integer...)=zeros(Pval, n...)

# TODO: think of other parser types with more moves.
# if we standardize move numbers (REDUCE=4), only NMOVE differs
# nmove can be a function.
# cost, valid, move are different
# arc and constructor and fields are the same:
# how to use that?
# also think of arceasy.

@compat typealias Move UInt8
const SHIFT=convert(Move,1)
const RIGHT=convert(Move,2)
const LEFT=convert(Move,3)

abstract Parser

type ArcHybrid <: Parser
    nword::Pval   # number of words in sentence
    nmove::Pval   # number of legal moves
    wptr::Pval    # index of first word in buffer
    sptr::Pval    # index of last word (top) of stack
    head::Pvec    # 1xn vector of heads
    stack::Pvec   # 1xn vector for stack of indices
    lcnt::Pvec    # lcnt(h): number of left deps for h
    rcnt::Pvec    # rcnt(h): number of right deps for h
    ldep::Pmat    # nxn matrix for left dependents
    rdep::Pmat    # nxn matrix for right dependents
    
    function ArcHybrid(n::Integer)
        p = new(n, 3, 1, 0, pzeros(n), pzeros(n), pzeros(n), pzeros(n), pzeros(n,n), pzeros(n,n))
        move!(p, SHIFT)
        return p
    end
end

# In the archybrid system:
# A token starts life without any arcs in the buffer.
# It becomes n0 after a number of shifts.
# n0 acquires ldeps using lefts.
# It becomes s0 using shift.
# s0 acquires rdeps using shift+right.
# Finally gets a head with left or right.

function move!(p::ArcHybrid, op::Move)
    if op == SHIFT
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    elseif op == RIGHT
        arc!(p, p.stack[p.sptr-1], p.stack[p.sptr])
        p.sptr -= 1
    elseif op == LEFT
        arc!(p, p.wptr, p.stack[p.sptr])
        p.sptr -= 1
    else
        error("Move $op is not supported")
    end
end

move!(p::ArcHybrid, op::Integer)=move!(p, convert(Move, op))

# Oracle cost counts gold arcs that become impossible after possible
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

function cost(p::ArcHybrid, gold::AbstractArray, c::Pvec=Array(Pval,p.nmove))
    gold = convert(Pvec, gold)
    @assert (length(gold) == p.nword)
    @assert (length(c) == p.nmove)
    fill!(c, Pinf)
    n0 = p.wptr
    if (n0 <= p.nword)      # shift is legal
        n0h = gold[n0]
        c[SHIFT] = sum(gold[p.stack[1:p.sptr]] .== n0) + 
        sum(p.stack[1:p.sptr-1] .== n0h) + 
        ((n0h == 0) && (p.sptr >= 1)) 
    end

    if (p.sptr >= 1)
        s0 = p.stack[p.sptr]
        s0h = gold[s0]
        s0b = sum(gold[n0:end] .== s0)

        if (n0 <= p.nword)    # left is legal
            c[LEFT] = s0b + 
            ((s0h > n0) || 
             ((p.sptr == 1) && (s0h == 0)) || 
             ((p.sptr >  1) && (s0h == p.stack[p.sptr-1])))
        end

        if (p.sptr >= 2)      # right is legal
            c[RIGHT] = s0b + (s0h >= n0)
        end
    end
    @assert (valid(p) == (c .< Pinf))
    return c
end # cost

function valid(p::ArcHybrid, v::AbstractVector{Bool}=Array(Bool, p.nmove))
    v[SHIFT] = (p.wptr <= p.nword)
    v[RIGHT] = (p.sptr >= 2)
    v[LEFT]  = ((p.sptr >= 1) && (p.wptr <= p.nword))
    return v
end # valid

function arc!(p::ArcHybrid, h::Pval, d::Pval)
    p.head[d] = h;
    if d < h
        p.lcnt[h] += 1
        p.ldep[h, p.lcnt[h]] = d
    else
        p.rcnt[h] += 1
        p.rdep[h, p.rcnt[h]] = d
    end # if
end # arc

import Base.copy!
function copy!(dst::ArcHybrid, src::ArcHybrid)
    assert(dst.nword == src.nword)
    assert(dst.nmove == src.nmove)
    dst.wptr = src.wptr
    dst.sptr = src.sptr
    copy!(dst.head, src.head)
    copy!(dst.stack, src.stack)
    copy!(dst.lcnt, src.lcnt)
    copy!(dst.rcnt, src.rcnt)
    copy!(dst.ldep, src.ldep)
    copy!(dst.rdep, src.rdep)
end

