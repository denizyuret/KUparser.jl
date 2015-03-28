# arceager13.jl, Deniz Yuret, March 27, 2015
# Transition based greedy arc-eager parser, modified to have a single ROOT word.
# [GN13] Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# [N03]  Nivre, Joakim. An efficient algorithm for projective dependency parsing. IWPT 2003.
# [H13]  http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing

typealias ArcEager Parser{:ArcEager}

# In the arc-eager system (N03), a configuration c= (σ,β,A) consists of
# a stack σ, a buffer β, and a set A of dependency arcs.

# There are four types of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# REDUCE[(σ|s, β, A)] = (σ, β, A)
# RIGHT_lb[(σ|s, b|β, A)] = (σ|s|b, β, A∪{(s,lb,b)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A∪{(b,lb,s)})

# Moves are represented by integers 1..p.nmove
# They correspond to REDUCE,L1,R1,L2,R2,..,L[ndeps],R[ndeps],SHIFT

REDUCE(p::ArcEager)=1
SHIFT(p::ArcEager)=p.nmove
LMOVES(p::ArcEager)=(2:2:(p.nmove-2))
RMOVES(p::ArcEager)=(3:2:(p.nmove-1))
LMOVE(p::ArcEager, l::DepRel)=(l<<1)
RMOVE(p::ArcEager, l::DepRel)=(1+l<<1)
LABEL(p::ArcEager, m::Move)=convert(DepRel,m>>1)

# In GN13 the initial configuration has an empty stack, and a buffer
# with special symbol ROOT to the right of all the words at w[n+1], i.e.
# [][w1,w2,...,wn,ROOT]. The final state in GN13 is only the ROOT token
# in the buffer and nothing on the stack, i.e. [][ROOT].  The special
# ROOT token can have more than one child, thus the sentence does not
# always have a unique rootword (which we will fix).

# In this implementation we don't use an explicit ROOT token.  Our final
# state is the single root word at s0, and an empty buffer
# i.e. [rootword][].  The unique rootword is marked with head=0 and
# deprel=0.  The only legal first move, SHIFT, is performed at
# initialization so the initial configuration is [w1][w2,...,wn].  This
# ensures 2n-2 moves for each sentence.

init!(p::ArcEager)=(p.nmove=(2+p.ndeps<<1);move!(p,SHIFT(p)))

# GN13 has the following preconditions for moves: "There is a
# precondition on the RIGHT and SHIFT transitions to be legal only when
# b != ROOT (p.wptr <= p.nword), and for LEFT, RIGHT and REDUCE to be
# legal only when the stack is non-empty (p.sptr >= 1). Moreover, LEFT
# is only legal when s does not have a parent in A, and REDUCE when s
# does have a parent in A."

# GN13 ends up with multiple rootwords by shifting them to the stack and
# popping them with left moves when the only token left in the buffer is
# ROOT.  We can prevent multiple rootwords by:
# (1) avoid SHIFT on last buffer word unless the stack is empty.
# (2) avoid RIGHT on last buffer word if multiple headless words in stack.

shift_ok(p::ArcEager) = ((p.wptr < p.nword) || ((p.wptr == p.nword) && (p.sptr == 0)))
right_ok(p::ArcEager) = ((p.sptr >= 1) && ((p.wptr < p.nword) || ((p.wptr == p.nword) && (1==sum(p.head[p.stack[1:p.sptr]].==0)))))
reduce_ok(p::ArcEager) = ((p.sptr >= 1) && (p.head[p.stack[p.sptr]] != 0))
left_ok(p::ArcEager) = ((p.sptr >= 1) && (p.wptr <= p.nword) && (p.head[p.stack[p.sptr]] == 0))

# Proof: Buffer words never have heads, some stack words do.  As long as
# we have words in the buffer we can recover a single root.  Once the
# buffer is empty, nothing can go back into it.  With an empty buffer,
# REDUCE is the only possible move, (we don't allow left moves with an
# empty buffer) now new non-root arcs can be added.  So before we get to
# the empty buffer we need to make sure the stack will contain a single
# headless rootword.  Stack words are either headless or are next to
# their parents, i.e. a non-empty stack always contains at least one
# headless word.  SHIFTing the last word into a non-empty stack would
# add another, so can't do it.  RIGHT with the last word would not
# create another headless stack word, but we have to make sure there is
# only one before doing RIGHT on the last word.

function move!(p::ArcEager, m::Integer)
    @assert (1 <= m <= p.nmove) "Move $m is not supported"
    if m == SHIFT(p)
        @assert shift_ok(p)
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    elseif m == REDUCE(p)
        @assert reduce_ok(p)
        p.sptr -= 1
    elseif in(m, RMOVES(p))
        @assert right_ok(p)
        arc!(p, p.stack[p.sptr], p.wptr, LABEL(p,m))
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    else # in(m, LMOVES(p))
        @assert left_ok(p)
        arc!(p, p.wptr, p.stack[p.sptr], LABEL(p,m))
        p.sptr -= 1
    end
end # move!

# WolframAlpha suggested the following succinct form for anyvalidmoves(p).
# REDUCE if ((p.sptr >= 1) && (p.head[p.stack[p.sptr]] != 0))
# otherwise (p.wptr <= p.nword) is true
# SHIFT if (p.sptr == 0)
# otherwise (p.sptr >= 1) is true
# so (p.head[p.stack[p.sptr]] != 0) is false
# and LEFT is possible

anyvalidmoves(p::ArcEager)=((p.wptr <= p.nword) || ((p.sptr >= 1) && (p.head[p.stack[p.sptr]] != 0)))


# movecosts() counts gold arcs that become impossible after each move.
# A token starts life without any arcs in the buffer.  It moves to the
# head of the buffer (n0) with shift or right moves (each right ends
# with a shift).  Left deps are acquired first while at n0 using left
# moves (each left move ends with a reduce) possibly interspersed with
# other reduces (to get rid of s0 that already have heads).  The token
# moves to s0 with a right or shift.  Then rdeps are acquired while at
# s0 using right moves.  Head is acquired as n0 before rdeps (which
# moves the token to s0, so buffer words never have heads) or as s0
# after rdeps.

function movecosts(p::ArcEager, head::AbstractArray, deprel::AbstractArray, 
                   cost::Pvec=Array(Position,p.nmove))
    @assert (length(head) == p.nword)
    @assert (length(deprel) == p.nword)
    @assert (length(cost) == p.nmove)
    fill!(cost, Pinf)
    st = p.stack[1:p.sptr]
    n0 = p.wptr
    nw = p.nword
    s0 = (p.sptr >= 1 ? p.stack[p.sptr] : 0)
    s1 = (p.sptr > 1 ? p.stack[p.sptr-1] : 0)
    s0h = (s0 != 0 ? p.head[s0] : 0)
    n0h = (n0 <= nw ? head[n0] : 0)
    n0l = sum((head[st] .== n0) & (p.head[st] .== 0))
    s0r = sum(head[n0:end] .== s0)

    if shift_ok(p)                                              # SHIFT moves n0 to s
        cost[SHIFT(p)] = n0l + in(n0h, st)                      # n0 gets no more ldeps or lhead
    end
    if reduce_ok(p)                                             # REDUCE pops s0
        cost[REDUCE(p)] = s0r                                   # s0 gets no more rdeps
    end
    if right_ok(p)                                              # RIGHT adds (s0,n0) and shifts n0 to s
        rcost = (n0l + (n0h > n0) + (n0h == 0) +                # n0 gets no more ldeps, rhead, 0head,
                 in(n0h, p.stack[1:(p.sptr-1)]))                # or lhead<s0
        if n0h == s0                                            # if we have the correct head
            cost[RMOVES(p)] = rcost + 1                         # +1 for the wrong labels
            cost[RMOVE(p,deprel[n0])] -= 1                      # except for the correct label
        else
            cost[RMOVES(p)] = rcost                             # otherwise we are done
        end
    end
    if left_ok(p)                                               # LEFT  adds (n0,s0) and reduces s0
        lcost = (s0r + (head[s0] > n0) + (head[s0] == 0))       # s0 gets no more rdeps, rhead>n0, 0head
        if (head[s0] == n0)                                     # if we have the correct head
            cost[LMOVES(p)] = lcost+1                           # +1 for the wrong labels
            cost[LMOVE(p,deprel[s0])] -= 1                      # except for the correct label
        else
            cost[LMOVES(p)] = lcost                             # otherwise we are done
        end
    end
    return cost
end # movecosts

