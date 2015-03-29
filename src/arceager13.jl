# arceager13.jl, Deniz Yuret, March 28, 2015
# Transition based greedy arc-eager parser based on:
# [GN13] Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# [N03]  Nivre, Joakim. An efficient algorithm for projective dependency parsing. IWPT 2003.
# [H13]  http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing

typealias ArcEager13 Parser{:ArcEager13}

# In the arc-eager system (N03), a configuration c= (σ,β,A) consists of
# a stack σ, a buffer β, and a set A of dependency arcs.

# There are four types of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# REDUCE[(σ|s, β, A)] = (σ, β, A)
# RIGHT_lb[(σ|s, b|β, A)] = (σ|s|b, β, A∪{(s,lb,b)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A∪{(b,lb,s)})

# Moves are represented by integers 1..p.nmove
# They correspond to REDUCE,L1,R1,L2,R2,..,L[ndeps],R[ndeps],SHIFT

@inline REDUCE(p::ArcEager13)=1
@inline SHIFT(p::ArcEager13)=p.nmove
@inline LMOVES(p::ArcEager13)=(2:2:(p.nmove-2))
@inline RMOVES(p::ArcEager13)=(3:2:(p.nmove-1))

# Dependency labels (deprel) are represented by integers 1..p.ndeps
# The special ROOT deprel is represented by 0.  We use REDUCE for the
# LEFT move from the ROOT token.
@inline LMOVE(p::ArcEager13, l::DepRel)=(l<<1)
@inline RMOVE(p::ArcEager13, l::DepRel)=(1+l<<1)
@inline LABEL(p::ArcEager13, m::Move)=convert(DepRel,m>>1)

# In GN13 the initial configuration has an empty stack, and a buffer
# with special symbol ROOT to the right of all the words at w[n+1], i.e.
# [][w1,w2,...,wn,ROOT]. The final state in GN13 is only the ROOT token
# in the buffer and nothing on the stack, i.e. [][ROOT].  The special
# ROOT token can have more than one child, thus the sentence does not
# always have a unique rootword.

# The only legal first move is SHIFT.  The only legal last move is
# REDUCE (representing ROOT LEFT).  We perform the first SHIFT during
# initialization so our initial state is [w1][w2,...,wn,ROOT].  We don't
# perform the last REDUCE so our final state is [wi][ROOT].  This
# ensures 2n-2 moves for each sentence.

init!(p::ArcEager13)=(p.nmove=(2+p.ndeps<<1);move!(p,SHIFT(p)))

# GN13 has the following preconditions for moves: "There is a
# precondition on the RIGHT and SHIFT transitions to be legal only when
# b != ROOT (p.wptr <= p.nword), and for LEFT, RIGHT and REDUCE to be
# legal only when the stack is non-empty (p.sptr > 0). Moreover, LEFT
# is only legal when s does not have a parent in A, and REDUCE when s
# does have a parent in A."  Since we implement the ROOT-LEFT as a
# REDUCE, our LEFT gets an extra contition (p.wptr <= p.nword) and
# REDUCE gets an extra option.

@inline BUFFER(p::ArcEager13)=(p.wptr <= p.nword)
@inline STACK(p::ArcEager13)=(p.sptr > 0)
@inline S0HEAD(p::ArcEager13)=(p.head[p.stack[p.sptr]]!=0)
@inline SHIFTOK(p::ArcEager13)=BUFFER(p)
@inline RIGHTOK(p::ArcEager13)=(STACK(p) && BUFFER(p))
@inline LEFTOK(p::ArcEager13)=(STACK(p) && BUFFER(p) && !S0HEAD(p))
@inline REDUCEOK(p::ArcEager13)=(STACK(p) && (S0HEAD(p) || (!BUFFER(p) && (p.sptr > 1))))

@inline anyvalidmoves(p::ArcEager13)=(BUFFER(p) || (STACK(p) && (S0HEAD(p) || (p.sptr > 1))))


function move!(p::ArcEager13, m::Integer)
    @assert (1 <= m <= p.nmove) "Move $m is not supported"
    if m == SHIFT(p)
        @assert SHIFTOK(p)
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    elseif m == REDUCE(p)
        @assert REDUCEOK(p)
        p.sptr -= 1
    elseif in(m, RMOVES(p))
        @assert RIGHTOK(p)
        arc!(p, p.stack[p.sptr], p.wptr, LABEL(p,m))
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    else # in(m, LMOVES(p))
        @assert LEFTOK(p)
        arc!(p, p.wptr, p.stack[p.sptr], LABEL(p,m))
        p.sptr -= 1
    end
end # move!


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

function movecosts(p::ArcEager13, head::AbstractArray, deprel::AbstractArray, 
                   cost::Pvec=Array(Position,p.nmove))
    @assert (length(head) == p.nword)
    @assert (length(deprel) == p.nword)
    @assert (length(cost) == p.nmove)
    fill!(cost, Pinf)
    n0 = p.wptr
    nw = p.nword
    s0 = (p.sptr >= 1 ? p.stack[p.sptr] : 0)
    s1 = (p.sptr > 1 ? p.stack[p.sptr-1] : 0)
    s0h = (s0 != 0 ? p.head[s0] : 0)
    n0h = (n0 <= nw ? head[n0] : 0)
    n0l = 0; n0 <= nw && (for i=1:p.sptr; si=p.stack[i]; head[si]==n0 && p.head[si]==0 && (n0l+=1); end)
    s0r = 0; s0 != 0  && (for i=n0:p.nword; head[i]==s0 && (s0r += 1); end)

    if SHIFTOK(p)                                               # SHIFT moves n0 to s
        cost[SHIFT(p)] = (n0l +                                 # n0 gets no more ldeps
                          (findprev(p.stack, n0h, p.sptr) > 0)) # or lhead
    end
    if REDUCEOK(p)                                              # REDUCE pops s0
        cost[REDUCE(p)] = s0r                                   # s0 gets no more rdeps
    end
    if RIGHTOK(p)                                               # RIGHT adds (s0,n0) and shifts n0 to s
        rcost = (n0l + (n0h > n0) + (n0h == 0) +                # n0 gets no more ldeps, rhead, 0head,
                 (findprev(p.stack, n0h, p.sptr-1) > 0))        # or lhead<s0
        if n0h == s0                                            # if we have the correct head
           cost[RMOVES(p)] = rcost + 1                          # +1 for the wrong labels
           cost[RMOVE(p,deprel[n0])] -= 1                       # except for the correct label
        else
           cost[RMOVES(p)] = rcost                              # otherwise we are done
        end
    end
    if LEFTOK(p)                                                # LEFT adds (n0,s0) and reduces s0
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

