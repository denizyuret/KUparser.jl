# archybrid.jl, Deniz Yuret, July 7, 2014: Transition based greedy arc-hybrid parser based on:
# http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
# Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# Modified valid moves to output a single root-child.

typealias ArcHybrid Parser{:ArcHybrid}

# Moves are represented by integers 1..nmoves(p), 0 is not valid.
# They correspond to SHIFT,L1,R1,L2,R2,...,L[ndeps],R[ndeps]

@compat typealias Move Integer
nmoves(p::ArcHybrid)=(1+p.ndeps<<1)
isshift(p::ArcHybrid, m::Move)=(m==1)
isleft(p::ArcHybrid, m::Move)=(m&1==0)
isright(p::ArcHybrid, m::Move)=(m&1==1)
movelabel(p::ArcHybrid, m::Move)=convert(DepRel,m>>1)
leftmove(p::ArcHybrid, lab::DepRel)=(lab<<1)
rightmove(p::ArcHybrid, lab::DepRel)=(1+lab<<1)


# The mandatory first shift move is performed during initialization.  So
# our initial state is [w1][w2,w3,...,wn].
init!(p::ArcHybrid)=move!(p,1)


# move!(p,m) executes the move m on parser p.
# In the archybrid system we have three type of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# RIGHT_lb[(σ|s1|s0, β, A)] = (σ|s1, β, A ∪ {(s1, lb, s0)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A ∪ {(b, lb, s)})

function move!(p::ArcHybrid, m::Move)
    @assert (1 <= m <= nmoves(p)) "Move $m is not supported"
    if isshift(p,m)
        @assert (p.wptr <= p.nword)
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    elseif isright(p,m)
        @assert (p.sptr >= 2)
        arc!(p, p.stack[p.sptr-1], p.stack[p.sptr], movelabel(p,m))
        p.sptr -= 1
    else # isleft(p,m)
        @assert ((p.sptr >= 1) && (p.wptr <= p.nword))
        arc!(p, p.wptr, p.stack[p.sptr], movelabel(p,m))
        p.sptr -= 1
    end
end # move!

# anyvalidmoves(p) quickly tells us if we have any moves left

anyvalidmoves(p::ArcHybrid)=((p.wptr <= p.nword) || (p.sptr >= 2))

# movecosts() counts gold arcs that become impossible after possible
# moves.  Tokens start their lifecycle in the buffer without links.
# They move to the top of the buffer (n0) with SHIFT moves.  There they
# acquire left dependents using LEFT moves.  After that a single SHIFT
# moves them to the top of the stack (s0).  There they acquire right
# dependents with SHIFT-RIGHT pairs.  Finally from s0 they acquire a
# head with a LEFT or RIGHT move.  Any token from the buffer may become
# the right head but only s1 from the stack may become a left head for
# s0.  The parser terminates with a single word at s0 whose head is ROOT
# (represented as head=deprel=0).
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

function movecosts(p::ArcHybrid, head::AbstractArray, deprel::AbstractArray, 
                   cost::Pvec=Array(Position,nmoves(p)))
    nmove = nmoves(p)
    @assert (length(head) == p.nword)
    @assert (length(deprel) == p.nword)
    @assert (length(cost) == nmove)
    fill!(cost, Pinf)
    n0 = p.wptr
    SHIFT = 1
    LMOVES = 2:2:nmove
    RMOVES = 3:2:nmove

    if (n0 <= p.nword)                                          # SHIFT valid if n0, moving n0 to s0
        n0h = head[n0]                                          # n0h is the actual head of n0
        cost[SHIFT] = (sum(head[p.stack[1:p.sptr]] .== n0) +	# no more left dependents for n0
                       sum(p.stack[1:p.sptr-1] .== n0h) +       # no heads to the left of s0 for n0
                       ((n0h == 0) && (p.sptr >= 1)))           # no root head for n0 if there is s0
    end
    if (p.sptr >= 1)                                            # LEFT/RIGHT only valid if stack nonempty
        s0 = p.stack[p.sptr]                                    # s0 is top of stack
        s0h = head[s0]                                          # s0h is the actual head of s0
        s0r = sum(head[n0:end] .== s0)                          # s0r is the number of right children for s0

        if (n0 <= p.nword)                                      # LEFT valid if n0, making n0 head of s0
            lcost = (s0r +                                      # no more right children for s0
                     ((s0h > n0) ||                             # no heads to the right of n0 for s0
                      ((p.sptr == 1) && (s0h == 0)) ||          # no root head for s0 if alone
                      ((p.sptr > 1) &&                          # no more s1 for head of s0
                       (s0h == p.stack[p.sptr-1]))))
            if (s0h == n0)                                      # if we have the correct head
                cost[LMOVES] = lcost + 1                        # +1 for all the wrong labels
                cost[leftmove(p,deprel[s0])] -= 1               # except for the correct label
            else
                cost[LMOVES] = lcost                            # if n0 is not the actual head we are done
            end
        end
        if (p.sptr >= 2)                                        # RIGHT valid if s1 making s1 head of s0
            s1 = p.stack[p.sptr-1]                              # s1 is the stack element before s0
            rcost = s0r + (s0h >= n0)                           # no more right head or dependent for s0
            if (s0h == s1)                                      # if we have the correct head
                cost[RMOVES] = rcost + 1                        # +1 for all the wrong labels
                cost[rightmove(p,deprel[s0])] -= 1              # except for the correct label
            else
                cost[RMOVES] = rcost                            # if s1 is not the actual head we are done
            end
        end
    end
    return cost
end # movecosts

