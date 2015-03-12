# parser[1..np] candidate parser states on the beam
# pscore[1..np] their current cumulative scores
# x is the pre-allocated feature matrix and idx is its last used column
# z is the zero-allocated one-of-k correct move matrix corresponding to x
# c[j,i] is the cost of j'th move for i'th parser
# z[:,idx+i] is the one-of-k encoded mincost move for the i'th parser
# x[:,idx+i] is the feature vector for the i'th parser
# y[j,i] is the score for the j'th move of i'th parser
# TODO: make KUnet.predict work with subarrays so this mess is not necessary:
# The y that comes back has unnormalized log probabilities
#    We need normalized log probabilities to use as scores
# TODO: implement normalize on gpu
# TODO: We are parsing the sentence, so we should not do early stop
#    if correct parse falls out of beam?
# Idea: use x of size np and copy it to the big array
#    simplifies the code and can then handle earlystop by not copying once correct parse drops out of beam
# In KUnet, can we avoid reallocating everything unless we need more space?
#    If batch gets smaller, just run the big batch through and copy part of the result?
# Now we sort the nc candidates and copy top np back to the beam
# softperm! may not be necessary when nc <= beam (unless we care about the first candidate being the best)
# do we need features when we have 0 or 1 valid move?
# handle the idx+i business with subarrays: x-> sub(x, :, idx+1:idx+np)

function bparse(sentence::Sentence, net::Net, feats::Features, beam::Integer)
    while true
        for i=1:np
            cost(parser[i], sentence.head, (ci=sub(c,:,i)))
            z[indmin(ci), i] = one(eltype(z))
            features(parser[i], sentence, feats, sub(x, :, i))
        end
        any(isfinite(c)) || break
        (xxcols != np) && (xxcols = np; KUnet.free(xx); xx = similar(net[1].w, (xrows, xxcols)))
        copy!(xx, (1:xrows, 1:np), x, (1:xrows, idx+1:idx+np))
        yy = KUnet.forw(net, xx, false)
        copy!(y, (1:yrows, 1:np), yy, (1:yrows, 1:np))
        normalize!(y)
        nc = 0
        for i=1:np
            for j=1:yrows
                isfinite(c[j,i]) || continue
                nc += 1
                cparser[nc] = i
                cmove[nc] = j
                cscore[nc] = pscore[i] + y[j,i]
            end
        end
        sortperm!(sub(csorted, 1:nc), sub(cscore, 1:nc); rev=true)
        for i=1:min(nc,beam)
            c=csorted[i]
            p=cparser[c]
            copy!(parser2[i], parser[p])
            move!(parser2[i], cmove[c])
            pscore2[i] = pscore[p] + cscore[c]
        end
        parser,pscore,parser2,pscore2 = parser2,pscore2,parser,pscore
    end
    return parser[1].head
end

function normalize!(y::Matrix)
    yrows,ycols = size(y)
    for j=1:ycols
        ymax = typemin(eltype(y))
        for i=1:yrows; y[i,j] > ymax && (ymax = y[i,j]); end
        z = zero(eltype(y))
        for i=1:yrows; z += exp((y[i,j] -= ymax)); end
        logz = log(z)
        for i=1:yrows; y[i,j] -= logz; end
    end
end



#     # svalid = Array(Int, batch)          # indices of valid sentences in current batch
#     # idx = 0                             # index of last used column in x, y, z
#     # for s=1:nsent; p[s] = ArcHybrid(wcnt(corpus[s])); end
#     # xxcols = batch
#     # xx = similar(net[1].w, (xrows, xxcols)) # device array

#     # parse corpus[b:e] in parallel
#     for b = 1:batch:nsent
#         e = b + batch - 1
#         (e > nsent) && (e = nsent; batch = e - b + 1)
#         for s=b:e
#             p[s] = ArcHybrid(wcnt(corpus[s]))
#             svalid[s-b+1] = s
#         end
#         nvalid = batch
#         while true
#             # Update svalid and nvalid
#             nv = 0
#             for i=1:nvalid
#                 s = svalid[i]
#                 vs = sub(v, :, s)
#                 valid(p[s], vs)
#                 any(vs) && (nv += 1; svalid[nv] = s)
#             end
#             (nv == 0) && break
#             nvalid = nv

#             # svalid[1:nvalid] are the indices of still valid sentences in current batch
#             # Take the next move with them
#             # First calculate features x[:,idx+1:idx+nvalid]
#             for i=1:nvalid
#                 s = svalid[i]
#                 feat ? features(p[s], corpus[s], feats, sub(x, :, idx + i)) : rand!(sub(x,:,idx+i))
#             end

#             # Next predict y in bulk
#             (xxcols != nvalid) && (xxcols = nvalid; KUnet.free(xx); xx = similar(net[1].w, (xrows, xxcols)))
#             copy!(xx, (1:xrows, 1:nvalid), x, (1:xrows, idx+1:idx+nvalid))
#             yy = KUnet.forw(net, xx, false)
#             copy!(y, (1:yrows, idx+1:idx+nvalid), yy, (1:yrows, 1:nvalid))

#             # Finally find best moves and execute max score valid moves
#             for i=1:nvalid
#                 s = svalid[i]
#                 bestmove = indmin(cost(p[s], corpus[s].head))
#                 z[bestmove, idx+i] = one(xtype)
#                 maxmove, maxscore = 0, -Inf
#                 for j=1:yrows
#                     yj = y[j,idx+i]
#                     v[j,s] && (yj > maxscore) && ((maxmove, maxscore) = (j, yj))
#                 end
#                 move!(p[s], maxmove)
#             end # for i=1:nvalid
#             idx = idx + nvalid
#         end # while true
#     end # for b = 1:batch:nsent
#     KUnet.free(xx)
#     h = Array(Pvec, nsent) 	# predicted heads
#     for s=1:nsent; h[s] = p[s].head; end
#     return (h, x, y, z)         # TODO: do not need to alloc or return y
# end





# # # We need to return (h,x,y,z)
# # # h is the heads from the final parser state.  no need for history.  definite need for multiple parsers.
# # # x is all features we encountered during search, 
# # # we dont really need y!  we can always recompute from model and x.
# # # we do need z, the mincostmove from that state for later training.
# # # so get a new parser, compute feats, move costs, move scores, save feats and mincostmove
# # # sort based on scores and keep going.
# # # agenda does not have to consist of parsers, maybe must parent/move/score tuples.
# # # parsers are huge with their matrices etc.  reuse?  just keep beam of them?
# # # may need copy parser?  gparser just used one.  even that can be improved with tuples for cost and valid.

# # # Attention: mincost may not be 0 for all sentences!  
# # # Better look at cost at the beginning to decide when we fall out of beam.

# # # Transition based beam parser.

# # # Pstate represents a snapshot of the parser during search.

# # type Pstate

# # # Do we need the previous states?  Isn't the last parser state enough?

# #     lastmove::Move             # move that led to this state from prev
# #     sumscore::Fval             # cumulative score including lastmove
# #     prev::Pstate               # previous state

# # # do we have to have a parser for each state?  can't we use a single parser?  we may need unmove.
# # # we can always reconstruct parser state from the moves.
# # # besides, the last state will always have all the heads.
# # # we need to return (h,x,y,z).

# #     parser::Parser             # the parser state

# # # do we need to keep around costs?  if so can we use a tuple? tuple([1,2]...) works.  avoid arrays in ArcHybrid!
# #     cost::Pvec                 # costs of moves from this state

# # # ditto for fvec, do we need it?  we need to save x,y pairs for training.
# #     fvec::Fvec                 # feature vector for parser state

# # # again, does this need to be stored?
# #     score::Fvec                # scores of moves from this state
# #     Pstate()=new()
# # end


# # function bparse(s::Sentence, n::Net, f::Features, beamwidth::Integer)
    
# #     beamlength = 2 * wcnt(s)                    # maximum number of parse moves

# # # this does not initialize the objects in the array, just undef pointers
# # # any reason to pre-allocate?  why can't we just have two arrays, one for parents, one for children.
# # # need to reconstruct history with prev pointers?
# # # agenda needs to be sorted anyway, just let the top beamwidth procreate.
# # # is it because we don't want to keep realloc when doing a corpus?

# # # beam -> batch -> darray
# # # no need for batch due to beam?

# # # just write the shortest program first!

# #     beam = Array(Pstate, beamwidth, beamlength) # beam[i,j] is the i'th candidate Pstate before the j'th move.
# #     nbeam = Array(Int, beamlength)              # nbeam[j] is the number of candidates on the j'th column of beam.
# #     nmove = ArcHybrid(1).nmove                  # number of legal moves from each Pstate TODO: do not hardcode ArcHybrid
# #     agenda = Array(Pstate, beamwidth * nmove)   # agenda contains children of candidates to be sorted next.
# #     fdims = flen(wdim(s), f)                    # dimensionality of feature vectors
# #     fmatrix = Array(Xval, fdims, beamwidth)     # fmatrix[i,j] is the i'th feature of the j'th candidate

# #     t = 1
# #     nbeam[1] = 1
# #     beam[1,1].sumscore = 0
# #     beam[1,1].parser = ArcHybrid(wcnt(s))

    


# #     nagenda = 0
    
# #     nbeam[1] = 1
# #     beam[1,1].sumscore = zero(Fval)
# #     beam[1,1].ismincost = true
# #     ibeam = 1

# #     # At
# #     # the end of the search, beam[1,n] will be the maxscorestate and
# #     # by following its prev links we can construct the maxscorepath.

# #     # We do not need to track the mincostpath!!!
# #     # However mincoststate has to be tracked
# #     # separately.  If the candidates list does not include a
# #     # mincoststate we still should have the mincoststate that just
# #     # fell out of the beam.
# # end

# # # As we are beam-searching for the best parse, we need to keep track
# # # of two paths: the mincostpath and the maxscorepath.  The cost is the
# # # number of gold arcs that are no longer possible from a Pstate.  The
# # # score is the cumulative score assigned to the moves along a path by
# # # the model.
# # #
# # # The initialstate is common to both paths.  It has no prev or
# # # lastmove fields.
# # #
# # # The final states come about in two ways:
# # #
# # # 1. If the sentence is finished.  In this case it will not have
# # # any valid moves.  So it does not need the fields feats, valid,
# # # cost, score.  It will have sumscore, prev, and lastmove.
# # #
# # # 2. If we hit early stop, i.e. the mincostpath dropped out of beam.
# # #
# # # In both cases the final states need to be included in the path to
# # # retrieve the last move and the last sumscore.
# # # 

