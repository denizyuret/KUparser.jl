# DONE: build test parser
## no costs? find another way to handle valid moves!
## do we return loss vs LAS?
# DONE: freeze LM model
# DONE: precompute word vectors

using JLD, Knet, KUparser
using KUparser: movecosts
using AutoGrad: getval
MAJOR=1                         # 1:column-major 2:row-major
MAXWORD=32                      # truncate long words at this length. length("counterintelligence")=19
MAXSENT=64                      # skip longer sentences during training
MINSENT=2                       # skip shorter sentences during training
LOGGING=0
macro msg(_x) :(if LOGGING>0; join(STDERR,[Dates.format(now(),"HH:MM:SS"), $_x,'\n'],' '); end) end
macro log(_x) :(@msg($(string(_x))); $(esc(_x))) end

Knet.gcdebug(true)              #DBG

function oracletrain(;model=_model, optim=_optim, corpus=_corpus, vocab=_vocab, parsertype=ArcHybridR1, feats=Flist.hybrid25, batchsize=128)
    # global grads, optim, sentbatches, sentences
    srand(1)
    sentbatches = minibatch(corpus,batchsize; maxlen=MAXSENT, minlen=MINSENT, shuf=true)
    nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
    nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
    println("nsent=$nsent/$nsent0 nword=$nword/$nword0")
    niter = 0
    @time for sentences in sentbatches
        grads = oraclegrad(model, sentences, vocab, parsertype, feats)
        update!(model, grads, optim)
        print('.')
    end
    println()
end

# function oracleloss(pmodel, cmodel, sentences, vocab, parsertype, feats)
function oracleloss(pmodel, sentences, vocab, parsertype, feats)
    # global parsers,fmatrix,beamends,cscores,pscores,parsers0,beamends0,totalloss,loss,pcosts,pcosts0
    # fillvecs!(cmodel, sentences, vocab)
    parsers = map(parsertype, sentences)
    mcosts = Array(Cost, parsers[1].nmove)
    done = falses(length(parsers))
    totalloss = 0
    while !all(done)
        fmatrix = features(parsers, feats, pmodel)
        if true #GPU
            @assert isa(getval(fmatrix),KnetArray{Float32,2})
        else #CPU
            @assert isa(getval(fmatrix),Array{Float32,2})
            fmatrix = KnetArray(fmatrix)
        end
        scores = mlp(pmodel[:parser], fmatrix)
        logprob = logp(scores,MAJOR)
        for (i,p) in enumerate(parsers)
            if done[i]; continue; end
            movecosts(p, p.sentence.head, p.sentence.deprel, mcosts)
            goldmove = indmin(mcosts)
            if mcosts[goldmove] == typemax(Cost)
                done[i] = true
                p.sentence.parse = p
            else
                totalloss -= logprob[goldmove,i]
                move!(p, goldmove)
            end
        end
    end
    return totalloss / length(sentences)
end

oraclegrad = grad(oracleloss)

function beamtrain(;model=_model, optim=_optim, corpus=_corpus, vocab=_vocab, parsertype=ArcHybridR1, feats=Flist.hybrid25, beamsize=10, batchsize=128)
    # global grads, optim, sentbatches, sentences
    srand(1)
    sentbatches = minibatch(corpus,batchsize; maxlen=MAXSENT, minlen=MINSENT, shuf=true)
    nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
    nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
    println("nsent=$nsent/$nsent0 nword=$nword/$nword0")
    niter = 0
    for sentences in sentbatches
        #DBG gc(); Knet.knetgc(); gc()
        #DBG @show niter+=1
        #DBG @show length(sentences[1])
        #DBG ploss = beamloss(model, sentences, vocab, parsertype, feats, beamsize)
        grads = beamgrad(model, sentences, vocab, parsertype, feats, beamsize; earlystop=true)
        update!(model, grads, optim)
        print('.')
        #DBG gc()
        #DBG Knet.memdbg()
        #DBG Knet.gpuinfo(n=10)
        #DBG readline()
    end
    println()
end

function beamtest(;model=_model, corpus=_corpus, vocab=_vocab, parsertype=ArcHybridR1, feats=Flist.hybrid25, beamsize=10, batchsize=128)
    sentbatches = minibatch(corpus,batchsize)
    for sentences in sentbatches
        beamloss(model, sentences, vocab, parsertype, feats, beamsize; earlystop=false)
        print('.')
    end
    println()
    las(corpus)
end

function las(corpus)
    nword = ncorr = 0
    for s in corpus
        p = s.parse
        nword += length(s)
        ncorr += sum((s.head .== p.head) & (s.deprel .== p.deprel))
    end
    ncorr / nword
end

# parse a minibatch of sentences using beam search, global normalization, early updates and report loss.
# leave the results in sentences[i].parse, return per sentence loss
# beamloss(_model, _model, _corpus, _vocab, ArcHybridR1, Flist.hybrid25, 10)
# function beamloss(pmodel, cmodel, sentences, vocab, parsertype, feats, beamsize; earlystop=false)
function beamloss(pmodel, sentences, vocab, parsertype, feats, beamsize; earlystop=false)
    # global parsers,fmatrix,beamends,cscores,pscores,parsers0,beamends0,totalloss,loss,pcosts,pcosts0
    # fillvecs!(cmodel, sentences, vocab)
    @log parsers = parsers0 = map(parsertype, sentences)
    @log beamends = beamends0 = collect(1:length(parsers)) # marks end column of each beam, initially one parser per beam
    @log pcosts  = pcosts0  = zeros(Int, length(sentences))
    @log pscores = pscores0 = zeros(Float32, length(sentences))
    @log totalloss = stepcount = 0
    while length(beamends) > 0
        # features (vcat) are faster on cpu, mlp is faster on gpu
        @log fmatrix = features(parsers, feats, pmodel)
        if true #GPU
            @assert isa(getval(fmatrix),KnetArray{Float32,2})
        else #CPU
            @assert isa(getval(fmatrix),Array{Float32,2})
            fmatrix = KnetArray(fmatrix)
        end
        @log cscores = Array(mlp(pmodel[:parser], fmatrix)) .+ pscores' # candidate cumulative scores
        @log parsers,pscores,pcosts,beamends,loss = nextbeam(parsers, cscores, pcosts, beamends, beamsize; earlystop=earlystop)
        totalloss += loss
        stepcount += 1
    end
    # emptyvecs!(sentences)       # if we don't empty, gc cannot clear these vectors; empty if finetuning wvecs
    if earlystop; @show (maximum(map(length,sentences)),stepcount); end
    return totalloss / length(sentences)
end

beamgrad = grad(beamloss)

function nextbeam(parsers, mscores, pcosts, beamends, beamsize; earlystop=false)
    #global mcosts
    n = beamsize * length(beamends) + 1
    newparsers, newscores, newcosts, newbeamends, loss = Array(Any,n),Array(Any,n),Array(Int,n),Int[],0.0
    nmoves,nparsers = size(mscores)                     # mscores[m,p] is the score of move m for parser p
    #TEST: will not have mcosts
    mcosts = Array(Any, nparsers)                       # mcosts[p][m] will be the cost vector for parser[p],move[m] if needed
    n = p0 = 0
    for p1 in beamends                                  # parsers[p0+1:p1], mscores[:,p0+1:p1] is the current beam belonging to a common sentence
        s0,s1 = 1 + nmoves*p0, nmoves*p1                # mscores[s0:s1] are the linear indices for mscores[:,p0:p1]
        nsave = n                                       # newparsers,newscores,newcosts[nsave+1:n] will the new beam for this sentence
        #TEST: will not have ngold
        ngold = 0                                       # ngold, if nonzero, will be the index of the gold path in beam
        sorted = sortperm(mscores[s0:s1], rev=true)	
        for isorted in sorted
            (move,parent) = ind2sub(mscores, isorted + s0 - 1) # find cartesian index of the next best score
            parser = parsers[parent]
            if !moveok(parser,move); continue; end  # skip illegal move
            #TEST: will not have mcosts
            if earlystop && !isassigned(mcosts, parent) # not every parent may have children, avoid unnecessary movecosts
                mcosts[parent] = movecosts(parser, parser.sentence.head, parser.sentence.deprel)
            end
            n += 1
            newparsers[n] = copy(parser); move!(newparsers[n], move)
            newscores[n] = mscores[move,parent]
            #TEST: no newcosts during test
            if earlystop
                newcosts[n] = pcosts[parent] + mcosts[parent][move]
                if newcosts[n] == 0
                    if ngold==0
                        ngold=n
                    else
                        @msg("multiple gold moves for $(parser.sentence)")
                    end
                end
            end
            if n-nsave == beamsize; break; end
        end
        if n == nsave
            if parsers[p1].nword == 1                   # single word sentences give us no moves
                s = parsers[p1].sentence
                s.parse = parsers[p1]
            else
                error("No legal moves?")                # otherwise this is suspicious
            end
        #TEST: there will be no ngold during test
        elseif earlystop && ngold == 0                  # gold path fell out of beam, early stop
            gs = goldscore(parsers,mscores,pcosts,mcosts,(p0+1):p1)
            if !isnan(gs)                               # could be nan for non-projective sentences
                newscores[n+1] = gs
                loss = loss - newscores[n+1] + logsumexp2(newscores,nsave+1:n+1)
            end
            n = nsave
        elseif endofparse(newparsers[n])                # all parsers in beam have finished, gold among them if earlystop
            #TEST: there will be no ngold during test, cannot return beamloss, just return the highest scoring parse, no need for normalization
            if earlystop
                loss = loss - newscores[ngold] + logsumexp2(newscores,nsave+1:n)
            end
            s = newparsers[n].sentence
            @assert s == newparsers[nsave+1].sentence
            @assert newscores[nsave+1] >= newscores[n]
            s.parse = newparsers[nsave+1]
            n = nsave                                   # do not add finished parsers to new beam
        else                                            # all good keep going
            push!(newbeamends, n)
        end
        p0 = p1
    end
    return newparsers[1:n], newscores[1:n], newcosts[1:n], newbeamends, loss
end

function logsumexp2(a,r)
    z = 0; amax = a[r[1]]
    for i in r; z += exp(a[i]-amax); end
    return log(z) + amax
end

function goldscore(parsers,mscores,pcosts,mcosts,beamrange)
    parent = findfirst(view(pcosts,beamrange),0)
    if parent == 0; error("cannot find gold parent in $beamrange"); end
    parent += first(beamrange) - 1
    if !isassigned(mcosts, parent)
        parser = parsers[parent]
        mcosts[parent] = movecosts(parser, parser.sentence.head, parser.sentence.deprel)
    end
    move = findfirst(mcosts[parent],0)
    if move == 0
        @msg("cannot find gold move for $(parsers[parent].sentence)")
        return NaN
    end
    return mscores[move,parent]
end

endofparse(p)=(p.sptr == 1 && p.wptr > p.nword)

using AutoGrad
import Base: sortperm, ind2sub
@zerograd sortperm(a;o...)
@zerograd ind2sub(a,i...)

function mlp(w,x)
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

function fillvecs!(model, sentences, vocab)
    #global cids,wids,maxword,maxsent,sow,eow,cdata,cmask,wembed,sos,eos,wdata,wmask,forw,back
    @log cids,wids,maxword,maxsent = maptoint(sentences, vocab)
    # Get word embeddings: do this in minibatches otherwise may run out of memory
    @log sow,eow = vocab.cdict[vocab.sowchar],vocab.cdict[vocab.eowchar]

    # this blows up gpu memory:
    # @log cdata,cmask = tokenbatch(cids,maxword,sow,eow)
    # @log wembed = charlstm(model,cdata,cmask)

    # minibatch variant is easier on memory:
    wembed = Any[]; wbatch = 128 # TODO: configurable batchsize
    for i=1:wbatch:length(cids)    
        j=min(i+wbatch-1,length(cids))
        cij = view(cids,i:j)
        @log cdata,cmask = tokenbatch(cij,maxword,sow,eow) # cdata/cmask[T+2][V] where T: longest word (140), V: input word vocab (19674)
        @log push!(wembed, charlstm(model,cdata,cmask))    # wembed[C,V] where V: input word vocab, C: charlstm hidden (350)
    end
    wembed = (MAJOR==1 ? hcatn(wembed...) : vcatn(wembed...))
    #DBG @show (maxword,length(cids),size(wembed)); Knet.gpuinfo(n=10); readline()

    # Get context embeddings
    @log sos,eos = vocab.idict[vocab.sosword],vocab.idict[vocab.eosword]
    @log wdata,wmask = tokenbatch(wids,maxsent,sos,eos) # wdata/wmask[T+2][B] where T: longest sentence (159), B: batch size (12543)
    @log forw,back = wordlstm(model,wdata,wmask,wembed)  # forw/back[T][W,B] where T: longest sentence, B: batch size, W: wordlstm hidden (300)
    # Get embeddings into sentences -- must empty later for gc!
    @log fillwvecs!(sentences, wids, wembed)
    @log fillcvecs!(sentences, forw, back)
end

function emptyvecs!(sentences)
    for s in sentences   
        empty!(s.wvec); empty!(s.fvec); empty!(s.bvec)
    end
end    

function fillwvecs!(sentences, wids, wembed)
    if MAJOR != 1; error(); end # not impl yet
    @inbounds for (s,wids) in zip(sentences,wids)
        empty!(s.wvec)
        for w in wids
            if true #GPU
                push!(s.wvec, wembed[:,w])
            else #CPU
                push!(s.wvec, Array(wembed[:,w]))
            end
        end
    end
end

function fillcvecs!(sentences, forw, back)
    if MAJOR != 1; error(); end # not impl yet
    T = length(forw)
    @inbounds for i in 1:length(sentences)
        s = sentences[i]
        empty!(s.fvec)
        empty!(s.bvec)
        N = length(s)
        for n in 1:N
            t = T-N+n
            if true #GPU
                push!(s.fvec, forw[t][:,i])
                push!(s.bvec, back[t][:,i])
            else #CPU
                push!(s.fvec, Array(forw[t][:,i]))
                push!(s.bvec, Array(back[t][:,i]))
            end
        end
    end
end

# initoptim creates optimization parameters for each numeric weight
# array in the model.  This should work for a model consisting of any
# combination of tuple/array/dict.
initoptim{T<:Number}(::KnetArray{T},otype)=eval(parse(otype))
initoptim{T<:Number}(::Array{T},otype)=eval(parse(otype))
initoptim(a::Associative,otype)=Dict(k=>initoptim(v,otype) for (k,v) in a) 
initoptim(a,otype)=map(x->initoptim(x,otype), a)

# treemap for general functions acting on a model
treemap{T<:Number}(f,x::KnetArray{T})=f(x)
treemap{T<:Number}(f,x::Array{T})=f(x)
treemap(f,a::Associative)=Dict(k=>treemap(f,v) for (k,v) in a)
treemap(f,a)=map(x->treemap(f,x), a)


# function loss(model, sentences, parsertype, beamsize)
function lmloss(model, sentences, vocab::Vocab; result=nothing)
    # map words and chars to Ints
    # words and sents contain Int (char/word id) arrays without sos/eos tokens
    # TODO: optionally construct cdict here as well
    # words are the char ids for each unique word
    # sents are the word ids for each sentence
    # wordlen and sentlen are the maxlen word and sentence
    words,sents,wordlen,sentlen = maptoint(sentences, vocab)

    # Find word vectors for a batch of sentences using character lstm or cnn model
    # note that tokenbatch adds sos/eos tokens and padding to each sequence
    sow,eow = vocab.cdict[vocab.sowchar],vocab.cdict[vocab.eowchar]
    cdata,cmask = tokenbatch(words,wordlen,sow,eow)
    wembed = charlstm(model,cdata,cmask)

    # Find context vectors using word blstm
    sos,eos = vocab.idict[vocab.sosword],vocab.idict[vocab.eosword]
    wdata,wmask = tokenbatch(sents,sentlen,sos,eos)
    forw,back = wordlstm(model,wdata,wmask,wembed)

    # Test predictions
    unk = vocab.odict[vocab.unkword]
    odata,omask = goldbatch(sentences,sentlen,vocab.odict,unk)
    total = lmloss1(model,odata,omask,forw,back; result=result)
    return -total/length(sentences)
end

function lmloss1(model,data,mask,forw,back; result=nothing)
    # data[t][b]::Int gives the correct word at sentence b, position t
    T = length(data); if !(T == length(forw) == length(back) == length(mask)); error(); end
    B = length(data[1])
    weight,bias = wsoft(model),bsoft(model)
    prd(t) = (if MAJOR==1; weight * vcat(forw[t],back[t]) .+ bias; else; hcat(forw[t],back[t]) * weight .+ bias; end)
    idx(t,b,n) = (if MAJOR==1; data[t][b] + (b-1)*n; else; b + (data[t][b]-1)*n; end)
    total = count = 0
    @inbounds for t=1:T
        ypred = prd(t)
        nrows,ncols = size(ypred)
        index = Int[]
        for b=1:B
            if mask[t][b]==1
                push!(index, idx(t,b,nrows))
            end
        end
        o1 = logp(ypred,MAJOR)
        o2 = o1[index]
        total += sum(o2)
        count += length(o2)
    end
    if result != nothing; result[1]+=AutoGrad.getval(total); result[2]+=count; end
    return total
end

# Find all unique words, assign an id to each, and lookup their characters in cdict
# TODO: construct/add-to cdict here as well?
# in vocab: reads sosword, eosword, unkchar, cdict; writes idict
function maptoint(sentences, v::Vocab)
    wdict = empty!(v.idict)
    cdict = v.cdict
    unkcid = cdict[v.unkchar]
    words = Vector{Int}[]
    sents = Vector{Int}[]
    wordlen = 0
    sentlen = 0
    @inbounds for w in (v.sosword,v.eosword)
        wid = get!(wdict, w, 1+length(wdict))
        word = Array(Int, length(w))
        wordi = 0
        for c in w
            word[wordi+=1] = get(cdict, c, unkcid)
        end
        if wordi != length(w); error(); end
        if wordi > wordlen; wordlen = wordi; end
        push!(words, word)
    end
    @inbounds for s in sentences
        sent = Array(Int, length(s.word))
        senti = 0
        for w in s.word
            ndict = length(wdict)
            wid = get!(wdict, w, 1+ndict)
            sent[senti+=1] = wid
            if wid == 1+ndict
                word = Array(Int, length(w))
                wordi = 0
                for c in w
                    word[wordi+=1] = get(cdict, c, unkcid)
                end
                if wordi != length(w); error(); end
                # Long words kill gpu memory
                if wordi > MAXWORD; wordi=MAXWORD; word = word[1:wordi]; end
                if wordi > wordlen; wordlen = wordi; end
                push!(words, word)
            end
        end
        if senti != length(s.word); error(); end
        if senti > sentlen; sentlen = senti; end
        push!(sents, sent)
    end
    if length(wdict) != length(words); error("wdict=$(length(wdict)) words=$(length(words))"); end
    return words,sents,wordlen,sentlen
end
    
# Create token batches, adding start/end tokens and masks, pad at the beginning
function tokenbatch(sequences,maxlen,sos,eos,pad=eos)
    B = length(sequences)
    T = maxlen + 2
    data = [ Array(Int,B) for t in 1:T ]
    mask = [ Array(Float32,B) for t in 1:T ]
    @inbounds for t in 1:T
        for b in 1:B
            N = length(sequences[b])
            n = t - T + N + 1
            if n < 0
                mask[t][b] = 0
                data[t][b] = pad
            else
                mask[t][b] = 1
                if n == 0
                    data[t][b] = sos
                elseif n <= N
                    data[t][b] = sequences[b][n]
                elseif n == N+1
                    data[t][b] = eos
                else
                    error()
                end
            end
        end
    end
    return data,mask
end

function goldbatch(sentences, maxlen, wdict, unkwid, pad=unkwid)
    B = length(sentences)
    T = maxlen # no need for sos/eos for gold
    data = [ Array(Int,B) for t in 1:T ]
    mask = [ Array(Float32,B) for t in 1:T ]
    @inbounds for t in 1:T
        for b in 1:B
            N = length(sentences[b])
            n = t - T + N
            if n <= 0
                mask[t][b] = 0
                data[t][b] = pad
            else
                mask[t][b] = 1
                data[t][b] = get(wdict, sentences[b].word[n], unkwid)
            end
        end
    end
    return data,mask
end

# Run charid arrays through the LSTM, collect last hidden state as word embedding
function charlstm(model,data,mask)
    weight,bias,embeddings = wchar(model),bchar(model),echar(model)
    T = length(data)
    B = length(data[1])
    H = div(length(bias),4)
    cembed(t)=(if MAJOR==1; embeddings[:,data[t]]; else; embeddings[data[t],:]; end)
    czero=(if MAJOR==1; fill!(similar(bias,H,B), 0); else; fill!(similar(bias,B,H), 0); end)
    hidden = cell = czero       # TODO: cache this
    mask = map(KnetArray, mask) # TODO: dont hardcode atype
    @inbounds for t in 1:T
        (hidden,cell) = lstm(weight,bias,hidden,cell,cembed(t);mask=mask[t])
    end
    return hidden
end

function wordlstm(model,data,mask,embeddings) # col-major
    weight,bias = wforw(model),bforw(model)
    T = length(data)
    B = length(data[1])
    H = div(length(bias),4)
    wembed(t)=(if MAJOR==1; embeddings[:,data[t]]; else; embeddings[data[t],:]; end)
    wzero=(if MAJOR==1; fill!(similar(bias,H,B), 0); else; fill!(similar(bias,B,H), 0); end)
    hidden = cell = wzero
    mask = map(KnetArray, mask)           # TODO: dont hardcode atype
    forw = Array(Any,T-2)       # exclude sos/eos
    @inbounds for t in 1:T-2
        (hidden,cell) = lstm(weight,bias,hidden,cell,wembed(t); mask=mask[t])
        forw[t] = hidden
    end
    weight,bias = wback(model),bback(model)
    if H != div(length(bias),4); error(); end
    hidden = cell = wzero
    back = Array(Any,T-2)
    @inbounds for t in T:-1:3
        (hidden,cell) = lstm(weight,bias,hidden,cell,wembed(t); mask=mask[t])
        back[t-2] = hidden
    end
    return forw,back
end

function lstm(weight,bias,hidden,cell,input; mask=nothing)
    if MAJOR==1
        gates   = weight * vcat(input,hidden) .+ bias
        H       = size(hidden,1)
        forget  = sigm(gates[1:H,:])
        ingate  = sigm(gates[1+H:2H,:])
        outgate = sigm(gates[1+2H:3H,:])
        change  = tanh(gates[1+3H:4H,:])
        if mask!=nothing; mask=reshape(mask,1,length(mask)); end
    else
        gates   = hcat(input,hidden) * weight .+ bias
        H       = size(hidden,2)
        forget  = sigm(gates[:,1:H])
        ingate  = sigm(gates[:,1+H:2H])
        outgate = sigm(gates[:,1+2H:3H])
        change  = tanh(gates[:,1+3H:end])
    end
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    if mask != nothing
        hidden = hidden .* mask
        cell = cell .* mask
    end
    return (hidden,cell)
end

function loadvocab()
    a = load("english_vocabs.jld")
    Vocab(a["char_vocab"],Dict{String,Int}(),a["word_vocab"],
          "<s>","</s>","<unk>",'↥','Ϟ','⋮', UPOSTAG, UDEPREL)
end

function loadcorpus(v::Vocab)
    corpus = Any[]
    s = Sentence(v)
    for line in eachline("foo-en-ud-train.conllu")
        if line == "\n"
            push!(corpus, s)
            s = Sentence(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing
            #                id   word   lem  upos   xpos feat head   deprel
            push!(s.word, m.captures[1])
            push!(s.postag, v.postags[m.captures[2]])
            push!(s.head, parse(Position,m.captures[3]))
            push!(s.deprel, v.deprels[m.captures[4]])
        end
    end
    return corpus
end

function loadmodel(v::Vocab; hidden=1024)
    model = Dict()
    m = load("english_ch12kmodel_2.jld")["model"]
    if MAJOR==1
        for k in (:cembed,); model[k] = KnetArray(m[k]'); end
        for k in (:forw,:soft,:back,:char); model[k] = map(KnetArray, m[k]'); end
    else
        for k in (:cembed,); model[k] = KnetArray(m[k]); end
        for k in (:forw,:soft,:back,:char); model[k] = map(KnetArray, m[k]); end
    end
    initk(d...) = KnetArray{Float32}(xavier(d...))
    inita(d...) = Array{Float32}(xavier(d...))
    for (k,n,d) in ((:postag,17,17),(:deprel,37,37),(:lcount,10,10),(:rcount,10,10),(:distance,10,10))
        if true #GPU
            model[k] = [ initk(d) for i=1:n ]
        else #CPU
            model[k] = [ inita(d) for i=1:n ]
        end
    end
    parsedims = (5796,hidden,73)
    model[:parser] = Any[]
    for i=2:length(parsedims)
        push!(model[:parser], initk(parsedims[i],parsedims[i-1]))
        push!(model[:parser], initk(parsedims[i],1))
    end
    return model
end

wsoft(model)=model[:soft][1]
bsoft(model)=model[:soft][2]
wchar(model)=model[:char][1]
bchar(model)=model[:char][2]
echar(model)=model[:cembed]
wforw(model)=model[:forw][1]
bforw(model)=model[:forw][2]
wback(model)=model[:back][1]
bback(model)=model[:back][2]
postagv(model)=model[:postag]   # 17 postags
deprelv(model)=model[:deprel]   # 37 deprels
lcountv(model)=model[:lcount]   # 10 lcount
rcountv(model)=model[:rcount]   # 10 rcount
distancev(model)=model[:distance] # 10 distance

function minibatch(corpus, batchsize; maxlen=typemax(Int), minlen=1, shuf=false)
    data = Any[]
    sorted = sort(corpus, by=length)
    i1 = findfirst(x->(length(x) >= minlen), sorted)
    if i1==0; error("No sentences >= $minlen"); end
    i2 = findfirst(x->(length(x) > maxlen), sorted)
    if i2==0; i2=length(corpus); else; i2=i2-1; end
    for i in i1:batchsize:i2
        j = min(i2, i+batchsize-1)
        push!(data, sorted[i:j])
    end
    if shuf; data=shuffle(data); end
    return data
end

function main(;batch=3000,nsent=0,result=zeros(2))
    global _vocab = loadvocab()
    global _model = loadmodel(_vocab)
    global _optim = initoptim(_model,"Adam()") # "Sgd(lr=.1)")
    global _corpus = loadcorpus(_vocab)
    @time fillvecs!(_model, _corpus, _vocab)
    c = (nsent == 0 ? _corpus : _corpus[1:nsent])
    _data = minibatch(c, batch)
    # global _data = [ _corpus[i:min(i+999,length(_corpus))] for i=1:1000:length(_corpus) ]
    # global _data = [ _corpus[1:1] ]
    for d in _data
        lmloss(_model, d, _vocab; result=result)
    end
    println(exp(-result[1]/result[2]))
    return result
end

# TODO: out-of-memory with hidden=4096?

function hiddenoptim(hidden,optim;epochs=1)
    srand(1)
    global _model, _optim
    _model = loadmodel(_vocab, hidden=hidden)
    _optim = initoptim(_model, optim)
    c = _corpus[1:128]
    for i=1:epochs; oracletrain(corpus=c); end
    beamtest(corpus=c,beamsize=1)
end

function getloss(d; result=zeros(2))
    lmloss(_model, d, _vocab; result=result)
    println(exp(-result[1]/result[2]))
    return result
end

# when fmatrix has mixed Rec and KnetArray, vcat does not do the right thing!  AutoGrad only looks at the first 2-3 elements!
#DBG: temp solution to AutoGrad vcat issue:
using AutoGrad
let cat_r = recorder(cat); global vcatn, hcatn
    function vcatn(a...)
        if any(x->isa(x,Rec), a)
            cat_r(1,a...)
        else
            vcat(a...)
        end
    end
    function hcatn(a...)
        if any(x->isa(x,Rec), a)
            cat_r(2,a...)
        else
            hcat(a...)
        end
    end
end

# Universal POS tags (17)
const UPOSTAG = Dict{String,PosTag}(
"ADJ"   => 1, # adjective
"ADP"   => 2, # adposition
"ADV"   => 3, # adverb
"AUX"   => 4, # auxiliary
"CCONJ" => 5, # coordinating conjunction
"DET"   => 6, # determiner
"INTJ"  => 7, # interjection
"NOUN"  => 8, # noun
"NUM"   => 9, # numeral
"PART"  => 10, # particle
"PRON"  => 11, # pronoun
"PROPN" => 12, # proper noun
"PUNCT" => 13, # punctuation
"SCONJ" => 14, # subordinating conjunction
"SYM"   => 15, # symbol
"VERB"  => 16, # verb
"X"     => 17, # other
)

# Universal Dependency Relations (37)
const UDEPREL = Dict{String,DepRel}(
"root"       => 1,  # root
"acl"        => 2,  # clausal modifier of noun (adjectival clause)
"advcl"      => 3,  # adverbial clause modifier
"advmod"     => 4,  # adverbial modifier
"amod"       => 5,  # adjectival modifier
"appos"      => 6,  # appositional modifier
"aux"        => 7,  # auxiliary
"case"       => 8,  # case marking
"cc"         => 9,  # coordinating conjunction
"ccomp"      => 10, # clausal complement
"clf"        => 11, # classifier
"compound"   => 12, # compound
"conj"       => 13, # conjunct
"cop"        => 14, # copula
"csubj"      => 15, # clausal subject
"dep"        => 16, # unspecified dependency
"det"        => 17, # determiner
"discourse"  => 18, # discourse element
"dislocated" => 19, # dislocated elements
"expl"       => 20, # expletive
"fixed"      => 21, # fixed multiword expression
"flat"       => 22, # flat multiword expression
"goeswith"   => 23, # goes with
"iobj"       => 24, # indirect object
"list"       => 25, # list
"mark"       => 26, # marker
"nmod"       => 27, # nominal modifier
"nsubj"      => 28, # nominal subject
"nummod"     => 29, # numeric modifier
"obj"        => 30, # object
"obl"        => 31, # oblique nominal
"orphan"     => 32, # orphan
"parataxis"  => 33, # parataxis
"punct"      => 34, # punctuation
"reparandum" => 35, # overridden disfluency
"vocative"   => 36, # vocative
"xcomp"      => 37, # open clausal complement
)

# """
# The input is minibatched and processed character based.
# The blstm language model used for preinit inputs characters outputs words.
# The characters of a word is used to obtain its embedding using an RNN or CNN.
# The input to the parser is minibatched sentences.
# The sentence lengths in a minibatch do not have to exactly match, we will support padding.
# """
# foo1

# """
# Let B be the minibatch size, and T the sentence length in the minibatch.
# input[t][b] is a string specifying the t'th word of the b'th sentence.    
# Do we include SOS/EOS tokens in input?  No, the parser does not need them.
# How about the ROOT token?  No, our parsers are written s.t. the root token is implicit.
# """
# foo2

# """
# We need head and deprel for loss calculation. It makes more sense to
# process an array of sentences, doing the grouping inside.
# """
# foo3

        # Need to check costs
        # Need to detect when beam is done and add its loss
        # Need to figure out what happens with nonprojective sentences, detect and skip? start parse?
        # Need to track the zero-cost parse(s)
        # Always have left=odd, right=even or vice-versa so it is easy to check
        # Exact cost not important, only whether nonzero
            # TODO: update cost, check early stop, regular stop
            # when do we check for anyvalidmoves?
            # we don't need to do costs if parent cost already > 0; only to check for illegal moves!
            # how do we check if we have gold path in beam (set flag in this for loop)
            # where do we find the gold path if not in beam (have to search outside)

# Loss function for beam search:

# Globally normalized probability for a sequence ending in BeamState s:
# Let score(s) be the cumulative score of the sequence up to s.
# prob(s) = exp(score(s)) - Z
# J(s) = -score(s) + log(Z)
# Z = exp(score(s)) summed over all paths

# Beam loss with early updates:
# wait until gold path falls out of the beam at step j, say s=gold[j]
# B is the beam at step j, together with gold state s
# J(s) = -score(s) + log(Z)
# Z = exp(score(b)) summed over all b in B

        # if endofparse(parsers[p1])
        #     igold = findfirst(pcosts,0)
        #     if igold == 0; error("cannot find gold path at end of parse"); end
        #     loss += -pscores(igold)+logsumexp(pscores)
        #     # This doesn't work because we dont have pscores, we have unnecessary mscores
        #     # This should be detected before we add these to the beam
        # end

        # There are two ways this should end
        # 1. End of parse: we will get no valid moves
        # should check for that, all parsers in beam have the same sentence, should end at the same time
        # in that case we need to calculate a loss based on the beam
        # needs tracking where the gold path is
        # if #2 works, we should have the gold path in the beam!, we still need to find it though.
        # 2. Early stop
        # needs tracking where the gold path is
        # if it doesn't make it to the beam, we should just return loss and not add anything to newparsers
        # we still need the scores on the beam to calculate Z

# function minibatch1(corpus, batchsize)
#     data = Any[]
#     dict = Dict{Int,Vector{Int}}()
#     for i in 1:length(corpus)
#         s = corpus[i]
#         l = length(s)
#         a = get!(dict, l, Int[])
#         push!(a, i)
#         if length(a) == batchsize
#             push!(data, corpus[a])
#             empty!(a)
#         end
#     end
#     for (k,a) in dict
#         if !isempty(a)
#             push!(data, corpus[a])
#         end
#     end
#     return data
# end

main(nsent=1)
nothing
