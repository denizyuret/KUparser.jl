# TODO: make it work for cpu
# TODO: add dropout

using JLD, Knet, KUparser, BenchmarkTools, ArgParse
using KUparser: movecosts
using AutoGrad: getval

Knet.gcdebug(true)              #DBG
LOGGING=1                       # lots of debug output if nonzero
MAJOR=1                         # 1:column-major 2:row-major
MAXWORD=32                      # truncate long words at this length. length("counterintelligence")=19
MAXSENT=64                      # skip longer sentences during training
MINSENT=2                       # skip shorter sentences during training
FTYPE=Float32                   # floating point type
GPUFEATURES=false               # whether to compute features on gpu (vcat on gpu is too slow)
macro msg(_x) :(if LOGGING>0; join(STDOUT,[Dates.format(now(),"HH:MM:SS"), $_x,'\n'],' '); end) end
macro log(_x) :(@msg($(string(_x))); $(esc(_x))) end
macro sho(_x) :(if LOGGING>0; @show $(esc(_x)); else; $(esc(_x)); end) end
date(x)=join(STDOUT,[Dates.format(now(),"HH:MM:SS"), x,'\n'],' ')
type StopWatch; tstart; nstart; ncurr; nnext; StopWatch()=new(time(),0,0,1000); end

FEATS=["s1c","s1v","s1p","s1A","s1a","s1B","s1b",
       "s1rL", # "s1rc","s1rv","s1rp",
       "s0c","s0v","s0p","s0A","s0B","s0a","s0b","s0d",
       "s0rL", # "s0rc","s0rv","s0rp",
       "n0lL", # "n0lc","n0lv","n0lp",
       "n0c","n0v","n0p","n0A","n0a",
       "n1c","n1v","n1p",
       ]

function main(args="")
    # global model, text, data, tok2int, o
    s = ArgParseSettings()
    s.description="rnnparser.jl"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="Input in conllu format. If provided, use first file for training, second for dev, others for test. If single file use both for train and dev.")
        ("--output"; help="Output parse of dev file in conllu format to this file")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--epochs"; arg_type=Int; default=1; help="Number of epochs for training.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[4096]; help="Sizes of parser mlp hidden layers.")
        ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--otrain"; arg_type=Int; default=0; help="Epochs of oracle training.")
        ("--btrain"; arg_type=Int; default=0; help="Epochs of beam training.")
        ("--arctype"; default="ArcHybridR1"; help="Move set to use: ArcEager{R1,13}, ArcHybrid{R1,13}")
        ("--feats"; default="$FEATS"; help="Feature set to use")
        ("--batchsize"; arg_type=Int; default=16; help="Number of sequences to train on in parallel.")
        ("--beamsize"; arg_type=Int; default=1; help="Beam size.")
        ("--dropout"; nargs='+'; arg_type=Float64; default=[0.0]; help="Dropout probabilities.")
        # ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of characters.")
        # ("--embed"; arg_type=Int; default=168; help="Size of the embedding vector.")
        # ("--seqlength"; arg_type=Int; default=100; help="Maximum number of steps to unroll the network for bptt. Initial epochs will use the epoch number as bptt length for faster convergence.")
        # ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        # ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        # ("--fast"; action=:store_true; help="skip loss printing for faster run")
        # ("--bestfile"; help="Save best model to file")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println(s.description)
    println("opts=",[(k,v) for (k,v) in o]...)
    if o[:seed] > 0; srand(o[:seed]); end
    if length(o[:dropout])==1; o[:dropout]=[o[:dropout][1],o[:dropout][1]]; end
    # o[:atype] = eval(parse(o[:atype])) # using cpu for features, gpu for everything else

    global vocab, corpora, wmodel, pmodel
    @msg o[:loadfile]
    d = load(o[:loadfile])

    o[:arctype] = get(d,"arctype",eval(parse(o[:arctype])))
    o[:feats] = get(d,"feats",eval(parse(o[:feats])))

    vocab = makevocab(d)
    wmodel = makewmodel(d)
    corpora = []
    for f in o[:datafiles]; @msg f
        c = loadcorpus(f,vocab)
        push!(corpora,c)
    end
    ppl = fillvecs!(wmodel,vcat(corpora...),vocab)
    @msg "perplexity=$ppl"

    ctrn = corpora[1]
    cdev = length(corpora) > 1 ? corpora[2] : corpora[1]

    @msg :initmodel
    (pmodel,optim) = makepmodel(d,o,ctrn[1])
    save1(file)=savefile(file, vocab, wmodel, pmodel, optim, o[:arctype], o[:feats])
    parentmodel = replace(o[:loadfile],".jld","")
    # save1(@sprintf("%sinit.jld", parentmodel)))

    function report(epoch,beamsize=o[:beamsize])
        las = beamtest(model=pmodel,corpus=cdev,vocab=vocab,arctype=o[:arctype],feats=o[:feats],beamsize=beamsize,batchsize=o[:batchsize])
        println((:epoch,epoch,:beam,beamsize,:las,las))
    end
    # report(0,1)
    @msg :parsing
    report(0,o[:beamsize])

    # training
    gc(); Knet.knetgc(); gc()
    for epoch=1:o[:otrain]
        oracletrain(model=pmodel,optim=optim,corpus=ctrn,vocab=vocab,arctype=o[:arctype],feats=o[:feats],batchsize=o[:batchsize],pdrop=o[:dropout])
        report("oracle$epoch",1); # save1(@sprintf("oracle%02d.jld",epoch))
    end
    gc(); Knet.knetgc(); gc()
    for epoch=1:o[:btrain]
        beamtrain(model=pmodel,optim=optim,corpus=ctrn,vocab=vocab,arctype=o[:arctype],feats=o[:feats],beamsize=o[:beamsize],pdrop=o[:dropout],batchsize=1) # larger batchsizes slow down beamtrain considerably
        report("beam$epoch"); # save1(@sprintf("%sbeam%02d.jld",parentmodel,epoch))
    end

    # savemodel
    if o[:savefile] != nothing; save1(o[:savefile]); end

    # output dev parse: this will be ready because of report()
    if o[:output] != nothing    # TODO: parse all data files?
        inputfile = length(corpora) > 1 ? o[:datafiles][2] : o[:datafiles][1]
        writeconllu(cdev, inputfile, o[:output])
    end
end



function beamtrain(;model=_model, optim=_optim, corpus=_corpus, vocab=_vocab, arctype=ArcHybridR1, feats=FEATS, beamsize=4, pdrop=(0,0), batchsize=1) # larger batchsizes slow down beamtrain considerably
    # global grads, optim, sentbatches, sentences
    # srand(1)
    sentbatches = minibatch(corpus,batchsize; maxlen=MAXSENT, minlen=MINSENT, shuf=true)
    if LOGGING > 0
        nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
        nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
        @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
    end
    nwords = StopWatch()
    nsteps = Any[0,0]
    # niter = 0
    for sentences in sentbatches
        #DBG gc(); Knet.knetgc(); gc()
        #DBG @sho niter+=1
        #DBG @sho length(sentences[1])
        #DBG ploss = beamloss(model, sentences, vocab, arctype, feats, beamsize)
        grads = beamgrad(model, sentences, vocab, arctype, feats, beamsize; earlystop=true, steps=nsteps, pdrop=pdrop)
        update!(model, grads, optim)
        #print('.')
        #DBG gc()
        #DBG Knet.memdbg()
        #DBG Knet.gpuinfo(n=10)
        #DBG readline()
        nw = sum(map(length,sentences))
        speed = inc(nwords, nw)
        if speed != nothing
            date("$(nwords.ncurr) words $(round(Int,speed)) wps $(round(Int,100*nsteps[1]/nsteps[2]))% steps")
            nsteps[:] = 0
            gc(); Knet.knetgc(); gc()
        end
    end
    println()
end

function beamtest(;model=_model, corpus=_corpus, vocab=_vocab, arctype=ArcHybridR1, feats=FEATS, beamsize=4, batchsize=128) # large batchsize does not slow down beamtest
    for s in corpus; s.parse = nothing; end
    sentbatches = minibatch(corpus,batchsize)
    for sentences in sentbatches
        beamloss(model, sentences, vocab, arctype, feats, beamsize; earlystop=false)
        #print('.')
    end
    #println()
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

function splitmodel(pmodel)
    # optimization: do all getindex operations outside, otherwise each getindex creates a new node
    # TODO: fix this in general
    mlpmodel = Any[]
    mlptemp = parserv(pmodel)
    for i=1:length(mlptemp); push!(mlpmodel, mlptemp[i]); end
    featmodel = Array(Any,5)
    for k in 1:5 # (:postag,:deprel,:lcount,:rcount,:distance)
        featmodel[k] = Any[]
        pmodel_k = pmodel[k]
        for i in 1:length(pmodel_k)
            push!(featmodel[k], pmodel_k[i])
        end
    end
    return (featmodel,mlpmodel)
end

# parse a minibatch of sentences using beam search, global normalization, early updates and report loss.
# leave the results in sentences[i].parse, return per sentence loss
# beamloss(_model, _model, _corpus, _vocab, ArcHybridR1, Flist.hybrid25, 10)
# function beamloss(pmodel, cmodel, sentences, vocab, arctype, feats, beamsize; earlystop=false)
function beamloss(pmodel, sentences, vocab, arctype, feats, beamsize; earlystop=true, steps=nothing, pdrop=(0,0))
    # global parsers,fmatrix,beamends,cscores,pscores,parsers0,beamends0,totalloss,loss,pcosts,pcosts0
    # fillvecs!(cmodel, sentences, vocab)
    parsers = parsers0 = map(arctype, sentences)
    beamends = beamends0 = collect(1:length(parsers)) # marks end column of each beam, initially one parser per beam
    pcosts  = pcosts0  = zeros(Int, length(sentences))
    pscores = pscores0 = zeros(FTYPE, length(sentences))
    totalloss = stepcount = 0
    featmodel,mlpmodel = splitmodel(pmodel)

    while length(beamends) > 0
        # features (vcat) are faster on cpu, mlp is faster on gpu
        fmatrix = features(parsers, feats, featmodel) # nfeat x nparser
        if GPUFEATURES && gpu()>=0 #GPU
            @assert isa(getval(fmatrix),KnetArray{FTYPE,2})
        else #CPU
            @assert isa(getval(fmatrix),Array{FTYPE,2})
            if gpu()>=0; fmatrix = KnetArray(fmatrix); end
        end
        cscores = Array(mlp(mlpmodel, fmatrix; pdrop=pdrop)) # candidate scores: nmove x nparser
        # @show (findmax(cscores),cscores)
        cscores = cscores .+ pscores' # candidate cumulative scores
        # @show (findmax(cscores),cscores)
        parsers,pscores,pcosts,beamends,loss = nextbeam(parsers, cscores, pcosts, beamends, beamsize; earlystop=earlystop)
        totalloss += loss
        stepcount += 1
    end
    # emptyvecs!(sentences)       # if we don't empty, gc cannot clear these vectors; empty if finetuning wvecs
    # if earlystop; @msg ((maximum(map(length,sentences)),stepcount)); end
    if steps != nothing
        steps[1] += stepcount
        steps[2] += length(sentences[1])*2-2
    end
    # println(stepcount)
    return totalloss / length(sentences)
end

beamgrad = grad(beamloss)

function nextbeam(parsers, mscores, pcosts, beamends, beamsize; earlystop=true)
    #global mcosts
    n = beamsize * length(beamends) + 1
    newparsers, newscores, newcosts, newbeamends, loss = Array(Any,n),Array(Int,n),Array(Int,n),Int[],0.0
    nmoves,nparsers = size(mscores)                     # mscores[m,p] is the score of move m for parser p
    #TEST: will not have mcosts
    mcosts = Array(Any, nparsers)                       # mcosts[p][m] will be the cost vector for parser[p],move[m] if needed
    n = p0 = 0
    for p1 in beamends                                  # parsers[p0+1:p1], mscores[:,p0+1:p1] is the current beam belonging to a common sentence
        s0,s1 = 1 + nmoves*p0, nmoves*p1                # mscores[s0:s1] are the linear indices for mscores[:,p0:p1]
        nsave = n                                       # newparsers,newscores,newcosts[nsave+1:n] will the new beam for this sentence
        #TEST: will not have ngold
        ngold = 0                                       # ngold, if nonzero, will be the index of the gold path in beam
        sorted = sortperm(getval(mscores)[s0:s1], rev=true)	
        for isorted in sorted
            linidx = isorted + s0 - 1
            (move,parent) = ind2sub(size(mscores), linidx) # find cartesian index of the next best score
            parser = parsers[parent]
            if !moveok(parser,move); continue; end  # skip illegal move
            #TEST: will not have mcosts
            if earlystop && !isassigned(mcosts, parent) # not every parent may have children, avoid unnecessary movecosts
                mcosts[parent] = movecosts(parser, parser.sentence.head, parser.sentence.deprel)
            end
            n += 1
            newparsers[n] = copy(parser); move!(newparsers[n], move)
            newscores[n] = linidx
            #TEST: no newcosts during test
            if earlystop
                newcosts[n] = pcosts[parent] + mcosts[parent][move]
                if newcosts[n] == 0
                    if ngold==0
                        ngold=n
                    else
                        # @msg("multiple gold moves for $(parser.sentence)")
                    end
                end
            end
            if n-nsave == beamsize; break; end
        end
        if false # earlystop #DBG
            mvmax = newscores[nsave+1]; mvgold = goldindex(parsers,pcosts,mcosts,(p0+1):p1)
            mvmaxscore = mscores[mvmax]; mvmaxcost = newcosts[nsave+1]
            @sho (mvgold,mvmax,mvmaxcost,mvmaxscore)
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
            gindex = goldindex(parsers,pcosts,mcosts,(p0+1):p1)
            if gindex != 0
                newscores[n+1] = gindex
                loss = loss - mscores[gindex] + logsumexp2(mscores, newscores[nsave+1:n+1])
            end
            n = nsave
        elseif endofparse(newparsers[n])                # all parsers in beam have finished, gold among them if earlystop
            #TEST: there will be no ngold during test, cannot return beamloss, just return the highest scoring parse, no need for normalization
            if earlystop
                gindex = newscores[ngold]
                loss = loss - mscores[gindex] + logsumexp2(mscores, newscores[nsave+1:n])
            end
            s = newparsers[n].sentence
            # @assert s == newparsers[nsave+1].sentence
            # @assert mscores[newscores[nsave+1]] >= mscores[newscores[n]] "s[$(1+nsave)]=$(mscores[newscores[nsave+1]]) s[$n]=$(mscores[newscores[n]])"
            s.parse = newparsers[nsave+1]
            n = nsave                                   # do not add finished parsers to new beam
        else                                            # all good keep going
            push!(newbeamends, n)
        end
        p0 = p1
    end
    return newparsers[1:n], mscores[newscores[1:n]], newcosts[1:n], newbeamends, loss
end

function logsumexp2(a,r)
    # z = 0; amax = a[r[1]]
    # for i in r; z += exp(a[i]-amax); end
    # return log(z) + amax
    amax = getval(a)[r[1]]
    log(sum(exp(a[r] - amax))) + amax
end

function goldindex(parsers,pcosts,mcosts,beamrange)
    parent = findfirst(view(pcosts,beamrange),0)
    if parent == 0; error("cannot find gold parent in $beamrange"); end
    parent += first(beamrange) - 1
    if !isassigned(mcosts, parent)
        p = parsers[parent]
        mcosts[parent] = movecosts(p, p.sentence.head, p.sentence.deprel)
    end
    move = findfirst(mcosts[parent],0)
    if move == 0
        # @msg("cannot find gold move for $(parsers[parent].sentence)")
        return 0
    else
        msize = (parsers[1].nmove,length(parsers))
        return sub2ind(msize,move,parent)
    end
end

endofparse(p)=(p.sptr == 1 && p.wptr > p.nword)

function oracletrain(;model=_model, optim=_optim, corpus=_corpus, vocab=_vocab, arctype=ArcHybridR1, feats=FEATS, batchsize=16, maxiter=typemax(Int), pdrop=(0,0))
    # global grads, optim, sentbatches, sentences
    # srand(1)
    sentbatches = minibatch(corpus,batchsize; maxlen=MAXSENT, minlen=MINSENT, shuf=true)
    nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
    nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
    @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
    nwords = StopWatch()
    losses = Any[0,0,0]
    niter = 0
    @time for sentences in sentbatches
        grads = oraclegrad(model, sentences, vocab, arctype, feats; losses=losses, pdrop=pdrop)
        update!(model, grads, optim)
        nw = sum(map(length,sentences))
        if (speed = inc(nwords, nw)) != nothing
            date("$(nwords.ncurr) words $(round(Int,speed)) wps $(losses[3]) avgloss")
            gc(); Knet.knetgc(); gc()
        end
        if (niter+=1) >= maxiter; break; end
    end
    println()
end

# function oracleloss(pmodel, cmodel, sentences, vocab, arctype, feats)
function oracleloss(pmodel, sentences, vocab, arctype, feats; losses=nothing, pdrop=(0,0))
    # global parsers,mcosts,parserdone,fmatrix,scores,logprob,totalloss
    # fillvecs!(cmodel, sentences, vocab)
    parsers = map(arctype, sentences)
    mcosts = Array(Cost, parsers[1].nmove)
    parserdone = falses(length(parsers))
    totalloss = 0
    featmodel,mlpmodel = splitmodel(pmodel)

    while !all(parserdone)
        fmatrix = features(parsers, feats, featmodel)
        if GPUFEATURES && gpu()>=0 #GPU
            @assert isa(getval(fmatrix),KnetArray{FTYPE,2})
        else #CPU
            @assert isa(getval(fmatrix),Array{FTYPE,2})
            if gpu()>=0; fmatrix = KnetArray(fmatrix); end
        end
        scores = mlp(mlpmodel, fmatrix; pdrop=pdrop)
        logprob = logp(scores,MAJOR)
        for (i,p) in enumerate(parsers)
            if parserdone[i]; continue; end
            movecosts(p, p.sentence.head, p.sentence.deprel, mcosts)
            goldmove = indmin(mcosts)
            if mcosts[goldmove] == typemax(Cost)
                parserdone[i] = true
                p.sentence.parse = p
            else
                totalloss -= logprob[goldmove,i]
                move!(p, goldmove)
                if losses != nothing
                    loss1 = -getval(logprob)[goldmove,i]
                    losses[1] += loss1
                    losses[2] += 1
                    if losses[2] < 1000
                        losses[3] = losses[1]/losses[2]
                    else
                        losses[3] = 0.999 * losses[3] + 0.001 * loss1
                    end
                end
            end
        end
    end
    return totalloss / length(sentences)
end

oraclegrad = grad(oracleloss)

using AutoGrad
import Base: sortperm, ind2sub
@zerograd sortperm(a;o...)
@zerograd ind2sub(a,i...)

function mlp(w,x; pdrop=(0,0))
    dropout(x,pdrop[1])
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
        dropout(x,pdrop[2])
    end
    return w[end-1]*x .+ w[end]
end

function fillvecs!(model, sentences, vocab; batchsize=128)
    global iwords,isents,maxword,maxsent,sow,eow,cdata,cmask,wembed,sos,eos,wdata,wmask,forw,back
    iwords,isents,maxword,maxsent = maptoint(sentences, vocab)

    # Get word embeddings: do this in minibatches otherwise may run out of memory
    @msg :wordembeddings
    sow,eow = vocab.cdict[vocab.sowchar],vocab.cdict[vocab.eowchar]
    wembed = Any[]
    gc();Knet.knetgc();gc()
    for i=1:batchsize:length(iwords)    
        j=min(i+batchsize-1,length(iwords))
        wij = view(iwords,i:j)
        maxij = maximum(map(length,wij))
        cdata,cmask = tokenbatch(wij,maxij,sow,eow) # cdata/cmask[T+2][V] where T: longest word (140), V: input word vocab (19674)
        push!(wembed, charlstm(model,cdata,cmask))    # wembed[C,V] where V: input word vocab, C: charlstm hidden (350)
    end
    wembed = (MAJOR==1 ? hcatn(wembed...) : vcatn(wembed...))
    @msg :fillwvecs!
    fillwvecs!(sentences, isents, wembed)
    #DBG @sho (maxword,length(iwords),size(wembed)); Knet.gpuinfo(n=10); readline()

    # Get context embeddings
    @msg "contextembeddings,fillcvecs!,lmloss"
    sos,eos,unk = vocab.idict[vocab.sosword],vocab.idict[vocab.eosword],vocab.odict[vocab.unkword]
    result = zeros(2)
    gc();Knet.knetgc();gc()
    for i=1:batchsize:length(isents)
        j=min(i+batchsize-1,length(isents))
        isentij = view(isents,i:j)
        maxij = maximum(map(length,isentij))
        wdata,wmask = tokenbatch(isentij,maxij,sos,eos) # wdata/wmask[T+2][B] where T: longest sentence (159), B: batch size (12543)
        forw,back = wordlstm(model,wdata,wmask,wembed)  # forw/back[T][W,B] where T: longest sentence, B: batch size, W: wordlstm hidden (300)
        sentij = view(sentences,i:j)
        fillcvecs!(sentij,forw,back)
        odata,omask = goldbatch(sentij,maxij,vocab.odict,unk)
        lmloss1(model,odata,omask,forw,back; result=result)
    end

    # Get embeddings into sentences -- must empty later for gc if we are finetuning!
    # @msg :fillvecs
    # gc();Knet.knetgc();gc()
    # fillcvecs!(sentences, forw, back)

    # Test predictions
    # gc();Knet.knetgc();gc()
    # @msg :losscalc
    # result = zeros(2)
    # unk = vocab.odict[vocab.unkword]
    # odata,omask = goldbatch(sentences,maxsent,vocab.odict,unk)
    # gc();Knet.knetgc();gc()
    # total = lmloss1(model,odata,omask,forw,back; result=result)
    return exp(-result[1]/result[2])
end

function emptyvecs!(sentences)
    for s in sentences   
        empty!(s.wvec); empty!(s.fvec); empty!(s.bvec)
    end
end    

function fillwvecs!(sentences, isents, wembed)
    if MAJOR != 1; error(); end # not impl yet
    @inbounds for (s,isents) in zip(sentences,isents)
        empty!(s.wvec)
        for w in isents
            if GPUFEATURES #GPU
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
            if GPUFEATURES #GPU
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
treemap(f,::Void)=nothing

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
    if result != nothing; result[1]+=getval(total); result[2]+=count; end
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
    mask = [ Array(FTYPE,B) for t in 1:T ]
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
    mask = [ Array(FTYPE,B) for t in 1:T ]
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
    weight,bias,embeddings = wchar(model),bchar(model),cembed(model)
    T = length(data)
    B = length(data[1])
    H = div(length(bias),4)
    cembed_t(t)=(if MAJOR==1; embeddings[:,data[t]]; else; embeddings[data[t],:]; end)
    czero=(if MAJOR==1; fill!(similar(bias,H,B), 0); else; fill!(similar(bias,B,H), 0); end)
    hidden = cell = czero       # TODO: cache this
    if isa(weight,KnetArray); mask = map(KnetArray, mask); end
    @inbounds for t in 1:T
        (hidden,cell) = lstm(weight,bias,hidden,cell,cembed_t(t);mask=mask[t])
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
    if isa(weight,KnetArray); mask = map(KnetArray, mask); end
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

function loadcorpus(file,v::Vocab)
    corpus = Any[]
    s = Sentence(v)
    for line in eachline(file)
        if line == "\n"
            push!(corpus, s)
            s = Sentence(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing
            #                id   word   lem  upos   xpos feat head   deprel
            word = m.captures[1]
            postag = get(v.postags, m.captures[2], 0)
            if postag==0; Base.warn_once("Unknown postags"); end
            head = tryparse(Position, m.captures[3])
            head = isnull(head) ? 0 : head.value
            if head==0; Base.warn_once("Unknown heads"); end
            deprel = get(v.deprels, m.captures[4], 0)
            if deprel==0; Base.warn_once("Unknown deprels"); end
            push!(s.word, word)
            push!(s.postag, postag)
            push!(s.head, head)
            push!(s.deprel, deprel)
        end
    end
    return corpus
end

function minibatch(corpus, batchsize; maxlen=typemax(Int), minlen=1, shuf=false)
    data = Any[]
    sorted = sort(corpus, by=length)
    i1 = findfirst(x->(length(x) >= minlen), sorted)
    if i1==0; error("No sentences >= $minlen"); end
    i2 = findlast(x->(length(x) <= maxlen), sorted)
    if i2==0; error("No sentences <= $maxlen"); end
    for i in i1:batchsize:i2
        j = min(i2, i+batchsize-1)
        push!(data, sorted[i:j])
    end
    if shuf; data=shuffle(data); end
    return data
end

function inc(s::StopWatch, n, step=1000)
    s.ncurr += n
    if s.ncurr >= s.nnext
        tcurr = time()
        dt = tcurr - s.tstart
        dn = s.ncurr - s.nstart
        s.tstart = tcurr
        s.nstart = s.ncurr
        s.nnext += step
        return dn/dt
    end
end

# when fmatrix has mixed Rec and KnetArray, vcat does not do the right thing!  AutoGrad only looks at the first 2-3 elements!
#TODO: temp solution to AutoGrad vcat issue:
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

# omer's: model file format:
# d = load("model.jld"); m=d["model"]; m[:cembed]=Array; wv=d["word_vocab"]; wv::Dict{String,Int}
# cembed: Array
# forw,back,soft,char: [ Array, Array ]
# char_vocab: Dict{Char,Int}
# word_vocab: Dict{String,Int}

# proposed changes: use a flat structure for everything (so there is no model, "cembed" etc become direct keys)
# combine vocab and model fields in the same file
# write a converter that converts omer's format to this new format
# in addition to the fields above, the following fields will be added if not found:
# sosword,eosword,unkword: String (default: <s>,</s>,<unk>)
# sowchar,eowchar,unkchar: Char (default: PAD=0x11, SOW=0x12, EOW=0x13)
# postags, deprels: (default:UPOSTAG(17),UDEPREL(37))
# postagv(17), deprelv(37), lcountv(10), rcountv(10), distancev(10): (default: rand init embeddings)
# parser: Array(n) (default: rand init mlp weights)
# optim: optimization parameters

# load these as separate models for now, we could change this in the future
makewmodel(d)=(d1=makewmodel1(d); if gpu()>=0; map(KnetArray,d1); else; map(Array,d1); end)
makewmodel1(d)=[d["cembed"],d["char"][1],d["char"][2],d["forw"][1],d["forw"][2],d["back"][1],d["back"][2],d["soft"][1],d["soft"][2]]
cembed(m)=m[1]; wchar(m)=m[2]; bchar(m)=m[3];wforw(m)=m[4]; bforw(m)=m[5]; wback(m)=m[6]; bback(m)=m[7]; wsoft(m)=m[8]; bsoft(m)=m[9]
makevocab(d)=Vocab(d["char_vocab"],Dict{String,Int}(),d["word_vocab"],d["sosword"],d["eosword"],d["unkword"],d["sowchar"],d["eowchar"],d["unkchar"],get(d,"postags",UPOSTAG),get(d,"deprels",UDEPREL))

makepmodel(d,o,s)=(haskey(d,"parserv") ? makepmodel1(d) : makepmodel2(o,s))
postagv(m)=m[1]; deprelv(m)=m[2]; lcountv(m)=m[3]; rcountv(m)=m[4]; distancev(m)=m[5]; parserv(m)=m[6]

function makepmodel1(d)
    m = ([d["postagv"],d["deprelv"],d["lcountv"],d["rcountv"],d["distancev"],d["parserv"]],
         [d["postago"],d["deprelo"],d["lcounto"],d["rcounto"],d["distanceo"],d["parsero"]])
    if gpu() >= 0
        if GPUFEATURES
            map2gpu(m)
        else
            m = map2cpu(m)
            m[1][6] = map2gpu(m[1][6])
            m[2][6] = map2gpu(m[2][6])
            return m
        end
    else
        map2cpu(m)
    end
end

function makepmodel2(o,s)
    initx(d...) = (if gpu()>=0; KnetArray{FTYPE}(xavier(d...)); else; Array{FTYPE}(xavier(d...)); end)
    initr(d...) = (if GPUFEATURES && gpu()>=0; KnetArray{FTYPE}(0.1*randn(d...)); else; Array{FTYPE}(0.1*randn(d...)); end)
    model = Any[]
    for (k,n,d) in ((:postag,17,17),(:deprel,37,37),(:lcount,10,10),(:rcount,10,10),(:distance,10,10))
        push!(model, [ initr(d) for i=1:n ])
    end
    p = o[:arctype](s)
    f = features([p], o[:feats], model)
    mlpdims = (length(f),o[:hidden]...,p.nmove)
    @msg "mlpdims=$mlpdims"
    parser = Any[]
    for i=2:length(mlpdims)
        push!(parser, initx(mlpdims[i],mlpdims[i-1]))
        push!(parser, initx(mlpdims[i],1))
    end
    push!(model,parser)
    optim = initoptim(model,o[:optimization])
    return model,optim
end

# convert omer's format to new format
function convertfile(infile, outfile)
    atr(x)=transpose(Array(x)) # omer stores in row-major Array, we use col-major Array
    d = load(infile); m = d["model"]
    save(outfile, "cembed", atr(m[:cembed]), "forw", map(atr,m[:forw]),
         "back",map(atr,m[:back]), "soft",map(atr,m[:soft]), "char",map(atr,m[:char]),
         "char_vocab",d["char_vocab"], "word_vocab",d["word_vocab"], 
         "sosword","<s>","eosword","</s>","unkword","<unk>",
         "sowchar",'\x12',"eowchar",'\x13',"unkchar",'\x11')
end

map2cpu(x)=(if isbits(x); x; else; map2cpu2(x); end)
map2cpu(x::KnetArray)=Array(x)
map2cpu(x::Tuple)=map(map2cpu,x)
map2cpu(x::AbstractString)=x
map2cpu(x::DataType)=x
map2cpu(x::Array)=map(map2cpu,x)
map2cpu{T<:Number}(x::Array{T})=x
map2cpu(x::Associative)=(y=Dict();for (k,v) in x; y[k] = map2cpu(x[k]); end; y)
map2cpu2(x)=(y=deepcopy(x); for f in fieldnames(x); setfield!(y,f,map2cpu(getfield(x,f))); end; y)

map2gpu(x)=(if isbits(x); x; else; map2gpu2(x); end)
map2gpu(x::KnetArray)=x
map2gpu(x::AbstractString)=x
map2gpu(x::DataType)=x
map2gpu(x::Tuple)=map(map2gpu,x)
map2gpu(x::Array)=map(map2gpu,x)
map2gpu{T<:AbstractFloat}(x::Array{T})=KnetArray(x)
map2gpu(x::Associative)=(y=Dict();for (k,v) in x; y[k] = map2gpu(x[k]); end; y)
map2gpu2(x)=(y=deepcopy(x); for f in fieldnames(x); setfield!(y,f,map2gpu(getfield(x,f))); end; y)

# convert KnetArrays to Arrays
function convert2cpu(infile, outfile)
    d = load(infile)
    jldopen(outfile, "w") do file
        for (k,v) in d
            write(file,k,map2cpu(v))
        end
    end
end

# save file in new format
function savefile(file, vocab, wmodel, pmodel, optim, arctype, feats)
    save(file,  "cembed", map2cpu(cembed(wmodel)),
         "char", map2cpu([wchar(wmodel),bchar(wmodel)]),
         "forw", map2cpu([wforw(wmodel),bforw(wmodel)]),
         "back", map2cpu([wback(wmodel),bback(wmodel)]),
         "soft", map2cpu([wsoft(wmodel),bsoft(wmodel)]),

         "char_vocab",vocab.cdict, "word_vocab",vocab.odict,
         "sosword",vocab.sosword,"eosword",vocab.eosword,"unkword",vocab.unkword,
         "sowchar",vocab.sowchar,"eowchar",vocab.eowchar,"unkchar",vocab.unkchar,
         "postags",vocab.postags,"deprels",vocab.deprels,

         "postagv",map2cpu(postagv(pmodel)),"deprelv",map2cpu(deprelv(pmodel)),
         "lcountv",map2cpu(lcountv(pmodel)),"rcountv",map2cpu(rcountv(pmodel)),
         "distancev",map2cpu(distancev(pmodel)),"parserv",map2cpu(parserv(pmodel)),

         "postago",map2cpu(postagv(optim)),"deprelo",map2cpu(deprelv(optim)),
         "lcounto",map2cpu(lcountv(optim)),"rcounto",map2cpu(rcountv(optim)),
         "distanceo",map2cpu(distancev(optim)),"parsero",map2cpu(parserv(optim)),
    
         "arctype",arctype,"feats",feats,
    )
end

function writeconllu(sentences, inputfile, outputfile)
    # We only replace the head and deprel fields of the input file
    out = open(outputfile,"w")
    v = sentences[1].vocab
    deprels = Array(String, length(v.deprels))
    for (k,v) in v.deprels; deprels[v]=k; end
    s = p = nothing
    ns = nw = nl = 0
    for line in eachline(inputfile)
        nl += 1
        if ismatch(r"^\d+\t", line)
            # info("$nl word")
            if s == nothing
                s = sentences[ns+1]
                p = s.parse
            end
            f = split(line, '\t')
            nw += 1
            if f[1] != "$nw"; error(); end
            if f[2] != s.word[nw]; error(); end
            f[7] = string(p.head[nw])
            f[8] = deprels[p.deprel[nw]]
            print(out, join(f, "\t"))
        else
            if line == "\n"
                # info("$nl blank")
                if s == nothing; error(); end
                if nw != length(s.word); error(); end
                ns += 1; nw = 0
                s = p = nothing
            else
                # info("$nl non-word")
            end
            print(out, line)
        end
    end
    if ns != length(sentences); error(); end
    close(out)
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

### DEAD CODE

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

# init()


# oracle1epoch.jld: 0.724 LAS on _corpus[1:128] beamsize=1, 0.5759 with beamsize=10!
# oracle2epoch.jld: 0.759 LAS on _corpus[1:128] beamsize=1, 0.5957 with beamsize=10.
# beam1epoch.jld: 0.5129 beamsize=1, 0.4457 beamsize=10 (starting with fresh optim=Adam after oracle2epoch)
# beam2epoch.jld: 0.4762 beamsize=1, 0.4171 beamsize=10
# beam3epoch.jld: 0.4190 beamsize=1, 0.4147 beamsize=10
# more epochs?  Adam bad? Bug?

# function loadvocab()
#     a = load("english_vocabs.jld")
#     Vocab(a["char_vocab"],Dict{String,Int}(),a["word_vocab"],
#           "<s>","</s>","<unk>",'↥','Ϟ','⋮', UPOSTAG, UDEPREL)
# end

# function init(;batch=3000,nsent=1,result=zeros(2))
#     global LOGGING = 0
#     global _vocab = loadvocab()
#     global _model = loadmodel(_vocab,4096)
#     global _optim = initoptim(_model,"Adam()") # "Sgd(lr=.1)") # 
#     global _corpus = loadcorpus(_vocab)
#     @time fillvecs!(_model, _corpus, _vocab)
#     c = (nsent == 0 ? _corpus : _corpus[1:nsent])
#     _data = minibatch(c, batch)
#     # global _data = [ _corpus[i:min(i+999,length(_corpus))] for i=1:1000:length(_corpus) ]
#     # global _data = [ _corpus[1:1] ]
#     for d in _data
#         lmloss(_model, d, _vocab; result=result)
#     end
#     println(exp(-result[1]/result[2]))
#     return result
# end

# function hiddenoptim(optim,hidden...;epochs=1)
#     srand(1)
#     global _model, _optim
#     _model = loadmodel(_vocab, hidden...)
#     _optim = initoptim(_model, optim)
#     c = _corpus[1:128]
#     for i=1:epochs; oracletrain(corpus=c); end
#     beamtest(corpus=c,beamsize=1)
# end

# # function loss(model, sentences, arctype, beamsize)
# function lmloss(model, sentences, vocab::Vocab; result=nothing)
#     # map words and chars to Ints
#     # words and sents contain Int (char/word id) arrays without sos/eos tokens
#     # TODO: optionally construct cdict here as well
#     # words are the char ids for each unique word
#     # sents are the word ids for each sentence
#     # wordlen and sentlen are the maxlen word and sentence
#     words,sents,wordlen,sentlen = maptoint(sentences, vocab)

#     # Find word vectors for a batch of sentences using character lstm or cnn model
#     # note that tokenbatch adds sos/eos tokens and padding to each sequence
#     sow,eow = vocab.cdict[vocab.sowchar],vocab.cdict[vocab.eowchar]
#     cdata,cmask = tokenbatch(words,wordlen,sow,eow)
#     wembed = charlstm(model,cdata,cmask)

#     # Find context vectors using word blstm
#     sos,eos = vocab.idict[vocab.sosword],vocab.idict[vocab.eosword]
#     wdata,wmask = tokenbatch(sents,sentlen,sos,eos)
#     forw,back = wordlstm(model,wdata,wmask,wembed)

#     # Test predictions
#     unk = vocab.odict[vocab.unkword]
#     odata,omask = goldbatch(sentences,sentlen,vocab.odict,unk)
#     total = lmloss1(model,odata,omask,forw,back; result=result)
#     return -total/length(sentences)
# end

# function loadmodel(v::Vocab, hidden...)
#     model = Dict()
#     m = load("english_ch12kmodel_2.jld")["model"]
#     if MAJOR==1
#         for k in (:cembed,); model[k] = KnetArray{FTYPE}(m[k]'); end
#         for k in (:forw,:soft,:back,:char); model[k] = map(KnetArray{FTYPE}, m[k]'); end
#     else
#         for k in (:cembed,); model[k] = KnetArray{FTYPE}(m[k]); end
#         for k in (:forw,:soft,:back,:char); model[k] = map(KnetArray{FTYPE}, m[k]); end
#     end
#     initk(d...) = KnetArray{FTYPE}(xavier(d...))
#     inita(d...) = Array{FTYPE}(xavier(d...))
#     for (k,n,d) in ((:postag,17,17),(:deprel,37,37),(:lcount,10,10),(:rcount,10,10),(:distance,10,10))
#         if GPUFEATURES #GPU
#             model[k] = [ initk(d) for i=1:n ]
#         else #CPU
#             model[k] = [ inita(d) for i=1:n ]
#         end
#     end
#     parsedims = (5796,hidden...,73)
#     model[:parser] = Any[]
#     for i=2:length(parsedims)
#         push!(model[:parser], initk(parsedims[i],parsedims[i-1]))
#         push!(model[:parser], initk(parsedims[i],1))
#     end
#     return model
# end

# function getloss(d; result=zeros(2))
#     lmloss(_model, d, _vocab; result=result)
#     println(exp(-result[1]/result[2]))
#     return result
# end

if PROGRAM_FILE=="rnnparser.jl"
    main(ARGS)
end

nothing

