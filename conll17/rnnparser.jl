using JLD, Knet, KUparser
MAJOR=1 # 1:column-major 2:row-major
logging=1
macro msg(_x) :(if logging>0; join(STDERR,[Dates.format(now(),"HH:MM:SS"), $_x,'\n'],' '); end) end
macro log(_x) :(@msg($(string(_x))); $(esc(_x))) end


# parse a minibatch of sentences using beam search, global normalization, early updates and report loss.
# parseloss(_model, _corpus, _vocab, ArcHybridR1, Flist.hybrid25, 10)
function parseloss(model, sentences, vocab, parsertype, feats, beamsize)
    global cids,wids,maxword,maxsent,sow,eow,cdata,cmask,wembed,sos,eos,wdata,wmask,forw,back
    @log cids,wids,maxword,maxsent = maptoint(sentences, vocab)
    # Get word embeddings
    @log sow,eow = vocab.cdict[vocab.sowchar],vocab.cdict[vocab.eowchar]
    @log cdata,cmask = tokenbatch(cids,maxword,sow,eow) # cdata/cmask[T+2][V] where T: longest word (140), V: input word vocab (19674)
    @log wembed = charlstm(model,cdata,cmask)            # wembed[C,V] where V: input word vocab, C: charlstm hidden (350)
    # Get context embeddings
    @log sos,eos = vocab.idict[vocab.sosword],vocab.idict[vocab.eosword]
    @log wdata,wmask = tokenbatch(wids,maxsent,sos,eos) # wdata/wmask[T+2][B] where T: longest sentence (159), B: batch size (12543)
    @log forw,back = wordlstm(model,wdata,wmask,wembed)  # forw/back[T][W,B] where T: longest sentence, B: batch size, W: wordlstm hidden (300)
    # Get embeddings into sentences
    @log fillwvecs!(sentences, wids, wembed)
    @log fillcvecs!(sentences, forw, back)
    # Initialize parsers
    global parsers,fmatrix,beamend,cscores,pscores,parsers2,beamend2
    @log parsers = map(parsertype, sentences)
    @log fmatrix = features(parsers, feats, model)
    @log beamend = collect(1:length(parsers)) # marks end column of each beam, initially one parser per beam
    # Predict scores
    # no need: @log pscores = fill!(similar(fmatrix,1,size(fmatrix,2)),0) # parent cumulative scores, initially 0
    @log cscores = Array(mlp(model[:parser], fmatrix))     # candidate cumulative scores
    # Sort scores get next beam
    @log parsers2,pscores,beamend2 = nextbeam(parsers, cscores, beamend, beamsize)
end

function nextbeam(parsers, scores, beamend, beamsize)
    newparsers, newscores, newbeamend = Any[], Float32[], Int[]
    nrows,ncols = nmoves,nparsers = size(scores)
    @assert nparsers == length(parsers)
    @assert nmoves == parsers[1].nmove
    global cost = Array(Any, nparsers)
    p0 = 1; n1 = 0
    for p1 in beamend                                   # parsers[p0:p1], scores[:,p0:p1] is the next beam
        #n0 = n1 + 1                                    # newparsers[n0:n1],newscores[n0:n1] is going to be the next beam, n0 not used
        s0,s1 = 1 + nrows*(p0-1), nrows*p1              # scores[s0:s1] are the linear indices for scores[:,p0:p1]
        sorted = sortperm(view(scores,s0:s1), rev=true)	
        for isorted in sorted
            i = isorted + s0 - 1                        # linear index of the next best score
            (move,parser) = ind2sub(scores, i)
            parent = parsers[parser]
            if !moveok(parent,move); continue; end
            p = copy(parent)
            move!(p, move)
            # TODO: update cost, check early stop, regular stop
            push!(newparsers,p)
            push!(newscores,scores[i])
            if length(newparsers) == n1 + beamsize; break; end
        end
        p0 = p1+1
        n1 = length(newparsers)
        push!(newbeamend, n1)
    end
    return newparsers, newscores, newbeamend
end

function mlp(w,x)
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

function fillwvecs!(sentences, wids, wembed)
    if MAJOR != 1; error(); end # not impl yet
    @inbounds for (s,wids) in zip(sentences,wids)
        empty!(s.wvec)
        for w in wids
            push!(s.wvec, wembed[:,w])
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
            push!(s.fvec, forw[t][:,i])
            push!(s.bvec, back[t][:,i])
        end
    end
end

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
    for line in eachline("en-ud-train.conllu")
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

function loadmodel(v::Vocab)
    model = Dict()
    m = load("english_ch12kmodel_2.jld")["model"]
    if MAJOR==1
        for k in (:cembed,); model[k] = KnetArray(m[k]'); end
        for k in (:forw,:soft,:back,:char); model[k] = map(KnetArray, m[k]'); end
    else
        for k in (:cembed,); model[k] = KnetArray(m[k]); end
        for k in (:forw,:soft,:back,:char); model[k] = map(KnetArray, m[k]); end
    end
    initv(d...) = KnetArray{Float32}(xavier(d...))
    for (k,n,d) in ((:postag,17,17),(:deprel,37,37),(:lcount,10,10),(:rcount,10,10),(:distance,10,10))
        model[k] = [ initv(d) for i=1:n ]
    end
    parsedims = (5796,256,73)
    model[:parser] = Any[]
    for i=2:length(parsedims)
        push!(model[:parser], initv(parsedims[i],parsedims[i-1]))
        push!(model[:parser], initv(parsedims[i],1))
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

function minibatch(corpus, batchsize)
    data = Any[]
    sorted = sort(corpus, by=length)
    for i in 1:batchsize:length(corpus)
        j = min(length(corpus), i+batchsize-1)
        push!(data, sorted[i:j])
    end
    shuffle(data)
end

function minibatch1(corpus, batchsize)
    data = Any[]
    dict = Dict{Int,Vector{Int}}()
    for i in 1:length(corpus)
        s = corpus[i]
        l = length(s)
        a = get!(dict, l, Int[])
        push!(a, i)
        if length(a) == batchsize
            push!(data, corpus[a])
            empty!(a)
        end
    end
    for (k,a) in dict
        if !isempty(a)
            push!(data, corpus[a])
        end
    end
    return data
end

function main(;batch=3000,nsent=0,result=zeros(2))
    global _vocab = loadvocab()
    global _model = loadmodel(_vocab)
    global _corpus = loadcorpus(_vocab)
    c = (nsent == 0 ? _corpus : _corpus[1:nsent])
    global _data = minibatch(c, batch)
    # global _data = [ _corpus[i:min(i+999,length(_corpus))] for i=1:1000:length(_corpus) ]
    # global _data = [ _corpus[1:1] ]
    for d in _data
        lmloss(_model, d, _vocab; result=result)
    end
    println(exp(-result[1]/result[2]))
    return result
end

function getloss(d; result=zeros(2))
    lmloss(_model, d, _vocab; result=result)
    println(exp(-result[1]/result[2]))
    return result
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

main(nsent=1)
nothing
