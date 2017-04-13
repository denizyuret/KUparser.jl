using JLD, Knet
ORDER=1 # 1:column-major 2:row-major

typealias Word String           # represent words with strings instead of finite vocab, we will process chars
# typealias PosTag UInt8          # 17 universal part-of-speech tags
# typealias DepRel UInt8          # 37 universal dependency relations
# typealias Position Int16        # sentence position

immutable Vocab
    cdict::Dict{Char,Int}       # character dictionary (input)
    idict::Dict{String,Int}     # word dictionary (input, filled in by maptoint)
    odict::Dict{String,Int}     # word dictionary (output)
    sosword::String             # start-of-sentence word (input)
    eosword::String             # end-of-sentence word (input)
    unkword::String             # unknown word (output, input does not have unk)
    sowchar::Char               # start-of-word char
    eowchar::Char               # end-of-word char
    unkchar::Char               # unknown char
end

;                               # CONLLU FORMAT
immutable Sentence		# 1. ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
    word::Vector{Word}          # 2. FORM: Word form or punctuation symbol.
    #stem::Vector{Stem}         # 3. LEMMA: Lemma or stem of word form.
    #postag::Vector{PosTag}      # 4. UPOSTAG: Universal part-of-speech tag.
    #xpostag::Vector{Xpostag}   # 5. XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
    #feats::Vector{Feats}       # 6. FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    #head::Vector{Position}      # 7. HEAD: Head of the current word, which is either a value of ID or zero (0).
    #deprel::Vector{DepRel}      # 8. DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    #deps::Vector{Deps}         # 9. DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    #misc::Vector{Misc}         # 10. MISC: Any other annotation.
    #vocab::Vocab                # Common repository of symbols for upostag, deprel etc.
end

# just the words:
#Sentence(w::Vector{Word})=Sentence(w,[],[],[],Vocab([],[],[]))

Base.length(s::Sentence)=length(s.word)

# parse a minibatch of sentences using beam search, global normalization, early updates and report loss.
function parseloss(model, sentences, vocab, parsertype, beamsize)
    # global wembed,forw,back,cdata,cmask,wdata,wmask
    words,sents,wordlen,sentlen = maptoint(sentences, vocab)
    # Get word embeddings
    sow,eow = vocab.cdict[vocab.sowchar],vocab.cdict[vocab.eowchar]
    cdata,cmask = tokenbatch(words,wordlen,sow,eow) # cdata/cmask[W+2][V] where W: longest word (140), V: input word vocab (19674)
    wembed = charlstm(model,cdata,cmask)            # wembed[V,C] where V: input word vocab, C: char embedding (350)
    # Get context embeddings
    sos,eos = vocab.idict[vocab.sosword],vocab.idict[vocab.eosword]
    wdata,wmask = tokenbatch(sents,sentlen,sos,eos) # wdata/wmask[T+2][B] where T: longest sentence (159), B: batch size (12543)
    forw,back = wordlstm(model,wdata,wmask,wembed)  # forw/back[T][B,H] where T: longest sentence, B: batch size, H: hidden size (300)
    nothing
end

# function loss(model, sentences, parsertype, beamsize)
function lmloss(model, sentences, vocab::Vocab; result=nothing)
    # map words and chars to Ints
    # words and sents contain Int (char/word id) arrays without sos/eos tokens
    # TODO: optionally construct cdict here as well
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
    prd(t) = (if ORDER==1; weight * vcat(forw[t],back[t]) .+ bias; else; hcat(forw[t],back[t]) * weight .+ bias; end)
    idx(t,b,n) = (if ORDER==1; data[t][b] + (b-1)*n; else; b + (data[t][b]-1)*n; end)
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
        o1 = logp(ypred,ORDER)
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
    cembed(t)=(if ORDER==1; embeddings[:,data[t]]; else; embeddings[data[t],:]; end)
    czero=(if ORDER==1; fill!(similar(bias,H,B), 0); else; fill!(similar(bias,B,H), 0); end)
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
    wembed(t)=(if ORDER==1; embeddings[:,data[t]]; else; embeddings[data[t],:]; end)
    wzero=(if ORDER==1; fill!(similar(bias,H,B), 0); else; fill!(similar(bias,B,H), 0); end)
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
    if ORDER==1
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
    global _vocab = Vocab(a["char_vocab"],Dict{String,Int}(),a["word_vocab"],"<s>","</s>","<unk>",'↥','Ϟ','⋮')
    nothing
end

function loadcorpus()
    global _corpus = Any[]
    s = Sentence([])
    for line in eachline("en-ud-train.conllu")
        if line == "\n"
            push!(_corpus, s)
            s = Sentence([])
        elseif (m = match(r"^\d+\t(.+?)\t", line)) != nothing
            push!(s.word, m.captures[1])
        end
    end
end

function loadmodel()
    global _model = Dict()
    m = load("english_ch12kmodel_2.jld")["model"]
    if ORDER==1
        for k in (:cembed,); _model[k] = KnetArray(m[k]'); end
        for k in (:forw,:soft,:back,:char); _model[k] = map(KnetArray, m[k]'); end
    else
        for k in (:cembed,); _model[k] = KnetArray(m[k]); end
        for k in (:forw,:soft,:back,:char); _model[k] = map(KnetArray, m[k]); end
    end
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
    loadmodel()
    loadvocab()
    loadcorpus()
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
