using Knet,JLD

function main()
    loadmodel()
    loadvocab()
    loadcorpus()
    # global data1 = minibatch(corpus, 1)
    # global data100 = minibatch(corpus, 100)
    # perplexity(model,data100)
end

function loadmodel()
    global model = Dict()
    m = load("english_ch12kmodel.jld")["model"]
    for k in (:cembed,); model[k] = KnetArray(m[k]); end
    for k in (:forw,:soft,:back,:char); model[k] = map(KnetArray, m[k]); end
end

function loadvocab()
    a = load("english_vocabs.jld")
    global cvocab = a["char_vocab"]
    global wvocab = a["word_vocab"]
end

function loadcorpus()
    global intcorpus = Vector{Int32}[]
    global strcorpus = Vector{String}[]
    intsentence = Int32[]
    strsentence = String[]
    unk = wvocab["<unk>"]
    for line in eachline("en-ud-train.conllu")
        line = chomp(line)
        fields = split(line,'\t')
        if line == ""
            if !isempty(intsentence)
                push!(intcorpus, intsentence)
                intsentence = Int32[]
                push!(strcorpus, strsentence)
                strsentence = String[]
            end
        elseif ismatch(r"^\d+$", fields[1])
            push!(strsentence, fields[2])
            push!(intsentence, get(wvocab, fields[2], unk))
        elseif line[1] == '#'
            # skip
        elseif ismatch(r"^\d+[-.]\d+$", fields[1])
            # skip
        else
            error("Cannot parse [$line]")
        end
    end
end

function minibatch(corpus, batchsize)
    function batch(a)
        B = length(a)
        T = length(corpus[a[1]])
        s = [ zeros(Int32,B) for i=1:T ]
        for t=1:T, b=1:B
            s[t][b] = corpus[a[b]][t]
        end
        return s
    end
    data = Any[]
    d = Dict{Int,Vector{Int}}()
    for i in 1:length(corpus)
        s = corpus[i]
        l = length(s)
        a = get!(d, l, Int[])
        push!(a, i)
        if length(a) == batchsize
            push!(data, batch(a))
            empty!(a)
        end
    end
    for (k,a) in d
        if !isempty(a)
            push!(data, batch(a))
        end
    end
    return data
end

function blstm(model, sequence; result=nothing)
    # assume the sequence does not have <s> or </s> or padding and add it
    sos = sosinput(model, sequence)
    eos = eosinput(model, sequence)
    sequence = [ sos, sequence..., eos ]
    # @show sequence
    T = length(sequence)
    # sequence[t] is a Vector{Int32} minibatch of B token ids
    B = length(sequence[1])
    # forw[t] and back[t] are the hiddens used to predict sequence[t-1]
    forw = Array(Any,T)
    back = Array(Any,T)
    # zero hidden for first time step
    h0 = zerohidden(model, sequence)
    # run the forward lstm
    hidden = cell = forw[1] = h0
    weight,bias = model[:forw]
    embed = model[:fembed]
    for t=1:T-1
        hidden,cell = lstm(weight,bias,hidden,cell,embed[sequence[t],:])
        forw[t+1] = hidden
    end
    # run the backward lstm
    hidden = cell = back[T] = h0
    weight,bias = model[:back]
    embed = model[:bembed]
    for t=T:-1:2
        hidden,cell = lstm(weight,bias,hidden,cell,embed[sequence[t],:])
        back[t-1] = hidden
    end
    # run the predictions
    total = 0
    for t=1:T
        ypred = hcat(forw[t], back[t]) * model[:soft][1] .+ model[:soft][2]
        total += logprob(sequence[t], ypred)
        # @show tmp = logprob(sequence[t], ypred)
        # total += tmp
    end
    if result != nothing
        result[1] = total
        result[2] = B*T
    end
    # return -total 	    # total loss: longer sequences and larger minibatches have higher loss
    # return -total / (B*T) # per token loss: scale does not depend on sequence length or minibatch
    return -total / B       # per sequence loss: does not depend on minibatch, larger loss for longer seq
end

function blstm1(model, sequence; result=nothing)
    # assume the sequence does not have <s> or </s> or padding
    T = length(sequence)
    # sequence[t] is a Vector{Int32} minibatch of B token ids
    B = length(sequence[1])
    # forw[t] and back[t] are the hiddens used to predict sequence[t]
    forw = Array(Any,T)
    back = Array(Any,T)
    # zero hidden and sos token minibatch for first time step
    h0 = zerohidden(model, sequence)
    x0 = sosinput(model, sequence)
    # run the forward lstm
    hidden,cell,input = h0,h0,x0
    weight,bias = model[:forw]
    embed = model[:fembed]
    for t=1:T
        hidden,cell = lstm(weight,bias,hidden,cell,embed[input,:])
        forw[t] = hidden
        input = sequence[t]
    end
    # run the backward lstm
    hidden,cell,input = h0,h0,x0
    weight,bias = model[:back]
    embed = model[:bembed]
    for t=T:-1:1
        hidden,cell = lstm(weight,bias,hidden,cell,embed[input,:])
        back[t] = hidden
        input = sequence[t]
    end
    # run the predictions
    total = 0
    for t=1:T
        ypred = hcat(forw[t], back[t]) * model[:soft][1] .+ model[:soft][2]
        # total += logprob(sequence[t], ypred)
        @show tmp = logprob(sequence[t], ypred)
        total += tmp
    end
    if result != nothing
        result[1] = total
        result[2] = B*T
    end
    # return -total 	    # total loss: longer sequences and larger minibatches have higher loss
    # return -total / (B*T) # per token loss: scale does not depend on sequence length or minibatch
    return -total / B       # per sequence loss: does not depend on minibatch, larger loss for longer seq
end

function perplexity(model, data)
    result = zeros(2)
    total = zeros(2)
    for sequence in data
        blstm(model, sequence; result=result)
        total += result
    end
    @show total
    return exp(-total[1]/total[2])
end

# row-major lstm
function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

let h0=sos=eos=nothing; global zerohidden, sosinput, eosinput
    function zerohidden(model, sequence)
        B = length(sequence[1])
        W = model[:forw][1]
        H = div(size(W,2),4)
        if h0 == nothing || size(h0) != (B,H) || typeof(h0) != typeof(W)
            h0 = fill!(similar(W, B, H), 0)
        end
        return h0
    end
    function sosinput(model, sequence)
        if sos == nothing || length(sos) != length(sequence[1])
            sos = fill!(similar(sequence[1]), vocab["<s>"])
        end
        return sos
    end
    function eosinput(model, sequence)
        if eos == nothing || length(eos) != length(sequence[1])
            eos = fill!(similar(sequence[1]), vocab["</s>"])
        end
        return eos
    end
end

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end
