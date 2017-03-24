using ArgParse,JLD,KUparser,Knet
include("conll17.jl")
macro tm(_x) :(if o[:fast]; $(esc(_x)); else; info("$(now()) "*$(string(_x))); $(esc(_x)); end) end
macro msg(x); :(if !o[:fast]; info($x); end); end

# Some default data
grctrn = "/mnt/ai/data/nlp/conll17/ud-treebanks-conll2017/UD_Ancient_Greek/grc-ud-train.conllu"
grcdev = "/mnt/ai/data/nlp/conll17/ud-treebanks-conll2017/UD_Ancient_Greek/grc-ud-dev.conllu"
grctxt = "/mnt/ai/data/nlp/conll17/ud-treebanks-conll2017/UD_Ancient_Greek/grc-ud-dev.txt"
grcvec = "/mnt/ai/data/nlp/conll17/word-embeddings-conll17/Ancient_Greek/grc.vectors.xz"
grcudp = "/mnt/ai/data/nlp/conll17/UDPipe/udpipe-ud-2.0-conll17-170315/models/ancient_greek-ud-2.0-conll17-170315.udpipe"

function main(args="")
    isa(args, AbstractString) && (args=split(args))
    s = ArgParseSettings()
    s.description="Using greedy parser with oracle (static) trained MLP model"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        # Parsing options
        ("--parse"; default=grctxt; help="file in plain text format to be parsed")
        ("--udpipe"; default=grcudp; help="UDpipe model file")
        ("--load"; help="load model from jld file")
        ("--ptype"; default="ArcEagerR1"; help="Parser type")
        ("--flist"; default="Flist.zn11pv"; help="Feature list to use")

        # Training options
        ("--train"; nargs='*'; default=[grctrn,grcdev]; help="training file(s) in conllu format")
        ("--vectors";  default=grcvec; help="word vectors")
        ("--save"; help="save model to jld file")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--otype"; default="Adam()"; help="Optimization algorithm and parameters.")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--dropout"; arg_type=Float64; default=0.0; help="dropout probability")
        ("--l1"; arg_type=Float64; default=0.0; help="L1 regularization")
        ("--l2"; arg_type=Float64; default=0.0; help="L2 regularization")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
    end

    # Process args
    o = parse_args(args, s; as_symbols=true)
    @msg(s.description)
    @msg(string("opts=",[(k,v) for (k,v) in o]...))
    if o[:seed] > 0; setseed(o[:seed]); end
    atype = eval(parse(o[:atype]))
    ptype = eval(parse(o[:ptype]))
    flist = eval(parse(o[:flist]))

    # Load model
    model = optim = vocab = nothing
    if o[:load] != nothing
        a = load(o[:load])
        model = a["model"]
        optim = a["optim"]
        vocab = a["vocab"]
    end

    # Train model
    if !isempty(o[:train])
        if vocab == nothing; vocab = Vocab(); vocab.postags=UPOSTAG; vocab.deprels=UDEPREL; end
        readc(f)=readconllu(f,vocab)
        @tm corpora = map(readc, o[:train])
        if o[:vectors] != nothing
            @tm readvectors(o[:vectors], vocab)
        end
        data = map(corpora) do corpus
            @tm (p,x,y) = oparse(ptype, corpus, flist)
            @tm minibatch(x, y, o[:batchsize]; atype=atype)
        end
        if model == nothing
            xtrn,ytrn = data[1][1]
            model = initmodel(size(xtrn,1), o[:hidden]..., size(ytrn,1); atype=atype)
            optim = initoptim(model, o[:otype])
        end
        report(ep)=if !o[:fast]; info((ep,map(d->accuracy(model,d,mlp),data)...)); end
        report(0)
        for epoch=1:o[:epochs]
            for (x,y) in data[1]
                grads = softgrad(model,x,y,mlp;l1=o[:l1],l2=o[:l2],pdrop=o[:dropout])
                update!(model,grads,optim)
            end
            report(epoch)
        end
    end

    # Save model
    if o[:save] != nothing
        @tm save(o[:save], "model", model, "optim", optim, "vocab", vocab)
    end

    # Parse
    if o[:parse] != nothing
        # we need to test with predicted toks and tags from txt data.
        txtfile = tempname()
        @tm run(pipeline(`udpipe --tokenize --tag $(o[:udpipe]) $(o[:parse])`, txtfile))
        @tm corpus = readconllu(txtfile, vocab); rm(txtfile)
        pred(x,y)=copy!(y, mlp(model, atype(x)))
        @tm parses = gparse(ptype, corpus, flist, pred; nbatch=o[:batchsize])
        @tm writeconllu(corpus, parses)
    end
end

# function savedata()
#     @date c = readconllu(corpus, wordvecs)
#     @date (p,x,y) = oparse(pt, c, ft)
#     @date save("foo.en.jld", "x", x, "y", y, "c", c, "p", p)
# end

function minibatch(x, y, batch; atype=Array{Float32})
    nx = size(x,2)
    data = Any[]
    for i=1:batch:nx
        j=min(nx,i+batch-1)
        push!(data, (atype(view(x,:,i:j)),atype(view(y,:,i:j))))
    end
    return data
end

function mlp(w,x;pdrop=0)
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
        x = dropout(x, pdrop)
    end
    return w[end-1]*x .+ w[end]
end

# # For use with gparser until we change the interface:
# function predict(w,x,y)
#     x = KnetArray(x)
#     for i=1:2:length(w)-2
#         x = relu(w[i]*x .+ w[i+1])
#     end
#     # ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()) # uncomment for profiling
#     copy!(y, w[end-1]*x .+ w[end])
# end

using Knet: @cuda, Cptr

# target may be a subarray, TODO: handle in Knet
function Base.copy!{T}(dest::SubArray{T}, src::KnetArray{T})
    if length(dest) < length(src); throw(BoundsError()); end
    @cuda(cudart,cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
          pointer(dest), pointer(src), length(src)*sizeof(T), 2)
    return dest
end

# Calculate cross entropy loss of a model with weights w for one minibatch (x,p)
# Use non-zero l1 or l2 for regularization
function softloss(w,x,p,model;l1=0,l2=0,o...)
    y = model(w,x;o...)
    logphat = logp(y)
    J = -sum(p .* logphat) / size(x,2)  # dividing by number of instances for per-instance average
    if l1 != 0; J += l1 * sum(sumabs(wi)  for wi in w[1:2:end]); end
    if l2 != 0; J += l2 * sum(sumabs2(wi) for wi in w[1:2:end]); end
    return J
end

softgrad = grad(softloss)

function avgloss(w,data,model) # average loss for the whole dataset
    sum = cnt = 0
    for (x,y) in data
        sum += softloss(w,x,y,model)
        cnt += 1
    end
    return sum/cnt
end

function accuracy(w,data,model)
    corr = cnt = 0
    for (x,y) in data
        ypred = mlp(w,x)
        corr += sum(y .* (ypred .== maximum(ypred,1)))
        cnt += size(y,2)
    end
    return corr/cnt
end

function initmodel(d...; atype=Array{Float32})
    init(d...)=atype(xavier(Float32,d...))
    bias(d...)=atype(zeros(Float32,d...))
    w = Any[]
    for i=1:length(d)-1
        push!(w, init(d[i+1],d[i]))
        push!(w, bias(d[i+1],1))
    end
    return w
end

# This should work for any combination of tuple/array/dict
initoptim{T<:Number}(::KnetArray{T},otype)=eval(parse(otype))
initoptim{T<:Number}(::Array{T},otype)=eval(parse(otype))
initoptim(a::Associative,otype)=Dict(k=>initoptim(v,otype) for (k,v) in a) 
initoptim(a,otype)=map(x->initoptim(x,otype), a)

# function train(dtrn,dtst;
#                hidden = [ 128 ],
#                optimizer = Adam,
#                epochs = 10,
#                l1 = 0, l2 = 0, pdrop = 0
#                )
#     (x,y)=dtrn[1]
#     w = initmodel(size(x,1), hidden..., size(y,1))
#     p = oparams(w, optimizer)
#     report(ep)=println((ep,accuracy(w,dtrn,mlp),accuracy(w,dtst,mlp)))
#     report(0)
#     for epoch=1:epochs
#         for (x,y) in dtrn
#             g = softgrad(w,x,y,mlp;l1=l1,l2=l2,pdrop=pdrop)
#             update!(w,g,p)
#         end
#         report(epoch)
#     end
#     return w
# end

# julia> dtrn,dtst = loaddata()
# julia> train(dtrn,dtst)
# (0,0.008304813f0,0.01f0)
# (1,0.9219519f0,0.9071f0)
# (2,0.9373984f0,0.9188f0)
# (3,0.9448262f0,0.9239f0)
# (4,0.9502219f0,0.923f0)
# (5,0.95345455f0,0.9229f0)
# (6,0.95614976f0,0.923f0)
# (7,0.95832354f0,0.9228f0)
# (8,0.9598797f0,0.9223f0)
# (9,0.962f0,0.9258f0)
# (10,0.9629759f0,0.9233f0)

# julia> train(dtrn,dtst;hidden=1000)
# (0,0.0073877005f0,0.0074f0)
# (1,0.9364385f0,0.9173f0)
# (2,0.9535187f0,0.9284f0)
# (3,0.9634011f0,0.9316f0)
# (4,0.97001874f0,0.9315f0)
# (5,0.97462296f0,0.9295f0)
# (6,0.9792781f0,0.9307f0)
# (7,0.982254f0,0.9316f0)
# (8,0.98386633f0,0.9325f0)
# (9,0.98568714f0,0.9291f0)
# (10,0.98813367f0,0.9312f0)

# julia> train(dtrn,dtst;hidden=10000)
# (0,0.0077700536f0,0.0077f0)
# (1,0.9415294f0,0.9223f0)
# (2,0.95814705f0,0.9313f0)
# (3,0.9689037f0,0.9358f0)
# (4,0.97555614f0,0.932f0)
# (5,0.98095185f0,0.9337f0)
# (6,0.9839626f0,0.934f0)
# (7,0.98727006f0,0.9358f0)
# (8,0.9889813f0,0.9356f0)
# (9,0.9904198f0,0.9358f0)
# (10,0.99142516f0,0.9338f0)

if PROGRAM_FILE=="oparser-mlp.jl"; main(ARGS); end
