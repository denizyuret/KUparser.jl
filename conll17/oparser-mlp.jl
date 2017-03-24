using KUparser,JLD,Knet

wordvecs = "/mnt/ai/data/nlp/conll17/word-embeddings-conll17/English/en.vectors.xz"
corpus = "/mnt/ai/data/nlp/conll17/ud-treebanks-conll2017/UD_English/en-ud-train.conllu"
pt = ArcEager13
ft = Flist.zn11pv

function savedata()
    @date c = readconllu(corpus, wordvecs)
    @date (p,x,y) = oparse(pt, c, ft)
    @date save("foo.en.jld", "x", x, "y", y, "c", c, "p", p)
end

function loaddata(;batch=100,test=10000)
    a = load("foo.en.jld")
    x = a["x"]; y = a["y"]
    dtst = Any[]
    for i=1:batch:test
        j=i+batch-1
        if j > test; break; end
        push!(dtst, (KnetArray(view(x,:,i:j)),KnetArray(view(y,:,i:j))))
    end
    dtrn = Any[]
    for i=test+1:batch:size(x,2)
        j=i+batch-1
        if j > size(x,2); break; end
        push!(dtrn, (KnetArray(view(x,:,i:j)),KnetArray(view(y,:,i:j))))
    end
    return (dtrn,dtst)
end

function mlp(w,x;pdrop=0)
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
        x = dropout(x, pdrop)
    end
    return w[end-1]*x .+ w[end]
end

# For use with gparser until we change the interface:
function predict(w,x,y)
    x = KnetArray(x)
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
    end
    ccall(("cudaDeviceSynchronize","libcudart"),UInt32,())
    copy!(y, w[end-1]*x .+ w[end])
end

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

function initmodel(d...)
    init(d...)=KnetArray(xavier(Float32,d...))
    bias(d...)=fill!(KnetArray(Float32,d...),0)
    w = Any[]
    for i=1:length(d)-1
        push!(w, init(d[i+1],d[i]))
        push!(w, bias(d[i+1],1))
    end
    return w
end

# This should work for any combination of tuple/array/dict
oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)

function train(dtrn,dtst;
               hidden = [ 128 ],
               optimizer = Adam,
               epochs = 10,
               l1 = 0, l2 = 0, pdrop = 0
               )
    (x,y)=dtrn[1]
    w = initmodel(size(x,1), hidden..., size(y,1))
    p = oparams(w, optimizer)
    report(ep)=println((ep,accuracy(w,dtrn,mlp),accuracy(w,dtst,mlp)))
    report(0)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = softgrad(w,x,y,mlp;l1=l1,l2=l2,pdrop=pdrop)
            update!(w,g,p)
        end
        report(epoch)
    end
    return w
end

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
