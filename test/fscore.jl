using KUparser, KUnet, HDF5, JLD, ArgParse, Compat, CUDArt

function fscore{T<:Parser}(pt::Type{T}, data::Vector{Corpus}, ndeps::Integer, feats::Fvec, args::Dict)
    # Assuming this is running on a worker, do not use ncpu when parsing.
    # Initialize training set using oparse on first corpus
    @date (p,x,y) = oparse(pt, data[1], ndeps, feats); p=nothing
    net = initnet(pt, data, ndeps, feats, args)
    @date for i=1:args["epochs"]
        train(net, x, y; batch=args["tbatch"], loss=KUnet.logploss, shuffle=true)
    end
    x = y = nothing
    if (args["parser"] == "gparser")
        @date p = gparse(pt, data[2], ndeps, feats, net, args["pbatch"])
    elseif (args["parser"] == "bparser")
        @date p = bparse(pt, data[2], ndeps, feats, net, args["nbeam"], args["pbatch"])
    else
        error("Unknown parser: "*args["parser"])
    end
    @show e = evalparse(p, data[2])
    net=p=nothing; gc()
    return e[1]  # e[1] is UAS including punct
end

function initnet{T<:Parser}(pt::Type{T}, data::Vector{Corpus}, ndeps::Integer, feats::Fvec, args::Dict)
    s1 = data[1][1]
    p1 = pt(wcnt(s1),ndeps)
    xrows=KUparser.flen(p1, s1, feats)
    yrows=p1.nmove
    # Initialize net:
    args["nogpu"] && KUnet.gpu(false)
    srand(42)
    net=newnet(relu, [xrows; args["hidden"]; yrows]...)
    net[end].f=KUnet.logp
    for k in [fieldnames(UpdateParam); :dropout]
        haskey(args, string(k)) || continue
        v = args[string(k)]
        if isempty(v)
            continue
        elseif length(v)==1
            setparam!(net, k, v[1])
        else 
            @assert length(v)==length(net) "$k should have 1 or $(length(net)) elements"
            for i=1:length(v)
                setparam!(net[i], k, v[i])
            end
        end
    end
    return net
end

function flip(a, x)
    b = copy(a)
    i = findfirst(b, x)
    i == 0 ? push!(b, x) : deleteat!(b, i)
    return b
end

