using KUparser, KUnet, HDF5, JLD, ArgParse, Compat, CUDArt
VERSION < v"0.4-" && eval(Expr(:using,:Dates))

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "datafiles"
        help = "JLD files for input: first one will be used for training"
        nargs = '+'
        required = true
        "--out"
        help = "JLD file for output trained network"
        default = ""
        "--net"
        help = "Initialize net from this file if specified"
        default = ""
        "--machinefile"
        help = "Need this to restart workers to avoid pmap leak"
        default = ""
        # "--nogpu"  # always use gpu
        # help = "Do not use gpu"
        # action = :store_true
        # "--parser"  # always use bparser
        # help = "Parsing algorithm to use: g(reedy)parser, b(eam)parser, or o(racle)parser for static training"
        # default = "oparser"
        "--arctype"
        help = "Move set to use: ArcEager{R1,13}, ArcHybrid{R1,13}"
        default = "ArcEager13"
        "--feats"
        help = "Features to use from KUparser.Flist"
        default = "tacl13hybrid"
        "--hidden"
        help = "One or more hidden layer sizes"
        nargs = '+'
        arg_type = Int
        default = [20000]
        "--epochs"
        help = "Minimum number of epochs to train"
        arg_type = Int
        default = 100
        # "--pepochs"
        # help = "Number of epochs between parsing"
        # arg_type = Int
        # default = 1
        # "--ncpu"
        # help = "Number of workers for multithreaded parsing"
        # arg_type = Int
        # default = 12
        "--pbatch"
        help = "Minibatch size for parsing"
        arg_type = Int
        default = 100
        "--tbatch"
        help = "Minibatch size for training"
        arg_type = Int
        default = 128
        "--nbeam"
        help = "Beam size for bparser"
        arg_type = Int
        default = 64
        "--adagrad"
        help = "If nonzero apply adagrad using arg as epsilon"
        arg_type = Float32
        nargs = '+'
        default = [1f-8]
        "--dropout"
        help = "Dropout probability"
        arg_type = Float32
        nargs = '+'
        default = [0.2f0, 0.7f0]
        "--l1reg"
        help = "L1 regularization parameter"
        arg_type = Float32
        nargs = '+'
        "--l2reg"
        help = "L2 regularization parameter"
        arg_type = Float32
        nargs = '+'
        "--learningRate"
        help = "Learning rate"
        arg_type = Float32
        nargs = '+'
        default = [2f-2]
        "--maxnorm"
        help = "If nonzero upper limit on weight matrix row norms"
        arg_type = Float32
        nargs = '+'
        "--momentum"
        help = "Momentum"
        arg_type = Float32
        nargs = '+'
        "--nesterov"
        help = "Apply nesterov's accelerated gradient"
        arg_type = Float32
        nargs = '+'
        "--seed"
        help = "Random number seed (-1 leaves the default seed)"
        arg_type = Int
        default = -1
    end
    args = parse_args(s)
    for (arg,val) in args
        print("$arg:$val ")
    end
    println("")
    return args
end

function pbatch(c::Corpus, n::Int)
    dist = Corpus[]
    for s1=1:n:length(c)
        s2=min(length(c), s1+n-1)
        push!(dist, c[s1:s2])
    end
    return dist
end

function evalpxy(pxy, data)
    p = map(a->a[1], pxy)
    evalp(p, data)
end

function evalp(p, d)
    pa = vcat(p...)
    da = vcat(d...)
    evalparse(pa, da)
end

function initdata(args)
    pt = eval(parse(args["arctype"]))
    data = Any[]
    ndeps = 0
    feats = eval(parse("Flist."*args["feats"]))
    net = nothing
    deprel = nothing
    for f in args["datafiles"]
        @date d=load(f)
        push!(data, pbatch(d["corpus"], args["pbatch"]))
        if deprel == nothing
            deprel = d["deprel"]
            ndeps = length(deprel)
        else
            @assert deprel == d["deprel"]
        end
    end
    if isempty(args["net"])
        s1 = data[1][1][1]
        p1 = pt(wcnt(s1),length(deprel))
        xrows=KUparser.flen(p1, s1, feats)
        yrows=p1.nmove
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
        # Initialize net using oparse on first corpus
        @date (p,x,y) = oparse(pt, data[1], ndeps, feats)
        @date train(net, x, y; batch=args["tbatch"], loss=KUnet.logploss, shuffle=true)
    else 
        net = newnet(args["net"])
    end
    return (pt, data, ndeps, feats, net)
end

function restart_all_workers(wlist)
    @date Base.terminate_all_workers()
    @date gc()
    @show Base.PGRP.refs
    @date addprocs(wlist)
    @date @everywhere eval(Expr(:using,:KUnet))
    @date @everywhere eval(Expr(:using,:KUparser))
end

function myparse{T<:Parser}(pt::Type{T}, ca::AbstractArray, ndeps::Integer, feats::Fvec, net::Net, args::Dict; trn::Bool=false)
    @date tnet = testnet(net)
    if trn
        @date pxy = pmap(ca) do c
            bparse(pt, c, ndeps, feats, copy(tnet,:gpu), args["nbeam"], args["pbatch"]; xy=true)
        end
        @show e = evalpxy(pxy, ca); flush(STDOUT)
        @date for (p,x,y) in pxy
            train(net, x, y; batch=args["tbatch"], loss=KUnet.logploss, shuffle=true)
        end
    else
        @date p = pmap(ca) do c
            bparse(pt, c, ndeps, feats, copy(tnet,:gpu), args["nbeam"], args["pbatch"])
        end
        @show e = evalp(p, ca); flush(STDOUT)
    end
    return e[1]
end

function rss()
    statm = split(readall("/proc/$(getpid())/statm"))
    pages = int(statm[2])
    mb = pages >> 8
end

function main()
    args = parse_commandline()
    args["seed"] >= 0 && (srand(args["seed"]); KUnet.gpuseed(args["seed"]))
    @date (pt, data, ndeps, feats, net) = initdata(args)
    @show (pt, ndeps, feats)
    @show map(size, data)
    @show net
    accuracy = Array(Float32, length(data))
    (bestscore,bestepoch,epoch)=(0,0,0)
    @show wlist = (isempty(args["machinefile"]) ? [] : split(readall(args["machinefile"])))

    while true
        @show epoch += 1; flush(STDOUT)
        for i=1:length(data)
            @show rss(); flush(STDOUT)
            @date isempty(wlist) || restart_all_workers(wlist)
            @show rss(); flush(STDOUT)
            @date gc()
            @show rss(); flush(STDOUT)
            @date accuracy[i] = myparse(pt, data[i], ndeps, feats, net, args; trn=(i==1))
            @show rss(); flush(STDOUT)
            @date gc()
            @show rss(); flush(STDOUT)
        end
        println("DATA:\t$epoch\t"*join(accuracy, '\t')); flush(STDOUT)

        # We look at accuracy[2] (dev score) by default,
        # accuracy[1] (trn score) if there is no dev.

        score = (length(accuracy) > 1 ? accuracy[2] : accuracy[1])

        # Update best score and save best net

        if (score > bestscore)
            @show (bestscore,bestepoch)=(score,epoch); flush(STDOUT);
            !isempty(args["out"]) && save(args["out"], net)
        end

        # Quit if we have the minimum number of epochs and have
        # not made progress in the last half.
        
        (epoch >= args["epochs"]) && (epoch >= 2*bestepoch) && break

    end # while true
end # main

main()
