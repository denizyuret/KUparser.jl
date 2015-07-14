using CUDArt
@everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
using KUparser, KUnet, HDF5, JLD, ArgParse, Compat

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "datafiles"
        help = "JLD files for input: first one will be used for training"
        nargs = '+'
        required = true
        "--out"
        help = "JLD file for saving last network"
        #default = ""
        "--in"
        help = "JLD file for loading init network"
        #default = ""
        "--best"
        help = "JLD file for best (dev) network"
        #default = ""
        "--nogpu"
        help = "Do not use gpu"
        action = :store_true
        "--ispunct"
        help = "Do not score punct as determined by this function"
        #default = "zn11punct"
        "--parser"
        help = "Parsing algorithm to use: g(reedy)parser, b(eam)parser, or o(racle)parser for static training"
        default = "bparser"
        "--arctype"
        help = "Move set to use: ArcEager{R1,13}, ArcHybrid{R1,13}"
        default = "ArcEager13"
        "--feats"
        help = "Features to use from KUparser.Flist"
        default = "zn11cpv"
        "--hidden"
        help = "One or more hidden layer sizes"
        nargs = '+'
        arg_type = Int
        #default = [16384]
        "--minepochs"
        help = "Minimum number of epochs to train"
        arg_type = Int
        #default = 100
        "--maxepochs"
        help = "Maximum number of epochs to train"
        arg_type = Int
        #default = 100
        "--epochs"
        help = "Exact number of epochs to train"
        arg_type = Int
        #default = 100
        "--ncpu"
        help = "Number of workers for multithreaded parsing"
        arg_type = Int
        default = 16
        "--sbatch"
        help = "Batch size for parsing"
        arg_type = Int
        default = 2500
        "--pbatch"
        help = "Minibatch size for parsing"
        arg_type = Int
        default = 25
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
        # default = [0.1f0, 0.5f0]
        "--l1reg"
        help = "L1 regularization parameter"
        arg_type = Float32
        nargs = '+'
        "--l2reg"
        help = "L2 regularization parameter"
        arg_type = Float32
        nargs = '+'
        "--lr"
        help = "Learning rate"
        arg_type = Float32
        nargs = '+'
        default = [0.01f0]
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
        "--shuffle"
        help = "Whether to shuffle the data every epoch"
        action = :store_true
    end
    args = parse_args(s)
    for (arg,val) in args
        print("$arg:$val ")
    end
    println("")
    return args
end

function main()
    args = parse_commandline()
    KUnet.gpu(!args["nogpu"])
    args["nogpu"] && blas_set_num_threads(20)
    args["seed"] >= 0 && (srand(args["seed"]); KUnet.gpuseed(args["seed"]))
    (data, ndeps) = loaddata(args["datafiles"])
    @show (map(size, data), ndeps)
    @show feats = eval(parse("Flist."*args["feats"]))
    @show pt = eval(parse(args["arctype"]))
    net = (args["in"]==nothing ? initnet(args, pt, ndeps) : 
           !isfile(args["in"]) ? (warn("Cannot find net file, creating blank net"); args["in"]=nothing; initnet(args, pt, ndeps)) :
           (@date loadnet(args["in"])))
    display(net)
    ispunct = (args["ispunct"]==nothing ? nothing : eval(parse("KUparser."*args["ispunct"])))
    @show ispunct
    accuracy = Array(Float32, length(data))
    (bestscore,bestepoch,epoch)=(0,0,0)
    
    while true
        @date println("\nepoch => $(epoch += 1)")

        corpus=data[1]          # use data[1] as the training corpus
        parses=Any[]
        for c1=1:args["sbatch"]:length(corpus)
            c2=min(length(corpus), c1+args["sbatch"]-1)
            @show (:corpus, 1, c1, c2)
            sentences=sub(corpus, c1:c2)
            if (args["parser"] == "oparser")     # || ((epoch==1) && (args["in"]==nothing)))
                @date p = oparse(pt, sentences, ndeps, args["ncpu"], feats)
            elseif (args["parser"] == "bparser")
                @date p = bparse(pt, sentences, ndeps, feats, net, args["nbeam"], args["pbatch"], args["ncpu"]; xy=true)
            elseif (args["parser"] == "gparser")
                @date p = gparse(pt, sentences, ndeps, feats, net, args["pbatch"], args["ncpu"]; xy=true)
            else
                error("Unknown parser")
            end
            @date for pxy in p
                append!(parses, pxy[1])
                train(net, pxy[2], pxy[3]; batch=args["tbatch"], shuffle=args["shuffle"])
            end
        end
        accuracy[1] = getscore(parses, corpus, ispunct)
        (args["out"] != nothing) && (@date savenet(args["out"], net))

        for idata=2:length(data)
            corpus=data[idata]
            parses=Any[]
            for c1=1:args["sbatch"]:length(corpus)
                c2=min(length(corpus), c1+args["sbatch"]-1)
                @show (:corpus, idata, c1, c2)
                sentences=sub(corpus, c1:c2)
                if in(args["parser"], ("oparser", "gparser"))
                    @date p = gparse(pt, sentences, ndeps, feats, net, args["pbatch"], args["ncpu"]; xy=false)
                elseif (args["parser"] == "bparser")
                    @date p = bparse(pt, sentences, ndeps, feats, net, args["nbeam"], args["pbatch"], args["ncpu"]; xy=false)
                else
                    error("Unknown parser")
                end
                for pp in p; append!(parses, pp); end
            end
            accuracy[idata] = getscore(parses, corpus, ispunct)
        end            

        println("DATA:\t$epoch\t"*join(accuracy, '\t')); flush(STDOUT)

        # We look at accuracy[2] (dev score) by default,
        # accuracy[1] (trn score) if there is no dev.

        myscore = (length(accuracy) > 1 ? accuracy[2] : accuracy[1])

        # Update best score and save best net

        if (myscore > bestscore)
            @show (bestscore,bestepoch)=(myscore,epoch)
            (args["best"] != nothing) && (@date savenet(args["best"], net))
        end

        # If epochs specified, do exactly epoch.
        # Otherwise quit if we have not made progress in the last half.
        # If minepochs specified, do not quit before minepoch.
        # If maxepochs specified, quit before maxepoch.

        if (args["epochs"] != nothing)
            (epoch >= args["epochs"]) && break
        elseif (epoch >= 2*bestepoch)
            (args["minepochs"]==nothing || epoch >= args["minepochs"]) && break
        elseif (args["maxepochs"]!=nothing)
            (epoch >= args["maxepochs"]) && break
        end

    end # while true
end # main

function getscore(parses, corpus, ispunct)
    if ispunct == nothing
        @show e = evalparse(parses, corpus)
        return e[1]
    else
        @show e = evalparse(parses, corpus; ispunct=ispunct)
        return e[2]
    end
end

function loaddata(files)
    data = Corpus[]
    deprel = postag = nothing
    ndeps = 0
    for f in files
        info("Loading $f")
        @time d=load(f)
        push!(data, d["corpus"])
        if deprel == nothing
            deprel = d["deprel"]
            postag = d["postag"]
            ndeps = length(deprel)
        else
            @assert deprel == d["deprel"]
            @assert postag == d["postag"]
        end
    end
    return (data, ndeps)
end

function initnet(args, pt, ndeps)
    net = Layer[]
    !isempty(args["dropout"]) && push!(net, Drop(args["dropout"][1]))
    for h in args["hidden"]
        append!(net, [Mmul(h), Bias(), Relu()])
        !isempty(args["dropout"]) && push!(net, Drop(args["dropout"][end]))
    end
    yrows = pt(1,ndeps).nmove
    # We are removing normalization
    #append!(net, [Mmul(yrows), Bias(), Logp(), LogpLoss()])
    append!(net, [Mmul(yrows), Bias()])
    for k in [fieldnames(Param)]
        haskey(args, string(k)) || continue
        v = args[string(k)]
        if isempty(v)
            continue
        elseif length(v)==1
            @eval setparam!($net; $k=$v[1])
        else 
            @assert length(v)==length(net) "$k should have 1 or $(length(net)) elements"
            for i=1:length(v)
                @eval setparam!($net[$i]; $k=$v[$i])
            end
        end
    end
    return net
end

main()

    # Initialize training set using oparse on first corpus
    # @date (p,x,y) = oparse(pt, data[1], ndeps, args["ncpu"], feats)
    # @date (p,x,y) = oparse(pt, data[1], ndeps, feats)
    # @show evalparse(p, data[1]); p=nothing
    # for i=2:length(data)
    #     #@date p = oparse(pt, data[i], ndeps, args["ncpu"])
    #     @date p = oparse(pt, data[i], ndeps)
    #     @show evalparse(p, data[i]); p=nothing
    # end

    # else # if DBG
    #     @date net = loadnet("foo11nnet3.jld")
    #     # @date d = load("zn11cpv.jld")
    #     # (x,y) = (d["xtrn"],d["ytrn"])
    #     @date x = Array(Float32, 3105,1820392)
    #     @date y = Array(Float32, 24,1820392)
    # end  # if DBG
        # @meminfo
        # if !DBG
        #     if isa(eltype(x), Array)
        #         @assert length(x)==length(y)
        #         for i=1:length(x)
        #             @show map(size, (x[i], y[i]))
        #             @date train(net, x[i], y[i]; batch=args["tbatch"], shuffle=true)
        #         end
        #     else
        #         @show map(size, (x, y))
        #         @date train(net, x, y; batch=args["tbatch"], shuffle=true)
        #     end
        # end # if !DBG

        # @meminfo
        # if (args["parser"] == "bparser")
        #     # The first corpus gives us the new training set
        #     p = x = y = nothing; gc()
        #     @date (@everywhere gc())
        #     @date (@everywhere require("KUparser"))
        #     @date (for w in workers(); @fetchfrom w require("errlog.jl"); end)
        #     @meminfo
        #     @date pxy = bparse(pt, data[1], ndeps, feats, net, args["nbeam"], args["pbatch"], args["ncpu"]; xy=true)
        #     @meminfo
        #     p = map(z->z[1], pxy); @show map(size, p)
        #     x = map(z->z[2], pxy); @show map(size, x)
        #     y = map(z->z[3], pxy); @show map(size, y)
        #     @meminfo
        #     #@show map(getbytes, (p,x,y))
        #     @show e = evalparse(mycat(p), data[1]); p=nothing
        #     accuracy[1] = e[2]
        #     for i=2:length(data)
        #         @date p = bparse(pt, data[i], ndeps, feats, net, args["nbeam"], args["pbatch"], args["ncpu"])
        #         @show e = evalparse(mycat(p), data[i]); p=nothing
        #         accuracy[i] = e[2]
        #     end
        #     #@date Main.restartmachines()
        #     #@date Main.restartcuda()
        # elseif (args["parser"] == "oparser")
        #     # We never change the training set with oparser, just report accuracy
        #     for i=1:length(data)
        #         @date p = gparse(pt, data[i], ndeps, feats, net, args["pbatch"], args["ncpu"])
        #         @show e = evalparse(p, data[i]); p=nothing
        #         accuracy[i] = e[2]  # e[2] is UAS excluding punct
        #     end
        # elseif (args["parser"] == "gparser")
        #     # The first corpus gives us the new training set
        #     p = x = y = nothing; 
        #     @date (@everywhere gc())
        #     @date (@everywhere require("KUparser"))
        #     @date (@everywhere require("errlog.jl"))
        #     @date (p,x,y) = gparse(pt, data[1], ndeps, feats, net, args["pbatch"], args["ncpu"]; xy=true)
        #     @show e = evalparse(p, data[1]); p=nothing
        #     accuracy[1] = e[2]
        #     for i=2:length(data)
        #         @date p = gparse(pt, data[i], ndeps, feats, net, args["pbatch"], args["ncpu"])
        #         @show e = evalparse(p, data[i]); p=nothing
        #         accuracy[i] = e[2]
        #     end
        #     @date Main.restartmachines()
        #     @date Main.restartcuda()
        # else
        #     error("Unknown parser "*args["parser"])
        # end
#macro meminfo() :(gc(); run(`nvidia-smi`); run(`ps auxww`|>`grep julia`); run(`free`)) end
#macro meminfo() :(run(`ps auxww`|>`grep julia`); run(`free`)) end
#macro meminfo() :(nothing) end

# function mycat(a::Array)
#     # concat elements of a along their last dimension
#     mx = size(a[1])[1:end-1]
#     nx = 0
#     for x in a
#         @assert size(x)[1:end-1] == mx
#         nx += size(x)[end]
#     end
#     b = similar(a[1], tuple(mx..., nx))
#     nb = 0
#     for x in a
#         copy!(b, nb+1, x, 1, length(x))
#         nb += length(x)
#     end
#     return b
# end

        # # gc() does not work, so we need these:
        # @date sleep(5)
        # @date Main.restartmachines()
        # @date sleep(5)
        # @date Main.restartcuda()
        # @date (@everywhere require("KUparser"))
        # @date (@everywhere gc())
