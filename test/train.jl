#TODO: save output

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
        default = "out.jld"
        "--nogpu"
        help = "Do not use gpu"
        action = :store_true
        "--parser"
        help = "Parsing algorithm to use: g(reedy)parser, b(eam)parser, or o(racle)parser for static training"
        default = "oparser"
        "--feats"
        help = "Features to use from KUparser.Flist"
        default = "fv021a"
        "--hidden"
        help = "One or more hidden layer sizes"
        nargs = '+'
        arg_type = Int
        default = [20000]
        "--epochs"
        help = "Number of epochs to train"
        arg_type = Int
        default = 256
        "--pepochs"
        help = "Number of epochs between parsing"
        arg_type = Int
        default = 1
        "--ncpu"
        help = "Number of workers for multithreaded parsing"
        arg_type = Int
        default = 12
        "--pbatch"
        help = "Minibatch size for parsing"
        arg_type = Int
        default = 2000
        "--tbatch"
        help = "Minibatch size for training"
        arg_type = Int
        default = 128
        "--nbeam"
        help = "Beam size for bparser"
        arg_type = Int
        default = 10
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
    end
    args = parse_args(s)
    for (arg,val) in args
        print("$arg:$val ")
    end
    println("")
    return args
end

macro meminfo() :(gc(); run(`nvidia-smi`); run(`ps auxww`|>`grep julia`); run(`free`)) end

function main()
    args = parse_commandline()
    KUnet.gpu(!args["nogpu"])
    args["nogpu"] && blas_set_num_threads(20)
    
    data = Corpus[]
    deprel = nothing
    ndeps = 0
    for f in args["datafiles"]
        @date d=load(f)
        push!(data, d["corpus"])
        if deprel == nothing
            deprel = d["deprel"]
            ndeps = length(deprel)
        else
            @assert deprel == d["deprel"]
        end
    end

    feats=eval(parse("KUparser.Flist."*args["feats"]))
    xrows=flen(wdim(data[1][1]), feats)
    yrows=KUparser.ArcHybrid(1,length(deprel)).nmove # TODO: change when multiple parsers implemented

    net=KUnet.newnet(KUnet.relu, [xrows; args["hidden"]; yrows]...)
    net[end].f=KUnet.logp
    for k in [fieldnames(KUnet.UpdateParam); :dropout]
        haskey(args, string(k)) || continue
        v = args[string(k)]
        if isempty(v)
            continue
        elseif length(v)==1
            KUnet.setparam!(net, k, v[1])
        else 
            @assert length(v)==length(net) "$k should have 1 or $(length(net)) elements"
            for i=1:length(v)
                KUnet.setparam!(net[i], k, v[i])
            end
        end
    end
    @show net

    trn = nothing
    for i=1:length(data)
        @date pxy = oparse(data[i], feats, ndeps, args["ncpu"])
        @show evalparse(pxy[1], data[i])
        i == 1 && (trn = pxy)
        pxy = nothing
    end
    accuracy = Array(Float32, length(data))
    
    for epoch=1:args["epochs"]
        @show epoch
        (p,x,y) = trn
        @date KUnet.train(net, x, y; batch=args["tbatch"], loss=KUnet.logploss)
        if epoch % args["pepochs"] == 0
            (args["parser"] != "oparser") && (trn=nothing)
            @meminfo
            for i=1:length(data)
                if in(args["parser"], ["oparser", "gparser"])
                    @date pxy = gparse(data[i], net, feats, ndeps, args["pbatch"], args["ncpu"])
                elseif (args["parser"] == "bparser")
                    @date pxy = bparse(data[i], net, feats, ndeps, args["nbeam"], args["pbatch"], args["ncpu"])
                else
                    error("Unknown parser "*args["parser"])
                end
                @show e = evalparse(pxy[1], data[i])
                accuracy[i] = e[1]
                (args["parser"] != "oparser") && (i == 1) && (trn=pxy)
                pxy = nothing
            end
            println("DATA:\t$epoch\t"*join(accuracy, '\t')); flush(STDOUT)
        end
    end
end

main()
