#TODO: save output

using HDF5, JLD, KUnet, ArgParse, Compat, CUDArt
using KUparser: ArcHybrid, Corpus, flen, wdim, oparse, bparse, gparse
if VERSION < v"0.4"
    using Dates
end

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

macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
macro meminfo() :(@everywhere gc(); run(`nvidia-smi`); run(`ps auxww`|>`grep julia`); run(`free`)) end
evalheads(p,c)=mean(vcat(vcat(map(q->q[1],p)...)...) .== vcat(map(s->s.head,c)...))

function initworkers(ncpu)
    (nworkers() < ncpu) && (addprocs(ncpu - nprocs() + 1))
    require("CUDArt")
    @everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
    require("CUBLAS")
    require("KUnet")
    require("KUparser")
end

function main()
    args = parse_commandline()
    KUnet.gpu(!args["nogpu"])
    args["nogpu"] && blas_set_num_threads(20)
    
    data = Corpus[]
    for f in args["datafiles"]
        @date d=load(f)
        d = collect(values(d))
        @assert length(d)==1 "$f has more than one variable"
        push!(data, d[1])
    end

    feats=eval(parse("KUparser.Flist."*args["feats"]))
    xrows=flen(wdim(data[1][1]), feats)
    yrows=ArcHybrid(1).nmove

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
    @date initworkers(args["ncpu"])
    for i=1:length(data)
        @date p = oparse(data[i], feats, args["ncpu"])
        @show evalheads(p, data[i])
        i == 1 && (trn = p)
        p = nothing
    end
    @date rmprocs(workers())
    accuracy = Array(Float32, length(data))
    
    for epoch=1:args["epochs"]
        @meminfo
        @show epoch
        @time for (h,x,y) in trn
            KUnet.train(net, x, y; batch=args["tbatch"], loss=KUnet.logploss)
        end
        (args["parser"] != "oparser") && (trn=nothing)
        @date initworkers(args["ncpu"])
        for i=1:length(data)
            if in(args["parser"], ["oparser", "gparser"])
                @date p = gparse(data[i], net, feats, args["pbatch"], args["ncpu"])
            elseif (args["parser"] == "bparser")
                @date p = bparse(data[i], net, feats, args["nbeam"], args["pbatch"], args["ncpu"])
            else
                error("Unknown parser "*args["parser"])
            end
            @show accuracy[i] = evalheads(p, data[i])
            (args["parser"] != "oparser") && (i == 1) && (trn=p)
            p = nothing
        end
        @date rmprocs(workers())
        println("DATA:\t$epoch\t"*join(accuracy, '\t')); flush(STDOUT)
    end
end

main()
