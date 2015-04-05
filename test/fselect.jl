using KUparser, KUnet, HDF5, JLD, ArgParse, Compat, CUDArt
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
require("fscore.jl")

function main()
    args = parse_commandline()
    (data, ndeps) = initdata(args)
    open(args["cache"],"a") do f end # this creates the cachefile if it does not exist
    pt = eval(parse(args["arctype"]))
    ftry = Array(Feature,nworkers())
    allfeats = eval(parse("Flist."*args["allfeats"]))

    # Init bestfeats, bestscore to the feats specified in args
    bestfeats = eval(parse("Flist."*args["feats"]))
    bestscore = getcache(args["cache"], bestfeats)
    if bestscore < 0
        bestscore = fscore(pt, data, ndeps, bestfeats, args)
        updatecache(args["cache"], bestfeats, bestscore)
    end

    # See if we can improve bestfeats with single feature flip steps
    updatedbest = true
    while updatedbest
        updatedbest = false

        # First check the cache to see if we already have a good step
        scores = map(allfeats) do f
            getcache(args["cache"], bestfeats, f)
        end
        (smax,imax) = findmax(scores)
        if smax > bestscore
            updatedbest = true
            bestscore = smax
            bestfeats = flip(bestfeats, allfeats[imax])
            continue
        end

        # If not check steps not in cache, nworkers at a time
        nf = 0
        while (nf < length(allfeats))
            nftry = 0
            while (nf < length(allfeats) && nftry < length(ftry))
                s = getcache(args["cache"], bestfeats, allfeats[nf+=1])
                s < 0 && (ftry[nftry+=1] = allfeats[nf])
            end
            nftry == 0 && break
            @everywhere gc()
            scores = pmap(ftry[1:nftry]) do f
                fscore(pt, data, ndeps, flip(bestfeats, f), args)
            end
            updatecache(args["cache"], bestfeats, ftry[1:nftry], scores)
            (smax,imax) = findmax(scores)
            if smax > bestscore
                updatedbest = true
                bestscore = smax
                bestfeats = flip(bestfeats, ftry[imax])
            end
        end # while (nf < length(allfeats))
    end # while updatedbest
    info("$bestscore\t$(join(sort(bestfeats), ' '))")
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "datafiles"
        help = "JLD files for input: first one will be used for training second for scoring"
        nargs = '+'
        required = true
        "--cache"
        help = "Cache file (plain text) for saving and reading results"
        default = ""
        "--nogpu"
        help = "Do not use gpu"
        action = :store_true
        # We use oparser on the training set to get training data.
        # We train the net for n epochs with fixed training data, no parsing.
        # Finally we evaluate by parsing the dev corpus using greedy or beam parser specified below.
        "--parser"
        help = "Parsing algorithm to use: g(reedy)parser, b(eam)parser"
        default = "gparser"
        "--arctype"
        help = "Move set to use: ArcEager{R1,13}, ArcHybrid{R1,13}"
        default = "ArcEager13"
        "--feats"
        help = "Features to start with from KUparser.Flist"
        default = "acl11eager"
        "--allfeats"
        help = "Complete feature set to use from KUparser.Flist"
        default = "acl11eager"
        "--hidden"
        help = "One or more hidden layer sizes"
        nargs = '+'
        arg_type = Int
        default = [20000]
        "--epochs"
        help = "Number of epochs to train"
        arg_type = Int
        default = 10
        # Don't need this, we only parse at the end:
        # "--pepochs"
        # help = "Number of epochs between parsing"
        # arg_type = Int
        # default = 1
        # We will just use whatever julia was started with
        # "--ncpu"
        # help = "Number of workers for multithreaded parsing"
        # arg_type = Int
        # default = 6
        "--pbatch"
        help = "Minibatch size for parsing"
        arg_type = Int
        default = 500
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
        # No dropout for 10-epoch training
        # default = [0.2f0, 0.7f0]
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

function getcache(cachefile::String, feats::Fvec, f::Feature="")
    isempty(f) || (feats = flip(feats, f))
    fkey = join(sort(feats), ' ')
    fval = -1.0
    open(cachefile) do fp
        for l in eachline(fp)
            (score, fstr) = split(chomp(l), '\t')
            fstr == fkey && (fval = float(score); break)
        end
    end
    return fval
end

function updatecache(cachefile::String, feats::Fvec, score::Real)
    open(cachefile, "a") do fp
        fkey = join(sort(feats), ' ')
        write(fp, "$score\t$fkey\n")
    end
end

function updatecache(cachefile::String, feats::Fvec, toflip::Fvec, scores::AbstractArray)
    @assert length(toflip) == length(scores)
    open(cachefile, "a") do fp
        for i=1:length(toflip)
            fkey = join(sort(flip(feats,toflip[i])), ' ')
            fval = scores[i]
            write(fp, "$fval\t$fkey\n")
        end
    end
end

function initdata(args::Dict)
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
    @assert length(data)==2 "fselect needs a training and a development set"
    (data, ndeps)
end


main()
