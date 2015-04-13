using KUparser, KUnet, HDF5, JLD, ArgParse, Compat, CUDArt
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
require("fscore.jl")

type Fselect allfeats; bestfeats; bestscore; idxqueue; scores; Fselect()=new(); end

function main()
    args = parse_commandline()
    (data, ndeps) = initdata(args)
    pt = eval(parse(args["arctype"]))

    # We want a single copy of the state variables
    fs = Fselect()
    fs.allfeats = eval(parse("Flist."*args["allfeats"]))
    fs.idxqueue = randperm(length(fs.allfeats))
    fs.bestfeats = eval(parse("Flist."*args["feats"]))
    fs.bestscore = getcache(args["cache"], fs.bestfeats)
    if fs.bestscore < 0
        fs.bestscore = @fetchfrom workers()[1] fscore(pt, data, ndeps, fs.bestfeats, args)
        updatecache(args["cache"], fs.bestfeats, fs.bestscore)
    end

    function getnextidx()
        # See if we can improve fs.bestfeats with single feature flip steps from cache
        while true
            fs.scores = map(fs.allfeats) do f
                getcache(args["cache"], fs.bestfeats, f)
            end
            (smax,imax) = findmax(fs.scores)
            smax <= fs.bestscore && break
            fs.bestscore = smax
            fs.bestfeats = flip!(fs.bestfeats, fs.allfeats[imax])
            fs.idxqueue = randperm(length(fs.allfeats))
        end

        @show fs; flush(STDOUT)

        # Find the next neighbor to compute
        while (!isempty(fs.idxqueue) && (fs.scores[fs.idxqueue[1]] >= 0))
            shift!(fs.idxqueue)
        end

        return (minimum(fs.scores) >= 0 ? nothing : # terminate if all scores known
                isempty(fs.idxqueue) ? 0 :          # wait for remaining computations if queue empty
                shift!(fs.idxqueue))                # compute the next element in queue
    end

    # Feeder tasks based on multi.jl:pmap implementation:
    while !(isempty(fs.idxqueue) && (getnextidx()==nothing))
        @show fs; flush(STDOUT)
        @sync for wpid in workers()
            @async begin
                idx = getnextidx()
                # This got messed up because gc() is a leaky bucket
                # while idx != nothing
                if idx == nothing
                    # do nothing, time to terminate
                elseif idx == 0
                    sleep(10)
                else
                    @show ("$wpid gets fs.allfeats[$idx]=$(fs.allfeats[idx])"); flush(STDOUT)
                    @show feats = flip(fs.bestfeats, fs.allfeats[idx])
                    try 
                        score = remotecall_fetch(wpid, fscore, pt, data, ndeps, feats, args)
                        @show ("$wpid gets score[$(fs.allfeats[idx])]=$score"); flush(STDOUT)
                        if isa(score, Number)
                            updatecache(args["cache"], feats, score)
                        else
                            push!(fs.idxqueue, idx)
                            @show ("Got $score from $wpid"); flush(STDOUT); sleep(10)
                        end
                    catch ex
                        push!(fs.idxqueue, idx)
                        @show ("Caught $ex from $wpid"); flush(STDOUT); sleep(10)
                    end
                end  # if idx == 0
                # idx = getnextidx()
                # end
            end  # @async begin
        end  # @sync for wpid in workers()
        @show fs; flush(STDOUT)
        @date Main.restartmachines(); flush(STDOUT)
        require("KUparser")
        require("fscore.jl")
    end  # while
    info("$(fs.bestscore)\t$(join(sort(fs.bestfeats), ' '))")
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
    if isfile(cachefile)
        open(cachefile) do fp
            for l in eachline(fp)
                (score, fstr) = split(chomp(l), '\t')
                fstr == fkey && (fval = float(score); break)
            end
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
