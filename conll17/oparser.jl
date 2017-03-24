using ArgParse,KUparser
include("conll17.jl")
macro tm(_x) :(if o[:fast]; $(esc(_x)); else; info("$(now()) "*$(string(_x))); $(esc(_x)); end) end
macro msg(x); :(if !o[:fast]; info($x); end); end

# Some default data
grcdev = "/mnt/ai/data/nlp/conll17/ud-treebanks-conll2017/UD_Ancient_Greek/grc-ud-dev.conllu"
grcudp = "/mnt/ai/data/nlp/conll17/UDPipe/udpipe-ud-2.0-conll17-170315/models/ancient_greek-ud-2.0-conll17-170315.udpipe"

function main(args="")
    isa(args, AbstractString) && (args=split(args))
    s = ArgParseSettings()
    s.description="Oracle parser takes a conllu file and parses with the best possible moves."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        # Parsing options
        ("--parse"; default=grcdev; help="file in conllu format to be parsed")
        ("--udpipe"; default=grcudp; help="UDpipe model file")
        ("--ptype"; default="ArcEagerR1"; help="Parser type")
        ("--fast"; action=:store_true; help="fewer messages")
    end

    # Process args
    o = parse_args(args, s; as_symbols=true)
    @msg(s.description)
    @msg(string("opts=",[(k,v) for (k,v) in o]...))
    ptype = eval(parse(o[:ptype]))

    # Parse
    if o[:parse] != nothing
        @tm global corpus = readconllu(o[:parse])
        @tm parses = oparse(ptype, corpus)
        @tm writeconllu(corpus, parses, o[:parse])
    end
end

if basename(PROGRAM_FILE)==basename(@__FILE__); main(ARGS); end
