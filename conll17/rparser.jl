using ArgParse,KUparser
include("conll17.jl")
macro tm(_x) :(if o[:fast]; $(esc(_x)); else; info("$(now()) "*$(string(_x))); $(esc(_x)); end) end
macro msg(x); :(if !o[:fast]; info($x); end); end

# Some default data
grctxt = "/mnt/ai/data/nlp/conll17/ud-treebanks-conll2017/UD_Ancient_Greek/grc-ud-dev.txt"
grcudp = "/mnt/ai/data/nlp/conll17/UDPipe/udpipe-ud-2.0-conll17-170315/models/ancient_greek-ud-2.0-conll17-170315.udpipe"

function main(args="")
    isa(args, AbstractString) && (args=split(args))
    s = ArgParseSettings()
    s.description="Random parser"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        # Parsing options
        ("--parse"; default=grctxt; help="file in plain text format to be parsed")
        ("--udpipe"; default=grcudp; help="UDpipe model file")
        ("--ptype"; default="ArcEagerR1"; help="Parser type")
        ("--fast"; action=:store_true; help="fewer messages")
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
    end

    # Process args
    o = parse_args(args, s; as_symbols=true)
    @msg(s.description)
    @msg(string("opts=",[(k,v) for (k,v) in o]...))
    if o[:seed] > 0; setseed(o[:seed]); end
    ptype = eval(parse(o[:ptype]))

    # Parse
    if o[:parse] != nothing
        # we need to test with predicted toks and tags from txt data.
        txtfile = tempname()
        @tm run(pipeline(`udpipe --tokenize --tag $(o[:udpipe]) $(o[:parse])`, txtfile))
        @tm global corpus = readconllu(txtfile); rm(txtfile)
        @tm parses = rparse(ptype, corpus)
        @tm writeconllu(corpus, parses)
    end
end

if basename(PROGRAM_FILE)==basename(@__FILE__); main(ARGS); end
