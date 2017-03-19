module KUparser

using Compat, Knet, DistributedArrays

include("util.jl");	  export @date, evalparse #, testnet
include("sentence.jl");   export Sentence, Corpus, wdim, wcnt, wtype
include("parser.jl");     export ArcEager13, ArcEagerR1, ArcHybrid13, ArcHybridR1, Parser, reset!
# include("sfeatures.jl");
include("features.jl");   export Fvec, flen, xsize, ysize
include("gparser.jl");    export gparse
include("bparser.jl");    export bparse, bparse_pmap
include("oparser.jl");    export oparse
include("rparser.jl");    export rparse
include("flist.jl");      export Flist

end # module

include("resetworkers.jl");
