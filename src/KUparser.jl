module KUparser

using Compat
using KUnet

# TOGO:
typealias ParserType Symbol
# include("archybrid.jl");
# include("arceager.jl");
# include("arceager13.jl");
# include("archybrid13.jl");

include("util.jl");	  export @date, evalparse, pxyequal, testnet
include("sentence.jl");   export Sentence, Corpus, wdim, wcnt, wtype
include("parser.jl");     export ArcEager13, ArcEagerR1, ArcHybrid13, ArcHybridR1, Parser, reset!
include("model.jl");      export Model # extends train and predict
include("features.jl");   export Feature, Fvec, flen, xsize, ysize
include("gparser.jl");    export gparse
include("bparser.jl");    export bparse
include("oparser.jl");    export oparse
include("rparser.jl");    export rparse
include("flist.jl");      export Flist

end # module

include("resetworkers.jl");
