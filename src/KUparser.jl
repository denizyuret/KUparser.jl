module KUparser

using Compat
using KUnet

abstract Parser
type Sentence form; postag; head; deprel; wvec; Sentence()=new(); end
typealias Corpus AbstractVector{Sentence}
wdim(s)=size(s.wvec,1)
wcnt(s)=size(s.wvec,2)

include("util.jl");	  export @date, evalparse
include("archybrid.jl");  # export ArcHybrid
include("flist.jl");      export Features, Flist 	# these are feature templates
include("features.jl");   export features, flen, fidx   # these extract features
include("gparser.jl");    export gparse
include("bparser.jl");    export bparse
include("oparser.jl");    export oparse
include("rparser.jl");    export rparse

end # module

include("resetworkers.jl");
