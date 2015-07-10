type Sentence form; postag; head; deprel; wvec; Sentence()=new(); end
typealias Corpus AbstractVector{Sentence}
wdim(s::Sentence)=size(s.wvec,1)
wcnt(s::Sentence)=size(s.wvec,2)
wtype(s::Sentence)=eltype(s.wvec)
wdim(c::Corpus)=size(c[1].wvec,1)
wcnt(c::Corpus)=(n=0;for s in c; n+=wcnt(s); end; n)
wtype(c::Corpus)=eltype(c[1].wvec)

function Base.show(io::IO, s::Sentence)
    print(io, (wcnt(s) <= 6 ?
               join(s.form, " ") :
               join([s.form[1:3], "...", s.form[end-2:end]], " ")))
end
