using HDF5,JLD,KUparser
@date @load "wsj.dict"
@date @load "wsj.dev.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);
@date @load "wsj.tst.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);
@date @load "wsj.trn.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);

@date @load "acl11.dict"
@date @load "acl11.dev.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);
@date @load "acl11.tst.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);
@date @load "acl11.trn.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);

@date @load "conll07.dict"
@date @load "conll07.tst.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);
@date @load "conll07.trn.jld3"
@date p=KUparser.oparse(corpus,KUparser.Flist.fv021a,length(deprel),20);
