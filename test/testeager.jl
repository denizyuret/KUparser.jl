using HDF5,JLD,KUparser
@date d = load("conll07.tst.jld4")
@date ndeps = length(d["deprel"])
@date s1 = d["corpus"][1]
@date p1 = Parser{:ArcEager}(wcnt(s1), ndeps)
@date r0 = rparse(:ArcEager, d["corpus"], ndeps)
@show evalparse(r0, d["corpus"])
@date f1 = Flist.tacl13hybrid
@date r1 = oparse(:ArcEager, d["corpus"], f1, ndeps)
@show evalparse(r1[1], d["corpus"])
@date f2 = Flist.acl11eager
@date r2 = oparse(:ArcEager, d["corpus"], f2, ndeps)
@show evalparse(r2[1], d["corpus"])
:ok
