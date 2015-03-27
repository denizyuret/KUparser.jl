using HDF5,JLD,KUparser
@date d = load("conll07.tst.jld4")
@date ndeps = length(d["deprel"])
@date s1 = d["corpus"][1]
@date p1 = Parser{:ArcHybrid}(wcnt(s1), ndeps)
@date r0 = rparse(:ArcHybrid, d["corpus"], ndeps)
@date f1 = Flist.tacl13hybrid
@date r1 = oparse(:ArcHybrid, d["corpus"], f1, ndeps)
@date f2 = Flist.acl11eager
@date r2 = oparse(:ArcHybrid, d["corpus"], f2, ndeps)
:ok
