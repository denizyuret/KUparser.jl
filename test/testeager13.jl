using HDF5,JLD,KUparser
@date d = load("conll07.tst.jld4")
@date ndeps = length(d["deprel"])
@date s1 = d["corpus"][1]
@date p1 = KUparser.Parser{:ArcEager13}(wcnt(s1), ndeps)
@date r0 = rparse(:ArcEager13, d["corpus"], ndeps)
@show evalparse(r0, d["corpus"])
@date f1 = Flist.tacl13hybrid
@date r1 = oparse(:ArcEager13, d["corpus"], f1, ndeps)
@show evalparse(r1[1], d["corpus"])
@date f2 = Flist.acl11eager
@date r2 = oparse(:ArcEager13, d["corpus"], f2, ndeps)
@show evalparse(r2[1], d["corpus"])
@date net = KUnet.newnet(KUnet.relu, KUparser.flen(p1,s1,f1), 20000, p1.nmove)
@date r3 = gparse(:ArcEager13, d["corpus"], net, f1, ndeps)
@show evalparse(r3[1], d["corpus"])
@date r4 = gparse(:ArcEager13, d["corpus"], net, f1, ndeps, 9999)
@show evalparse(r4[1], d["corpus"])
:ok
