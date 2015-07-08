using HDF5,JLD,KUparser,KUnet

@date net = loadnet("foo11nnet3.jld")
#@date d = load("acl11.trn.jld4")
@date d = load("acl11.dev.jld4")
@date co = d["corpus"][1:100]
@show (ndeps,nbeam,nbatch,ncpu) = (length(d["deprel"]),64,5,nworkers())
@show ft = Flist.zn11cpv
@show pt = ArcEager13

# @date Main.restartmachines()
# @date Main.restartcuda()
# @everywhere using KUparser

@show (pt, length(co), ndeps, length(ft), length(net), nbeam, nbatch, ncpu)
@date p1 = bparse(pt, co, ndeps, ft, net, nbeam, nbatch, ncpu; xy=true)

eqparse(a,b)=(isequal(a[1],b[1]) && isequal(sortcols(vcat(a[2],a[3])), sortcols(vcat(b[2],b[3]))))
@date p2 = bparse(pt, co, ndeps, ft, net, nbeam, nbatch; xy=true)
@show eqparse(p1,p2)


# @date (p0,x0,y0) = bparse(pt, corpus[1:10], ndeps, ft, net, nbeam; xy=true)
# @date (p1,x1,y1) = bparse(pt, corpus, ndeps, ft, net, nbeam; xy=true)
# @date (p2,x2,y2) = bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch; xy=true)
# @date (p3,x3,y3) = bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch, ncpu; xy=true)

# @date bparse(pt, co, ndeps, ft, net, nbeam, 128; xy=true)
# @date bparse(pt, co, ndeps, ft, net, nbeam, 64; xy=true)
# @date bparse(pt, co, ndeps, ft, net, nbeam, 32; xy=true)
# @date bparse(pt, co, ndeps, ft, net, nbeam, 16; xy=true)
# @date bparse(pt, co, ndeps, ft, net, nbeam, 8; xy=true)
# @date bparse(pt, co, ndeps, ft, net, nbeam, 4; xy=true)
# @date bparse(pt, co, ndeps, ft, net, nbeam, 2; xy=true)
# @date bparse(pt, co, ndeps, ft, net, nbeam; xy=true)

:ok
