using HDF5,JLD,KUparser,KUnet,Base.Test

# Load data
@date @load "acl11.trn.jld4"
ncpu = 20
nbatch = 99999
ndeps = length(deprel)
ft = Flist.acl11eager
pt = ArcEager13
p0 = pt(1,ndeps)
s1 = corpus[1]
net = newnet(relu, flen(p0,s1,ft), 20000, p0.nmove; learningRate=1)
net[end].f = logp

# Multi-cpu processing
@date p7=gparse(pt, corpus, ndeps, ft, net, nbatch, ncpu)
@date pxy8=gparse(pt, corpus, ndeps, ft, net, nbatch, ncpu; xy=true)
@show map(size, pxy8)
@test @show isequal(p7, pxy8[1])
