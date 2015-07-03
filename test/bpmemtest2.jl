using HDF5,JLD,KUparser,KUnet,Base.Test
using KUnet: gpumem
isdefined(:xtrn) || (@date @load "zn11cpv.jld")
@show map(size, (ytrn, xtrn, xdev, xtst))
# @show net=newnet("bp10acl11eager13.jld")
ft = Flist.zn11cpv
pt = ArcEager13
ncpu = 16
nbatch = 128
nbeam = 64
#ndeps = length(deprel)
#p0 = pt(1,ndeps)
h0 = 20000
net = [Mmul(h0), Bias(), Relu(), 
       Mmul(size(ytrn,1)), Bias(), Logp(), LogpLoss()]
setparam!(net, adagrad=1e-8)
@show gpumem();
@show (gc();gpumem();)
@date train(net, xtrn, ytrn; iters=100)
@show gpumem();
@show (gc();gpumem();)
# @date net = cpucopy(net)
# @show gpumem();
# @show (gc();gpumem();)

# @date pxy1=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch, ncpu; xy=true)
# @date pxy2=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch; xy=true)
# @test @show isequal(pxy1, pxy2)

# Space: Need ncpu x pbatch x nbeam < 7680 to fit in K20 memory

# * iui-k40-memory: Each K40 has 11.5GB for a total of 23GB.  The
# ZN11 ArcEager13 setup takes in MB GPU memory:
#   280*ncpu+0.1*ncpu*nbeam*nbatch
# For ncpu=20, nbeam=64, nbatch=128 we hit the upper limit.

# With that configuration 20cpu parse of acl11.trn takes 2336s.
