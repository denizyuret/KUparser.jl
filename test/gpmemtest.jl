using HDF5,JLD,KUparser,KUnet,Base.Test
@date @load "acl11.trn.jld4"
@show net=newnet("bp10acl11eager13.jld")
ft = Flist.acl11eager
pt = ArcEager13
ncpu = 20
ndeps = length(deprel)
nbatch = 2000
@date pxy1=gparse(pt, corpus, ndeps, ft, net, nbatch, ncpu; xy=true)
@date pxy2=gparse(pt, corpus, ndeps, ft, net, nbatch; xy=true)
@test @show pxyequal(pxy1, pxy2)

# The 12-cpu version takes 150 seconds on ilac.
# The RAM usage does not exceed 12GB according to free.
# The 1-cpu version takes 212 seconds on ilac.
# The RAM usage goes up to 22GB according to free.

# GPU mem:
# For 12-cpu nbatch=500: main=673 children=320 (peak at 380MB)
# (could unload main net for multi-cpu work.)
# For 1-cpu nbatch=500: main<=717MB.

# On iui:
# For 20-cpu nbatch=2000: main=681 children=460 time=70s (20s for init/term)
# For 1-cpu nbatch=2000: main<=900 time=147s
# Same with julia4, except init takes 10s longer.
# pxyequal() runs out of RAM and gets killed.
