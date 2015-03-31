using HDF5,JLD,KUparser,Base.Test
@date @load "acl11.trn.jld4"
ft = Flist.acl11eager
pt = ArcEager13
ncpu = 20
ndeps = length(deprel)
@date pxy1=oparse(pt, corpus, ndeps, ncpu, ft)
@show map(size, pxy1)
@date pxy2=oparse(pt, corpus, ndeps, ft)
@test @show isequal(pxy1, pxy2)

# No gpu use here we are only looking at memory.
# The raw data acl11.trn.jld4 is approx 1GB.
# The x created (2565,1820392) is approx 5GB.

# On ilac, the 12-cpu version:
# ps auxww shows ~2GB on each cpu.
# free shows 8GB used total.
# It takes 35 seconds, 15 seconds of which for worker init/term.

# On ilac the 1-cpu version:
# both free and ps show:
# Goes up to about 20GB, (an extra 12GB?) takes 67 seconds.
# ps shows but free doesn't:
# This goes up to 40GB during the isequal.

# On iui memory data is the same.
# 20-cpu time=28s (18s init/term)
# 1-cpu time=61s
