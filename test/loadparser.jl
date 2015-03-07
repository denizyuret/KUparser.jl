using KUnet
using MAT
include("../src/KUparser.jl")
include("../src/flist.jl")
isdefined(:d1mat) || (f=matopen("d1.mat");d1mat=read(f,"d1");close(f))

function readcorpus(m)
    d = KUparser.Sentence[]
    for ss in m
        s = KUparser.Sentence()
        s.wvec = ss["wvec"]
        s.head = isa(ss["head"], Number) ? [ss["head"]] : vec(ss["head"])
        push!(d, s)
    end
    return d
end

dev=readcorpus(d1mat["dev"])
trn=readcorpus(d1mat["trn"])
tst=readcorpus(d1mat["tst"])
s=dev[1]
p=KUparser.ArcHybrid(size(s.wvec,2))
v=KUparser.valid(p)
feats=Flist.fv021a
net = [KUnet.Layer("m11.h5"), KUnet.Layer("m12.h5")]
x=KUparser.features(p,s,feats)
y=KUnet.predict(net,x)
