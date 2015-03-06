using KUnet
using MAT
include("../src/KUparser.jl")
include("../src/flist.jl")
isdefined(:d1) || (f=matopen("d1.mat");d1=read(f,"d1");close(f))

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

dev=readcorpus(d1["dev"])
trn=readcorpus(d1["trn"])
tst=readcorpus(d1["tst"])
s=dev[1]
p=KUparser.ArcHybrid(size(s.wvec,2))
v=KUparser.valid(p)
f=Flist.fv021a
net = [KUnet.Layer("m11.h5"), KUnet.Layer("m12.h5")]
x=KUparser.features(p,s,f)
y=KUnet.predict(net,x)
