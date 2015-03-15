using KUparser
using MAT

function readcorpus(f,c)
    d = KUparser.Sentence[]
    for ss in read(f,c)
        s = KUparser.Sentence()
        s.wvec = ss["wvec"]
        s.head = isa(ss["head"], Number) ? [ss["head"]] : vec(ss["head"])
        push!(d, s)
    end
    return d
end

function loadcorpus(fname)
    f=matopen(fname)
    trn = readcorpus(f, "trn")
    dev = readcorpus(f, "dev")
    tst = readcorpus(f, "tst")
    close(f)
    (trn,dev,tst)
end
