using HDF5,JLD,MAT
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
@date require("KUparser")
@date d = load(ARGS[1])
d = collect(values(d))
@assert length(d)==1 "$ARGS[1] has more than one variable"
d = d[1]
feats = eval(parse("KUparser.Flist.$(ARGS[2])"))
f = fill(feats, length(d))
@date p = pmap(KUparser.oparse, d, f)
x = map(z->z[2], p)
y = map(z->z[3], p)
matwrite(ARGS[3], Dict("x" => x, "y" => y))
