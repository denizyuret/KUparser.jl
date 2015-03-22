NCPU=20

jld3: \
acl11.dev.jld3 \
acl11.trn.jld3 \
acl11.tst.jld3 \
conll07.trn.jld3 \
conll07.tst.jld3 \
wsj.dev.jld3 \
wsj.trn.jld3 \
wsj.tst.jld3

%.trn.jld3: conll2jld.jl %.dict %.trn.dp 
	julia $^ $@

%.dev.jld3: conll2jld.jl %.dict %.dev.dp 
	julia $^ $@

%.tst.jld3: conll2jld.jl %.dict %.tst.dp 
	julia $^ $@

dict: \
acl11.dict \
wsj.dict \
conll07.dict

%.dict: %.trn.dp %.dev.dp %.tst.dp
	cat $^ | julia conll2dict.jl $@

conll07.dict: conll07.trn.dp conll07.tst.dp
	cat $^ | julia conll2dict.jl $@

op.mat: \
acl11.dev.op.mat \
acl11.trn.op.mat \
acl11.tst.op.mat \
conll07.trn.op.mat \
conll07.tst.op.mat \
hsensoy.dev.op.mat \
hsensoy.trn.op.mat \
hsensoy.tst.op.mat

conll07.%.op.mat: conll07.%.jld
	julia --depwarn=no -p ${NCPU} oparse2mat.jl $< fv022b $@

%.op.mat: %.jld
	julia --depwarn=no -p ${NCPU} oparse2mat.jl $< fv021a $@
