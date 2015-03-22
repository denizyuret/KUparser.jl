NCPU=20

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
