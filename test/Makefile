PTB=/mnt/ai/data/ldc/treebank3/parsed/mrg/wsj
ACL11=/mnt/home/dyuret/src/acl11/data/en
CONLL07=/mnt/ai/home/vcirik/eParse/run/embedded/conll07EnglishToken_wikipedia2MUNK-100
WSJ=/mnt/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-100

wvec: \
acl11.dev.wvec \
acl11.tst.wvec \
acl11.trn.wvec \
conll07.tst.wvec \
conll07.dev.wvec \
conll07.trn.wvec \
# wsj.dev.wvec \
# wsj.tst.wvec \
# wsj.trn.wvec

acl11.dev.wvec: ${ACL11}/devr.txt wsj.dev.wvec
	join.pl $^ | awk '{if($$3!=""){print $$1,$$2,$$3+1,$$4,$$9}else{print $$3}}' OFS='\t' FS='\t' > $@

acl11.tst.wvec: ${ACL11}/testr.txt wsj.tst.wvec
	join.pl $^ | awk '{if($$3!=""){print $$1,$$2,$$3+1,$$4,$$9}else{print $$3}}' OFS='\t' FS='\t' > $@

acl11.trn.wvec: ${ACL11}/train.txt wsj.trn.wvec
	join.pl $^ | awk '{if($$3!=""){print $$1,$$2,$$3+1,$$4,$$9}else{print $$3}}' OFS='\t' FS='\t' > $@

# conll07.tst.wvec: ${CONLL07}/01/english_0101.dp
# 	cut -f2,5,7,8,11 $< > $@

# conll07.trn.wvec: ${CONLL07}/00/english_0001.dp
# 	cut -f2,5,7,8,11 $< > $@

conll07.dev.wvec: conll07.dev.dp wsj.dev.wvec
	join.pl $^ | cut -f2,4,7,8,15 > $@

conll07.dev.dp: ${PTB} pennconverter.jar
	cat ${PTB}/22/*.mrg | java -jar pennconverter.jar -conll2007 |\
	perl -pe 's/\tPRD\t/\tVMOD\t/;s/\tCLF\t/\tADV\t/;s/\tDEP\t/\tAMOD\t/' > $@

# PRD and CLF do not exist in conll07
# DEP does exist but most DEPs in pennconverter output are AMODs in conll07
# conll07 has TMP which does not appear in penncoverter output.

# wsj.dev.wvec: ${WSJ}/01/wsj_0101.dp
# 	cut -f2,5,7,8,11 $< > $@

# # original file has multiple sections and pos in column 4
# wsj.tst.wvec: ${WSJ}/02/wsj_0201.dp
# 	head -157098 $< | tail -59100 | cut -f2,4,7,8,11 > $@

# wsj.trn.wvec: ${WSJ}/00/wsj_0001.dp
# 	cut -f2,5,7,8,11 $< > $@


dict: \
acl11.dict \
wsj.dict \
conll07.dict

%.dict: %.trn.wvec %.dev.wvec %.tst.wvec
	cat $^ | julia conll2dict.jl $@

conll07.dict: conll07.trn.wvec conll07.tst.wvec
	cat $^ | julia conll2dict.jl $@


jld4: \
acl11.dev.jld4 \
acl11.trn.jld4 \
acl11.tst.jld4 \
conll07.trn.jld4 \
conll07.dev.jld4 \
conll07.tst.jld4 \
wsj.dev.jld4 \
wsj.trn.jld4 \
wsj.tst.jld4

%.trn.jld4: conll2jld.jl %.dict %.trn.wvec 
	julia $^ $@

%.dev.jld4: conll2jld.jl %.dict %.dev.wvec 
	julia $^ $@

%.tst.jld4: conll2jld.jl %.dict %.tst.wvec 
	julia $^ $@

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
