#!/bin/bash -x
TRN1=data/en/train.txt
DEV1=data/en/devr.txt
TST1=data/en/testr.txt
TRN2=/mnt/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-100/00/wsj_0001.dp
DEV2=/mnt/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-100/01/wsj_0101.dp
TST2a=/mnt/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-100/02/wsj_0201.dp
TST2=/tmp/sec23
head -157098 $TST2a | tail -59100 > $TST2
wc -l $TRN1 $TRN2 $DEV1 $DEV2 $TST1 $TST2
join.pl $TRN1 $TRN2 | awk '{if($3!=""){print $1,$2,$3+1,$4,$15}else{print $3}}' OFS='\t' FS='\t' > acl11.trn
join.pl $DEV1 $DEV2 | awk '{if($3!=""){print $1,$2,$3+1,$4,$15}else{print $3}}' OFS='\t' FS='\t' > acl11.dev
join.pl $TST1 $TST2 | awk '{if($3!=""){print $1,$2,$3+1,$4,$15}else{print $3}}' OFS='\t' FS='\t' > acl11.tst
