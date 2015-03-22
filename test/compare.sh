#!/bin/bash -x
TRN1=data/en/train.txt
DEV1=data/en/devr.txt
TST1=data/en/testr.txt
TRN2=/mnt/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-100/00/wsj_0001.dp
DEV2=/mnt/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-100/01/wsj_0101.dp
TST2a=/mnt/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-100/02/wsj_0201.dp
TST2=/tmp/sec23
head -157098 $TST2a | tail -59100 > $TST2

awk '{print $1}' $TRN1 > /tmp/trn11
awk '{print $3}' $TRN2 > /tmp/trn21
join.pl /tmp/trn11 /tmp/trn21 > /tmp/trn-word
echo `awk '$1!=$2' /tmp/trn-word|wc -l`/`wc -l /tmp/trn-word`

awk '{print $2}' $TRN1 > /tmp/trn12
awk '{print $4}' $TRN2 > /tmp/trn22
join.pl /tmp/trn12 /tmp/trn22 > /tmp/trn-pos
echo `awk '$1!=$2' /tmp/trn-pos|wc -l`/`wc -l /tmp/trn-pos`

awk '{print($3==""?$3:1+$3)}' $TRN1 > /tmp/trn13
awk '{print $7}' $TRN2 > /tmp/trn23
join.pl /tmp/trn13 /tmp/trn23 > /tmp/trn-head
echo `awk '$1!=$2' /tmp/trn-head|wc -l`/`wc -l /tmp/trn-head`

awk '{print $4}' $TRN1 > /tmp/trn14
awk '{print $8}' $TRN2 > /tmp/trn24
join.pl /tmp/trn14 /tmp/trn24 > /tmp/trn-deprel
echo `awk '$1!=$2' /tmp/trn-deprel|wc -l`/`wc -l /tmp/trn-deprel`

awk '{print $1}' $DEV1 > /tmp/dev11
awk '{print $3}' $DEV2 > /tmp/dev21
join.pl /tmp/dev11 /tmp/dev21 > /tmp/dev-word
echo `awk '$1!=$2' /tmp/dev-word|wc -l`/`wc -l /tmp/dev-word`

awk '{print $2}' $DEV1 > /tmp/dev12
awk '{print $4}' $DEV2 > /tmp/dev22
join.pl /tmp/dev12 /tmp/dev22 > /tmp/dev-pos
echo `awk '$1!=$2' /tmp/dev-pos|wc -l`/`wc -l /tmp/dev-pos`

awk '{print($3==""?$3:1+$3)}' $DEV1 > /tmp/dev13
awk '{print $7}' $DEV2 > /tmp/dev23
join.pl /tmp/dev13 /tmp/dev23 > /tmp/dev-head
echo `awk '$1!=$2' /tmp/dev-head|wc -l`/`wc -l /tmp/dev-head`

awk '{print $4}' $DEV1 > /tmp/dev14
awk '{print $8}' $DEV2 > /tmp/dev24
join.pl /tmp/dev14 /tmp/dev24 > /tmp/dev-deprel
echo `awk '$1!=$2' /tmp/dev-deprel|wc -l`/`wc -l /tmp/dev-deprel`

awk '{print $1}' $TST1 > /tmp/tst11
awk '{print $3}' $TST2 > /tmp/tst21
join.pl /tmp/tst11 /tmp/tst21 > /tmp/tst-word
echo `awk '$1!=$2' /tmp/tst-word|wc -l`/`wc -l /tmp/tst-word`

awk '{print $2}' $TST1 > /tmp/tst12
awk '{print $4}' $TST2 > /tmp/tst22
join.pl /tmp/tst12 /tmp/tst22 > /tmp/tst-pos
echo `awk '$1!=$2' /tmp/tst-pos|wc -l`/`wc -l /tmp/tst-pos`

awk '{print($3==""?$3:1+$3)}' $TST1 > /tmp/tst13
awk '{print $7}' $TST2 > /tmp/tst23
join.pl /tmp/tst13 /tmp/tst23 > /tmp/tst-head
echo `awk '$1!=$2' /tmp/tst-head|wc -l`/`wc -l /tmp/tst-head`

awk '{print $4}' $TST1 > /tmp/tst14
awk '{print $8}' $TST2 > /tmp/tst24
join.pl /tmp/tst14 /tmp/tst24 > /tmp/tst-deprel
echo `awk '$1!=$2' /tmp/tst-deprel|wc -l`/`wc -l /tmp/tst-deprel`

