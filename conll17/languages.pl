#!/usr/bin/perl -w
use strict;
while(<>) {
    my @a = split;
    for my $i ("train","dev") {
	for my $j ("conllu", "txt") {
	    my $p = "/mnt/ai/data/nlp/conll17/ud-treebanks-conll2017/UD_$a[0]/$a[1]-ud-$i.$j";
	    warn $p unless -e $p;
	}
    }
    my $p1 = "/mnt/ai/data/nlp/conll17/UDPipe/udpipe-ud-2.0-conll17-170315/models/$a[2]-ud-2.0-conll17-170315.udpipe";
    warn $p1 unless -e $p1;
    my $p2 = "/mnt/ai/data/nlp/conll17/word-embeddings-conll17/$a[3]/$a[4].vectors.xz";
    warn $p2 unless -e $p2;
}
