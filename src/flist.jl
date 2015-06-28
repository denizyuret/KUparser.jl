module Flist                    # feature matrices
using KUparser: SFeature, DFeature # sparse and dense features

# ZN11 dense variants.  The original has:
# (w,p) for (s,n,n1,n2,sh,sh2,sl,sr,nl,sl2,sr2,nl2): 24 features
# L for (s,sh,sl,sr,nl,sl2,sr2,nl2): 8 features
# sd, sa, sb, sA, sB, na, nA: 7 features = 39
# We replace (w,p) with 8 subsets of (v,c,p)

zn11n = DFeature["sA","nA","sa","na","sB","sb","sd",
                 "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L"]

zn11p = DFeature["sA","nA","sa","na","sB","sb","sd",
                 "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L",
                 "sp","np","n1p","n2p","shp","sh2p","slp","srp","nlp","sl2p","sr2p","nl2p"]

zn11v = DFeature["sA","nA","sa","na","sB","sb","sd",
                 "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L",
                 "sv","nv","n1v","n2v","shv","sh2v","slv","srv","nlv","sl2v","sr2v","nl2v"]

zn11c = DFeature["sA","nA","sa","na","sB","sb","sd",
                 "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L",
                 "sc","nc","n1c","n2c","shc","sh2c","slc","src","nlc","sl2c","sr2c","nl2c"]

zn11pv = DFeature["sA","nA","sa","na","sB","sb","sd",
                  "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L",
                  "sp","np","n1p","n2p","shp","sh2p","slp","srp","nlp","sl2p","sr2p","nl2p",
                  "sv","nv","n1v","n2v","shv","sh2v","slv","srv","nlv","sl2v","sr2v","nl2v"]

zn11cv = DFeature["sA","nA","sa","na","sB","sb","sd",
                  "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L",
                  "sc","nc","n1c","n2c","shc","sh2c","slc","src","nlc","sl2c","sr2c","nl2c",
                  "sv","nv","n1v","n2v","shv","sh2v","slv","srv","nlv","sl2v","sr2v","nl2v"]

zn11cp = DFeature["sA","nA","sa","na","sB","sb","sd",
                  "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L",
                  "sc","nc","n1c","n2c","shc","sh2c","slc","src","nlc","sl2c","sr2c","nl2c",
                  "sp","np","n1p","n2p","shp","sh2p","slp","srp","nlp","sl2p","sr2p","nl2p"]

zn11cpv = DFeature["sA","nA","sa","na","sB","sb","sd",
                   "sL","shL","slL","srL","nlL","sl2L","sr2L","nl2L",
                   "sc","nc","n1c","n2c","shc","sh2c","slc","src","nlc","sl2c","sr2c","nl2c",
                   "sp","np","n1p","n2p","shp","sh2p","slp","srp","nlp","sl2p","sr2p","nl2p",
                   "sv","nv","n1v","n2v","shv","sh2v","slv","srv","nlv","sl2v","sr2v","nl2v"]

# Sparse feature sets
#
# Each string specifies a particular feature and has the form:
#   [sn]\d?([hlr]\d?)*[wpdLabAB]
# 
# The meaning of each letter is below.  In the following i is a single
# digit integer which is optional if the default value is used:
#
# si: i'th stack word, default i=0 means top
# ni: i'th buffer word, default i=0 means first
# hi: i'th degree head, default i=1 means direct head
# li: i'th leftmost child, default i=1 means the leftmost child 
# ri: i'th rightmost child, default i=1 means the rightmost child
# w: word
# p: postag
# d: distance to the right.  e.g. s1d is s0s1 distance, s0d is s0n0
#    distance.  encoding: 1,2,3,4,5-10,10+ (from ZN11)
# L: dependency label (0 is ROOT or NONE)
# a: number of left children.  following ZN11, any number is allowed.
# b: number of right children.  following ZN11, any number is allowed.
# A: set of left dependency labels
# B: set of right dependency labels


# The original zn11 set in sparse format with feature conjunctions
zn11orig = SFeature[
["sw"]                  # STw		StackWord
["sp"]                  # STt		StackTag
["sw","sp"]             # STwt		StackWordTag
["nw"]                  # N0w		NextWord
["np"]                  # N0t		NextTag
["nw","np"]             # N0wt		NextWordTag
["n1w"]                 # N1w		Next+1Word
["n1p"]                 # N1t		Next+1Tag
["n1w","n1p"]           # N1wt		Next+1WordTag
["n2w"]                 # N2w		Next+2Word
["n2p"]                 # N2t		Next+2Tag
["n2w","n2p"]           # N2wt		Next+2WordTag
["shw"]                 # STHw		StackHeadWord
["shp"]                 # STHt		StackHeadTag
["sL"]                  # STi		StackLabel
["sh2w"]                # STHHw		StackHeadHeadWord
["sh2p"]                # STHHt		StackHeadHeadTag
["shL"]                 # STHi		StackLabel ?? StackHeadLabel
["slw"]                 # STLDw		StackLDWord
["slp"]                 # STLDt		StackLDTag
["slL"]                 # STLDi		StackLDLabel
["srw"]                 # STRDw		StackRDWord
["srp"]                 # STRDt		StackRDTag
["srL"]                 # STRDi		StackRDLabel
["nlw"]                 # N0LDw		NextLDWord
["nlp"]                 # N0LDt		NextLDTag
["nlL"]                 # N0LDi		NextLDLabel
["sl2w"]                # STL2Dw	StackL2DWord
["sl2p"]                # STL2Dt	StackL2DTag
["sl2L"]                # STL2Di	StackL2DLabel
["sr2w"]                # STR2Dw	StackR2DWord
["sr2p"]                # STR2Dt	StackR2DTag
["sr2L"]                # STR2Di	StackR2DLabel
["nl2w"]                # N0L2Dw	NextL2DWord
["nl2p"]                # N0L2Dt	NextL2DTag
["nl2L"]                # N0L2Di	NextL2DLabel
			## HTw		HeadStackWord
			## HTt		HeadStackTag
			## HTwt		HeadStackWordTag
["sw","sp","nw","np"]	# STwtN0wt	StackWordTagNextWordTag
["sw","sp","nw"]	# STwtN0w	StackWordTagNextWord
["sw","nw","np"]	# STwN0wt	StackWordNextWordTag
["sp","nw","np"]	# STtN0wt	StackTagNextWordTag
["sw","sp","np"]	# STwtN0t	StackWordTagNextTag
["sw","nw"]             # STwN0w	StackWordNextWord
["sp","np"]             # STtN0t	StackTagNextTag
["np","n1p"]            # N0tN1t	NextTagNext+1Tag
["np","n1p","n2p"]	# N0tN1tN2t	NextTagTrigram
["sp","np","n1p"]	# STtN0tN1t	StackTagNextTagNext+1Tag
["sp","np","nlp"]	# STtN0tN0LDt	StackTagNextTagNextLDTag
["np","nlp","nl2p"]     # N0tN0LDtN0L2Dt	StackTagNextTagNextLDTagNextTagNextL2DTag ?? NextTagNextLDTagNextL2DTag
["shp","sp","np"]       # STHtSTtN0t	StackHeadTagStackTagNextTag
			## HTtHT2tN0t	HeadStackTagHeadStack2TagNextTag
["sh2p","shp","sp"]     # STHHtSTHtSTt	StackHeadHeadTagStackHeadTagStackTag
["sp","slp","np"]       # STtSTLDtN0t	StackTagStackLDTagNextTag
["sp","slp","sl2p"]     # STtSTLDtSTL2Dt	StackTagStackLDTagStackL2DTag
["sp","srp","np"]       # STtSTRDtN0t	StackTagStackRDTagNextTag
["sp","srp","sr2p"]     # STtSTRDtSTR2Dt	StackTagStackRDTagStackR2DTag
["sw","sd"]             # STwd		StackWordDist
["sp","sd"]             # STtd		StackTagDist
["nw","sd"]             # N0wd		NextWordDist
["np","sd"]             # N0td		NextTagDist
["sw","nw","sd"]        # STwN0wd	StackWordNextWordDist
["sp","np","sd"]        # STtN0td	StackTagNextTagDist
["sw","sb"]             # STwra		StackWordRightArity
["sp","sb"]             # STtra		StackTagRightArity
["sw","sa"]             # STwla		StackWordLeftArity
["sp","sa"]             # STtla		StackTagLeftArity
["nw","na"]             # N0wla		NextWordRightArity ?? NextWordLeftArity
["np","na"]             # N0tla		NextTagRightArity  ?? NextTagLeftArity
["sw","sB"]             # STwrp		StackWordRightSetoftags
["sp","sB"]             # STtrp		StackTagRightSetoftags
["sw","sA"]             # STwlp		StackWordLeftSetoftags
["sp","sA"]             # STtlp		StackTagLeftSetoftags
["nw","nA"]             # N0wlp		Next0WordLeftSetoftags ?? NextWordLeftSetoftags
["np","nA"]             # N0tlp		Next0TagLeftSetoftags  ?? NextTagLeftSetoftags
			## STl		StackLemma
			## STc		StackCPOS
			## STf		StackFeats
			## N0l		NextLemma
			## N0c		NextCPOS
			## N0f		NextFeats
			## N1l		Next+1Lemma
			## N1c		Next+1CPOS
			## N1f		Next+1Feats
]

# The reduced zn11 set in sparse format with singleton features
zn11single = SFeature[
["sw"]                  # STw		StackWord
["sp"]                  # STt		StackTag
#["sw","sp"]             # STwt		StackWordTag
["nw"]                  # N0w		NextWord
["np"]                  # N0t		NextTag
#["nw","np"]             # N0wt		NextWordTag
["n1w"]                 # N1w		Next+1Word
["n1p"]                 # N1t		Next+1Tag
#["n1w","n1p"]           # N1wt		Next+1WordTag
["n2w"]                 # N2w		Next+2Word
["n2p"]                 # N2t		Next+2Tag
#["n2w","n2p"]           # N2wt		Next+2WordTag
["shw"]                 # STHw		StackHeadWord
["shp"]                 # STHt		StackHeadTag
["sL"]                  # STi		StackLabel
["sh2w"]                # STHHw		StackHeadHeadWord
["sh2p"]                # STHHt		StackHeadHeadTag
["shL"]                 # STHi		StackLabel ?? StackHeadLabel
["slw"]                 # STLDw		StackLDWord
["slp"]                 # STLDt		StackLDTag
["slL"]                 # STLDi		StackLDLabel
["srw"]                 # STRDw		StackRDWord
["srp"]                 # STRDt		StackRDTag
["srL"]                 # STRDi		StackRDLabel
["nlw"]                 # N0LDw		NextLDWord
["nlp"]                 # N0LDt		NextLDTag
["nlL"]                 # N0LDi		NextLDLabel
["sl2w"]                # STL2Dw	StackL2DWord
["sl2p"]                # STL2Dt	StackL2DTag
["sl2L"]                # STL2Di	StackL2DLabel
["sr2w"]                # STR2Dw	StackR2DWord
["sr2p"]                # STR2Dt	StackR2DTag
["sr2L"]                # STR2Di	StackR2DLabel
["nl2w"]                # N0L2Dw	NextL2DWord
["nl2p"]                # N0L2Dt	NextL2DTag
["nl2L"]                # N0L2Di	NextL2DLabel
			## HTw		HeadStackWord
			## HTt		HeadStackTag
			## HTwt		HeadStackWordTag
#["sw","sp","nw","np"]	# STwtN0wt	StackWordTagNextWordTag
#["sw","sp","nw"]	# STwtN0w	StackWordTagNextWord
#["sw","nw","np"]	# STwN0wt	StackWordNextWordTag
#["sp","nw","np"]	# STtN0wt	StackTagNextWordTag
#["sw","sp","np"]	# STwtN0t	StackWordTagNextTag
#["sw","nw"]             # STwN0w	StackWordNextWord
#["sp","np"]             # STtN0t	StackTagNextTag
#["np","n1p"]            # N0tN1t	NextTagNext+1Tag
#["np","n1p","n2p"]	# N0tN1tN2t	NextTagTrigram
#["sp","np","n1p"]	# STtN0tN1t	StackTagNextTagNext+1Tag
#["sp","np","nlp"]	# STtN0tN0LDt	StackTagNextTagNextLDTag
#["np","nlp","nl2p"]     # N0tN0LDtN0L2Dt	StackTagNextTagNextLDTagNextTagNextL2DTag ?? NextTagNextLDTagNextL2DTag
#["shp","sp","np"]       # STHtSTtN0t	StackHeadTagStackTagNextTag
			## HTtHT2tN0t	HeadStackTagHeadStack2TagNextTag
#["sh2p","shp","sp"]     # STHHtSTHtSTt	StackHeadHeadTagStackHeadTagStackTag
#["sp","slp","np"]       # STtSTLDtN0t	StackTagStackLDTagNextTag
#["sp","slp","sl2p"]     # STtSTLDtSTL2Dt	StackTagStackLDTagStackL2DTag
#["sp","srp","np"]       # STtSTRDtN0t	StackTagStackRDTagNextTag
#["sp","srp","sr2p"]     # STtSTRDtSTR2Dt	StackTagStackRDTagStackR2DTag
["sd"]
#["sw","sd"]             # STwd		StackWordDist
#["sp","sd"]             # STtd		StackTagDist
#["nw","sd"]             # N0wd		NextWordDist
#["np","sd"]             # N0td		NextTagDist
#["sw","nw","sd"]        # STwN0wd	StackWordNextWordDist
#["sp","np","sd"]        # STtN0td	StackTagNextTagDist
["sb"]
#["sw","sb"]             # STwra		StackWordRightArity
#["sp","sb"]             # STtra		StackTagRightArity
["sa"]
#["sw","sa"]             # STwla		StackWordLeftArity
#["sp","sa"]             # STtla		StackTagLeftArity
["na"]
#["nw","na"]             # N0wla		NextWordRightArity ?? NextWordLeftArity
#["np","na"]             # N0tla		NextTagRightArity  ?? NextTagLeftArity
["sB"]
#["sw","sB"]             # STwrp		StackWordRightSetoftags
#["sp","sB"]             # STtrp		StackTagRightSetoftags
["sA"]
#["sw","sA"]             # STwlp		StackWordLeftSetoftags
#["sp","sA"]             # STtlp		StackTagLeftSetoftags
["nA"]
#["nw","nA"]             # N0wlp		Next0WordLeftSetoftags ?? NextWordLeftSetoftags
#["np","nA"]             # N0tlp		Next0TagLeftSetoftags  ?? NextTagLeftSetoftags
			## STl		StackLemma
			## STc		StackCPOS
			## STf		StackFeats
			## N0l		NextLemma
			## N0c		NextCPOS
			## N0f		NextFeats
			## N1l		Next+1Lemma
			## N1c		Next+1CPOS
			## N1f		Next+1Feats
]


# Dense feature sets:

# Each string specifies a particular feature and has the form:
#   [sn]\d?([hlr]\d?)*[vcpdLabAB]
# 
# The meaning of each letter is below.  In the following i is a single
# digit integer which is optional if the default value is used:
#
# si: i'th stack word, default i=0 means top
# ni: i'th buffer word, default i=0 means first
# hi: i'th degree head, default i=1 means direct head
# li: i'th leftmost child, default i=1 means the leftmost child 
# ri: i'th rightmost child, default i=1 means the rightmost child
# v: word vector
# c: context vector
# p: postag
# d: distance to the right.  e.g. s1d is s0s1 distance, s0d is s0n0
#    distance.  encoding: 1,2,3,4,5-10,10+ (from ZN11)
# L: dependency label (0 is ROOT or NONE)
# a: number of left children.  encoding: 0,1,...,8,9+
# b: number of right children.  encoding: 0,1,...,8,9+
# A: set of left dependency labels
# B: set of right dependency labels


# fs11h54h27 (9144): local best for acl11, ArcHybrid13, start=hybrid27, all=hybrid54, bparser10, shuffled
hybrid25 = split("n0A n0a n0lL n0lc n0lv n0c n0v n1c n1v s0A s0B s0a s0b s0d s0c s0rL s0v s1A s1B s1a s1rL s1rc s1rv s1v s2v")

# fs11e64e27 (9098): local best for acl11, ArcEager13, start=eager27=hybrid27, all=eager64, bparser10, shuffled
eager25 = split("n0A n0a n0l2L n0lL n0lc n0lv n0c n0v n1c n1v n2c n2v s0B s0c s0v s1A s1B s1a s1d s1c s1rL s1rc s1v s2c s2v")

# fs11h54h28 (8997): local best in acl11, ArcEager13 (by mistake), start=hybrid28, all=hybrid54
hybrid27 = split("n0A n0a n0lL n0lc n0lv n0c n0v n1c n1v s0A s0B s0a s0d s0c s0v s1A s1B s1a s1b s1d s1c s1rL s1rc s1rv s1v s2c s2v")
eager27 = hybrid27

# fs11h54h14 (8992): local best for acl11, ArcHybrid13, start=hybrid14, all=hybrid54
hybrid25b = split("n0a n0l2c n0l2v n0lc n0lv n0v n1c n1v n2v s0B s0a s0d s0l2L s0lL s0lc s0c s0rc s0v s1B s1b s1lL s1lc s1c s1r2c s1v")

# fs11e64e20 (8926): local best for acl11, ArcEager13, start=eager20, all=eager64
eager20 = split("n0A n0a n0lL n0c n0v n1c n1v n2c n2v s0B s0L s0b s0d s0hc s0hv s0c s0rL s0v s1c s1v")

# fs11e39e21 (8906): local best for acl11, ArcEager13, start=eager21, all=eager39
eager23 = split("n0A n0lL n0lc n0c n0v n1c n1v n2c n2v s0B s0L s0a s0b s0h2c s0hv s0l2L s0l2c s0l2v s0lc s0c s0r2L s0r2c s0v")

# fs07h54h28 (8797): local best for conll07, ArcHybrid13, start=hybrid28, all=hybrid54
hybrid28b = split("n0A n0a n0l2v n0lL n0lc n0c n0v n1c n1v n2v s0A s0B s0a s0b s0d s0c s0v s1A s1a s1b s1d s1c s1rL s1rc s1rv s1v s2c s2v")

# fs07h54h13 (8791): Local best for conll07, ArcHybrid13, start=tacl13hybrid (aka hybrid13), all=hybrid54
hybrid14 = split("n0lc n0lv n0v n1c n1v s0d s0lc s0c s0rc s0v s1B s1lc s1c s1v")

# fs07e39e39 (8725): local best in conll07, ArcEager13, start=acl11eager, all=acl11eager
eager36 = split("n0A n0a n0l2L n0l2c n0l2v n0lL n0lc n0lv n0c n0v n1c n1v n2c n2v s0A s0B s0L s0b s0d s0h2c s0hL s0hc s0hv s0l2L s0l2c s0l2v s0lL s0lc s0lv s0c s0r2c s0r2v s0rL s0rc s0rv s0v")

# fs07e39e21 (8705): Local best for conll07, ArcEager13, start=eager21, all=eager39
eager20b = split("n0l2L n0lv n0c n0v n1v n2c n2v s0B s0L s0a s0h2c s0hv s0l2L s0l2c s0l2v s0lc s0c s0r2L s0r2c s0v")

# fs07e64e20 (8677): Local best for conll07, ArcEager13, start=eager20, all=eager64
eager21b = split("n0A n0a n0l2L n0lL n0c n0v n1c n1v n2c n2v s0B s0L s0b s0d s0hc s0hv s0c s0rL s0v s1c s1v")

# more complete set than acl11eager: added s1 features
eager64 = split("n0A n0a n0l2L n0l2c n0l2v n0lL n0lc n0lv n0c n0v n1c n1v n2c n2v s0A s0B s0L s0a s0b s0d s0h2c s0h2v s0hL s0hc s0hv s0l2L s0l2c s0l2v s0lL s0lc s0lv s0c s0r2L s0r2c s0r2v s0rL s0rc s0rv s0v s1A s1B s1L s1a s1b s1d s1h2c s1h2v s1hL s1hc s1hv s1l2L s1l2c s1l2v s1lL s1lc s1lv s1c s1r2L s1r2c s1r2v s1rL s1rc s1rv s1v")

# Manually constructed starting point
eager21 = split("n0A n0lL n0c n0v n1c n1v n2c n2v s0B s0L s0a s0h2c s0hv s0l2L s0l2c s0l2v s0lc s0c s0r2L s0r2c s0v")

# Manually constructed starting point
hybrid28 = split("n0A n0a n0lL n0lc n0lv n0c n0v n1c n1v s0A s0B s0a s0b s0d s0c s0v s1A s1B s1a s1b s1d s1c s1rL s1rc s1rv s1v s2c s2v")

# Features that may be relevant for ArcHybrid13
hybrid54 = DFeature[
"n0v", "n0c", "n0A", "n0a",
"s0v", "s0c", "s0A", "s0a", "s0B", "s0b", "s0d",
"s1v", "s1c", "s1A", "s1a", "s1B", "s1b", "s1d",
"n1v", "n1c", "n2v", "n2c", "s2v", "s2c",
"n0lv", "n0l2v", "s0lv", "s0l2v", "s0rv", "s0r2v", "s1lv", "s1l2v", "s1rv", "s1r2v",
"n0lc", "n0l2c", "s0lc", "s0l2c", "s0rc", "s0r2c", "s1lc", "s1l2c", "s1rc", "s1r2c",
"n0lL", "n0l2L", "s0lL", "s0l2L", "s0rL", "s0r2L", "s1lL", "s1l2L", "s1rL", "s1r2L"
]

# Goldberg&Nivre TACL13 ArcHybrid features from: tacl2013dynamicoracles/lefttoright/features:HybridFeatures
# s0,s1,s2: stack word forms: a=-1,-2,-3 b=0 c=4
# w0,w1: buffer word forms: a=0,1 b=0 c=4
# Ts0: tag of s0 (-1 0 -4)
# Trcs1: tag of rightmost child of s1 (-2 1 -4)
# use word half of vector (c=+4) for form, context half of vector (c=-4) for tag
# No need for feature combinations, we have a nonlinear model.

tacl13hybrid = DFeature[
	# (1)
"s0v"	# append("s0_%s" % s0)
"s0c"	# append("Ts0_%s" % Ts0)
        # append("Ts0s0_%s_%s" % (Ts0, s0))

"s1v"	# append("s1_%s" % s1)
"s1c"	# append("Ts1_%s" % Ts1)
     	# append("Ts1s1_%s_%s" % (Ts1, s1))

"n0v"	# append("w0_%s" % w0)
"n0c"	# append("Tw0_%s" % Tw0)
	# append("Tw0w0_%s_%s" % (Tw0, w0))
	# +hybrid
"n1v"	# append("w1_%s" % w1)
"n1c"	# append("Tw1_%s" % Tw1)
	# append("Tw1w1_%s_%s" % (Tw1, w1))

	# (2)
	# append("s0s1_%s_%s" % (s0,s1))
	# append("Ts0Ts1_%s_%s" % (Ts0,Ts1))
	# append("Ts0Tw0_%s_%s" % (Ts0,Tw0))
	# append("s0Ts0Ts1_%s_%s_%s" % (s0,Ts0,Ts1))
	# append("Ts0s1Ts1_%s_%s_%s" % (Ts0,s1,Ts1))
	# append("s0s1Ts1_%s_%s_%s" % (s0,s1,Ts1))
	# append("s0Ts0s1_%s_%s_%s" % (s0,Ts0,s1))
	# append("s0Ts0Ts1_%s_%s_%s" % (s0,Ts0,Ts1))
	# +hybrid   
	# append("s0w0_%s_%s" % (s0,w0))
	# append("Ts0Tw0_%s_%s" % (Ts0,Tw0))
	# append("Ts0Tw1_%s_%s" % (Ts0,Tw1))
	# append("s0Ts0Tw0_%s_%s_%s" % (s0,Ts0,Tw0))
	# append("Ts0w0Tw0_%s_%s_%s" % (Ts0,w0,Tw0))
	# append("s0w0Tw0_%s_%s_%s" % (s0,w0,Tw0))
	# append("s0Ts0w0_%s_%s_%s" % (s0,Ts0,w0))
	# append("w0Tw0Tw1_%s_%s_%s" % (w0,Tw0,Tw1)) #
	
	# (3)
	# append("Ts0Tw0Tw1_%s_%s_%s" % (Ts0,Tw0,Tw1))
	# append("Ts1Ts0Tw0_%s_%s_%s" % (Ts1,Ts0,Tw0))
	# append("s0Tw0Tw1_%s_%s_%s" % (s0,Tw0,Tw1))
	# append("Ts1s0Tw0_%s_%s_%s" % (Ts1,s0,Tw0))
	
	# (4) rc -1  lc 1
"s1rc"	# append("Ts1Trcs1Tw0_%s_%s_%s" % (Ts1, Trcs1, Tw0)) => Trcs1
	# append("Ts1Trcs1Tw0_%s_%s_%s" % (Ts1, Trcs1, Ts0))
"s1lc"	# append("Ts1Tlcs1Ts0_%s_%s_%s" % (Ts1, Tlcs1, Ts0)) => Tlcs1
"s0lc"	# append("Ts1Ts0Tlcs0_%s_%s_%s" % (Ts1, Ts0, Tlcs0)) => Tlcs0
"s0rc"	# append("Ts1Trcs0Ts0_%s_%s_%s" % (Ts1, Trcs0, Ts0)) => Trcs0
	# append("Ts0Tlcs1s0_%s_%s_%s" % (Ts0, Tlcs1, s0))
	# append("Ts1s0Trcs0_%s_%s_%s" % (Ts1, s0, Trcs0))
	# append("Ts0Tlcs1s0_%s_%s_%s" % (Ts0, Trcs1, s0))
	# append("Ts1s0Trcs0_%s_%s_%s" % (Ts1, s0, Tlcs0))
	
	# +hybrid
	# append("Ts0Trcs0Tw0_%s_%s_%s" % (Ts0,Trcs0,Tw0))
	# append("Ts0Trcs0Tw0_%s_%s_%s" % (Ts0,Trcs0,Tw1))
	# append("Ts0Tlcs0Tw0_%s_%s_%s" % (Ts0,Tlcs0,Tw0))
"n0lc"	# append("Ts0Tw0Tlcw0_%s_%s_%s" % (Ts0,Tw0,Tlcw0)) => Tlcw0
	# append("Ts0Tlcs0w0_%s_%s_%s" % (Ts0,Tlcs0,w0))
	# append("Ts0Tlcs0w0_%s_%s_%s" % (Ts0,Trcs0,w0))
	# append("Ts0w0Tlcw0_%s_%s_%s" % (Ts0,w0,Tlcw0))
	
	# append("Ts0Tw0Trcs0Tlcw0_%s_%s_%s_%s" % (Ts0,Tw0,Tlcw0,Trcs0))
	# append("Ts0Ts1Trcs1Tlcs0_%s_%s_%s_%s" % (Ts0,Ts1,Tlcs0,Trcs1))
	
	# (5)
	# append("Ts2Ts1Ts0_%s_%s_%s" % (Ts2,Ts1,Ts0))
	# append("Ts1Ts0Tw0_%s_%s_%s" % (Ts1,Ts0,Tw0))
	# append("Ts0Tw0Tw1_%s_%s_%s" % (Ts0,Tw0,Tw1))
] # tacl13hybrid

# Variants of tacl13hybrid:

tacl13words = DFeature["s0lv", "s0v", "s0rv", "s1lv", "s1v", "s1rv", "n0lv", "n0v", "n1v"]
tacl13tags  = DFeature["s0lc", "s0c", "s0rc", "s1lc", "s1c", "s1rc", "n0lc", "n0c", "n1c"]
tacl13wordtag = [tacl13words; tacl13tags]

# Renamed:
hybrid13 = tacl13hybrid

# Goldberg&Nivre TACL13 ArcEager features from: 
# tacl2013dynamicoracles/lefttoright/features:EagerZhangNivre2011Extractor
# (seems to be the ones used in acl11)
# s0: top of stack (a=-1)
# n0,n1,n2: buffer (a=0,1,2)
# s0h,s0h2: parent and grandparent of s0
# s0l,s0r: left/rightmost child of s0
# s0l2,s0r2: second left/rightmost child of s0
# s0w,s0p,s0wp: form, postag, form-postag for s0
# d: refers to s0-n0 distance, encoded 1,2,3,...,9,10+
# s0vl,s0vr: number of left/right children of s0, flat encoding
# s0L: dependency label for s0 (NA for n0!)
# s0sl,s0sr: set of dependency labels to the left/right of s0

acl11eager = DFeature[
	# # Single Words
	# f("s0wp_%s" % (s0wp))
"s0v"	# f("s0w_%s"  % (s0w))
"s0c"	# f("s0p_%s"  % (s0p))
	# f("n0wp_%s" % (n0wp))
"n0v"	# f("n0w_%s"  % (n0w))
"n0c"	# f("n0p_%s"  % (n0p))
	# f("n1wp_%s" % (n1wp))
"n1v"	# f("n1w_%s"  % (n1w))
"n1c"	# f("n1p_%s"  % (n1p))
	# f("n2wp_%s" % (n2wp))
"n2v"	# f("n2w_%s"  % (n2w))
"n2c"	# f("n2p_%s"  % (n2p))
	# 
	# # Pairs
	# f("s0wp,n0wp_%s_%s" % (s0wp, n0wp))
	# f("s0wp,n0w_%s_%s" % (s0wp, n0w))
	# f("s0w,n0wp_%s_%s" % (s0w, n0wp))
	# f("s0wp,n0p_%s_%s" % (s0wp, n0p))
	# f("s0p,n0wp_%s_%s" % (s0p, n0wp))
	# f("s0w,n0w_%s_%s" % (s0w, n0w)) #?
	# f("s0p,n0p_%s_%s" % (s0p, n0p))
	# f("n0p,n1p_%s_%s" % (n0p, n1p))
	# 
	# # Tuples
	# f("n0p,n1p,n2p_%s_%s_%s" % (n0p, n1p, n2p))
	# f("s0p,n0p,n1p_%s_%s_%s" % (s0p, n0p, n1p))
	# f("s0hp,s0p,n0p_%s_%s_%s" % (s0hp, s0p, n0p)) => s0hp (handled in unigram section)
	# f("s0p,s0lp,n0p_%s_%s_%s" % (s0p, s0lp, n0p)) => s0lp (handled in unigram section)
	# f("s0p,s0rp,n0p_%s_%s_%s" % (s0p, s0rp, n0p)) => s0rp (handled in unigram section)
	# f("s0p,n0p,n0lp_%s_%s_%s" % (s0p, n0p, n0lp)) => n0lp (handled in unigram section)
	# 
	# # Distance
"s0d"	# f("s0wd_%s:%s" % (s0w, d)) => d
	# f("s0pd_%s:%s" % (s0p, d))
	# f("n0wd_%s:%s" % (n0w, d))
	# f("n0pd_%s:%s" % (n0p, d))
	# f("s0w,n0w,d_%s:%s:%s" % (s0w, n0w, d))
	# f("s0p,n0p,d_%s:%s:%s" % (s0p, n0p, d))
	# 
	# # Valence
"s0b"	# f("s0wvr_%s:%s" % (s0w, s0vr)) => s0vr
	# f("s0pvr_%s:%s" % (s0p, s0vr))
"s0a"	# f("s0wvl_%s:%s" % (s0w, s0vl)) => s0vl
	# f("s0pvl_%s:%s" % (s0p, s0vl))
"n0a"	# f("n0wvl_%s:%s" % (n0w, n0vl)) => n0vl
	# f("n0pvl_%s:%s" % (n0p, n0vl))
	# 
	# # Unigrams
"s0hv"	# f("s0hw_%s" % (s0hw))
"s0hc"	# f("s0hp_%s" % (s0hp))
"s0L"	# f("s0L_%s" % (s0L))
	# 
"s0lv"	# f("s0lw_%s" % (s0lw))
"s0lc"	# f("s0lp_%s" % (s0lp))
"s0lL"	# f("s0lL_%s" % (s0lL))
	# 
"s0rv"	# f("s0rw_%s" % (s0rw))
"s0rc"	# f("s0rp_%s" % (s0rp))
"s0rL"	# f("s0rL_%s" % (s0rL))
	# 
"n0lv"	# f("n0lw_%s" % (n0lw))
"n0lc"	# f("n0lp_%s" % (n0lp))
"n0lL"	# f("n0lL_%s" % (n0lL))
	# 
	# # Third-order
	# #do we really need the non-grandparent ones?
"s0h2v"	# f("s0h2w_%s" % (s0h2w))
"s0h2c"	# f("s0h2p_%s" % (s0h2p))
"s0hL"	# f("s0hL_%s"  % (s0hL))
"s0l2v"	# f("s0l2w_%s" % (s0l2w))
"s0l2c"	# f("s0l2p_%s" % (s0l2p))
"s0l2L"	# f("s0l2L_%s" % (s0l2L))
"s0r2v"	# f("s0r2w_%s" % (s0r2w))
"s0r2c"	# f("s0r2p_%s" % (s0r2p))
"s0r2L"	# f("s0r2L_%s" % (s0r2L))
"n0l2v"	# f("n0l2w_%s" % (n0l2w))
"n0l2c"	# f("n0l2p_%s" % (n0l2p))
"n0l2L"	# f("n0l2L_%s" % (n0l2L))
	# f("s0p,s0lp,s0l2p_%s_%s_%s" % (s0p, s0lp, s0l2p))
	# f("s0p,s0rp,s0r2p_%s_%s_%s" % (s0p, s0rp, s0r2p))
	# f("s0p,s0hp,s0h2p_%s_%s_%s" % (s0p, s0hp, s0h2p))
	# f("n0p,n0lp,n0l2p_%s_%s_%s" % (n0p, n0lp, n0l2p))
	# 
	# # Labels
"s0B"	# f("s0wsr_%s_%s" % (s0w, s0sr)) => s0sr
	# f("s0psr_%s_%s" % (s0p, s0sr))
"s0A"	# f("s0wsl_%s_%s" % (s0w, s0sl)) => s0sl
	# f("s0psl_%s_%s" % (s0p, s0sl))
"n0A"	# f("n0wsl_%s_%s" % (n0w, n0sl)) => n0sl
	# f("n0psl_%s_%s" % (n0p, n0sl))
] # tacl13eager

# Renamed:
eager39 = acl11eager
# fs11e39e39 (8906): eager39 also local best for acl11, ArcEager13, start=eager39, all=eager39


# Old feature format before we switched to ints:
# f::Features is a nx3 matrix whose rows determine which features to extract
# Each row of f consists of the following three values:
# 1. anchor word: 0:n0, 1:n1, 2:n2, ..., -1:s0, -2:s1, -3:s2, ...
# 2. target word: 0:self, 1:rightmost child, 2:second rightmost, -1:leftmost, -2:second leftmost ...
# 3. feature: One of the features listed below.
#
# 0:wvec (word+context if token, dim depends on encoding)
# +-1: exists/doesn't (one bit)
# +-2: right/left child count (4 bits, == encoding: 0,1,2,3+)
# +3: distance to right (4 bits, == encoding: 1,2,3,4+, root is 4+)
# -3: average of in-between tokens to the right (dim:wvec)
# +-4: word/context half of vector (dim:wvec/2, only valid for token encoding, assumes first half=word, second half=context)
# +-5: right/left child count (4 bits, >= encoding: >=1, >=2, >=3, >=4)
# +-6: right/left child count (4 bits, <= encoding: <=0, <=1, <=2, <=3)
# +7: distance to right, >= encoding (8 bits, >= encoding: >=2, >=3, >=4, >=6, >=8, >=12, >=16, >=20)
# -7: distance to right, <= encoding (8 bits, <= encoding: <=1, <=2, <=3, <=4, <=6, <=8, <=12, <=16)
# +-8: average of in-between word/context vectors to the right (dim:wvec/2)
# +-9: head exists/doesn't (one bit)



# 0.0418668 archybrid_conll07EnglishToken_wikipedia2MUNK-100_rbf376678_1014_cache.mat (trn/tst) (rbf-gamma:0.376747095368119)
fv022b = Int8[-3 0 8;-2 -1 -4;-2 -1 5;-2 0 -9;-2 0 -4;-2 0 4;-2 0 7;-2 1 1;-1 -1 -1;-1 -1 4;-1 0 -9;-1 0 -4;-1 0 4;-1 0 6;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4]

# 0.0325506937033084 archybrid_conllWSJToken_wikipedia2MUNK-100_rbf333140_cache.mat (rbf-gamma:0.333140404105682)
fv021a = Int8[-3 0 4;-3 1 -2;-2 0 4;-2 0 6;-2 1 -2;-1 -1 -4;-1 -1 2;-1 -1 4;-1 0 -4;-1 0 -1;-1 0 2;-1 0 4;-1 1 2;-1 1 4;0 -1 -4;0 -1 -1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4]

# 0.0443168273919168 archybrid_conll07EnglishToken_wikipedia2MUNK-100_rbf372064_cache.mat (rbf-gamma:0.372063662109375)
fv019 = Int8[-3 0 8;-2 -1 -4;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 6;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4]

# 0.03492252681764 archybrid_conllWSJToken_wikipedia2MUNK-100_rbf339506_cache.mat (rbf-gamma:0.339506144311523)
fv022a = Int8[-3 0 4;-3 1 -2;-2 0 -4;-2 0 4;-2 0 6;-2 1 -2;-1 -1 -4;-1 -1 2;-1 -1 4;-1 0 -4;-1 0 -1;-1 0 2;-1 0 4;-1 1 2;-1 1 4;0 -1 -4;0 -1 -1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4]

# 0.0444685231336006 archybrid13_conll07EnglishToken_wikipedia2MUNK-100_rbf3.720637e+05_cache.mat
fv023a = Int8[-3 0 8;-2 -1 -4;-2 -1 -1;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-2 1 4;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 5;-1 0 6;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4]

# BUGGY: 0.0325637 Best with rbf=0.339506144311523, archybrid_conllWSJToken_wikipedia2MUNK-100_fv130_dump.mat.
fv023 = Int8[-3 0 4;-3 1 -2;-2 -1 -4;-2 0 -4;-2 0 4;-2 0 6;-2 1 -2;-1 -1 -4;-1 -1 2;-1 -1 4;-1 0 -4;-1 0 -1;-1 0 2;-1 0 4;-1 1 2;-1 1 4;0 -1 -4;0 -1 -1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4]

# BUGGY: 0.0443493 Best with run_featselect_rbf('archybrid13', 'conll07EnglishToken_wikipedia2MUNK-100', 'fv136', 'fv021', 'rbf3721');
fv022 = Int8[-3 0 8;-2 -1 -1;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-2 1 4;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 6;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 -1 5;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4]

# 0.0445990 Best with rbf372111 archybrid_conll07EnglishToken_wikipedia2MUNK-100_fv136_dump.mat
fv021 = Int8[-3 0 8;-2 -1 -1;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 5;-1 0 6;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4]

# Best on run_featselect22('archybrid', 'conllWSJToken_wikipedia2MUNK-100', 'fv130', 'fv018');
fv017a = Int8[-3 0 -4;-3 0 4;-2 0 -4;-2 0 4;-2 1 -2;-1 -1 -4;-1 -1 4;-1 0 -4;-1 0 4;-1 1 -2;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4]

# This one does best on archybrid_conll07EnglishToken_wikipedia2MUNK-100_d5 (degree=5 kernel)
fv018a = Int8[-3 0 8;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-1 -1 1;-1 -1 4;-1 0 -4;-1 0 4;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4]

# This one does best on archybrid_conll07EnglishToken_wikipedia2MUNK-100 (degree=3 kernel)
fv015a = Int8[-3 0 8;-2 0 -4;-2 0 4;-2 1 1;-1 -1 4;-1 0 -4;-1 0 4;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4]

# This one is a local minimum on archybrid_conll07EnglishToken_wikipedia2MUNK-100 (degree=3 kernel)
fv031a = Int8[-3 0 -8;-3 0 -4;-3 0 -1;-3 0 8;-3 1 -4;-3 1 4;-2 -1 -4;-2 -1 4;-2 0 -8;-2 0 -4;-2 0 4;-2 0 8;-2 1 1;-2 1 2;-1 -1 -4;-1 -1 1;-1 -1 4;-1 0 -8;-1 0 -4;-1 0 2;-1 0 4;-1 0 8;-1 1 -4;-1 1 4;0 -1 -4;0 -1 1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4]

fv034 = zeros(Int8, 0, 3)
for a=-3:1
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   # no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   # no ldeps for buffer words other than n0
    if ((a <= -2) && (b < 0)) continue; end  # not interested in ldeps beyond s0
    if ((a <= -3) && (b > 0)) continue; end  # not interested in rdeps beyond s1
    for c=-9:9
      if ((c == 0) || (c == -3)) continue; end  # do not use the word+context combination, keep them separate
      if (abs(c) == 5 || abs(c) == 6) continue; end  # just use = encoding for child count
      if ((abs(c) == 2) && (b != 0)) continue; end # not interested in grand children
      if (abs(c) == 1) continue; end # existence not useful
      if (abs(c) == 9) continue; end # head not useful in archybrid
      if ((c == 3) || (c == -7)) continue; end  # just use >= encoding for distance
      if ((a > 0) && !in(c, [1,-1,4,-4])) continue; end      # no deps/dist/in-between/head for a>0
      if ((a == 0) && (b == 0) && !in(c, [1,-1,4,-4,-2,-5,-6])) continue; end  # no rdeps/dist/in-between/head for a=0
      if ((b != 0) && in(c, [3,-3,7,-7,8,-8,9,-9])) continue; end  # no dist/in-between/head for deps
      fv034 = [fv034; Int8[a b c]]
    end
  end
end



fv039 = Int8[
0 0 4; 		# [-196]    'n0w'       
-1 0 4; 	# [-119]    's0w'       
1 0 4; 		# [ -73]    'n1w'       
-2 0 4; 	# [ -43]    's1w'       
0 -1 4; 	# [ -20]    'n0l1w'     
0 0 -4; 	# [ -20]    'n0c'       
1 0 -4; 	# [ -19]    'n1c'       
-1 0 -4; 	# [ -16]    's0c'       
-1 -1 4; 	# [ -11]    's0l1w'     
-2 0 -4; 	# [ -10]    's1c'       
-3 -1 -4;     	# [  -8]    's2l1c'     
-1 -1 -4;     	# [  -8]    's0l1c'     
0 -1 -4; 	# [  -8]    'n0l1c'     
-3 -1 2; 	# [  -7]    's2l1r='    
-3 0 8; 	# [  -7]    's2aw'      
-2 1 1; 	# [  -7]    's1r1+'     
-1 0 -8; 	# [  -7]    's0ac'      
-1 0 7; 	# [  -7]    's0d>'      
-1 -1 1; 	# [  -6]    's0l1+'     
-1 0 8; 	# [  -6]    's0aw'      
-3 0 -4; 	# [  -5]    's2c'       
-3 0 4; 	# [  -5]    's2w'       
-2 -1 -4;     	# [  -5]    's1l1c'     
-1 1 4; 	# [  -5]    's0r1w'     
-3 0 -8; 	# [  -4]    's2ac'      
0 -1 1; 	# [  -4]    'n0l1+'     
-3 0 -1; 	# [  -3]    's2-'       
-3 0 7; 	# [  -3]    's2d>'      
-3 1 4; 	# [  -3]    's2r1w'     
-2 -1 4; 	# [  -3]    's1l1w'     
-2 0 8; 	# [  -3]    's1aw'      
-2 1 2; 	# [  -3]    's1r1r='    
-2 -1 -2;     	# [  -2]    's1l1l='    
-1 -1 -2;     	# [  -2]    's0l1l='    
-2 0 -8; 	# [  -1]    's1ac'      
-1 -1 2; 	# [  -1]    's0l1r='    
-1 1 -4; 	# [  -1]    's0r1c'     
-3 1 -4; 	# [   0]    's2r1c'     
-1 0 2; 	# [   0]    's0r='      
]


fv012 = Int8[
0 0 4; 		# [-203]    n0w       
-1 0 4; 	# [-125]    s0w       
-2 0 4; 	# [ -59]    s1w       
1 0 4; 		# [ -59]    n1w       
0 -1 4; 	# [ -41]    n0l1w     
-1 -1 4;	# [ -21]    s0l1w     

0 0 -4; 	# [ -17]    n0c       
-1 0 -4;	# [ -16]    s0c       
-2 0 -4;	# [ -13]    s1c       
1 0 -4; 	# [ -15]    n1c       
0 -1 -4; 	# [ -11]    n0l1c     
-1 -1 -4;	# [ -12]    s0l1c     
]


fv017 = Int8[       # 0.0476108, 'archybrid', 'conll07EnglishToken_wikipedia2MUNK-100'
0 0 4; 		# [-203]    n0w       
-1 0 4; 	# [-125]    s0w       
-2 0 4; 	# [ -59]    s1w       
1 0 4; 		# [ -59]    n1w       
0 -1 4; 	# [ -41]    n0l1w     
-2 1 -2;	# [ -30]    s1r1l=    
-1 -1 4;	# [ -21]    s0l1w     
-1 1 4; 	# [ -21]    s0r1w     
-1 1 -2;	# [ -19]    s0r1l=    
0 0 -4; 	# [ -17]    n0c       
-1 0 -4;	# [ -16]    s0c       
1 0 -4; 	# [ -15]    n1c       
-2 0 -4;	# [ -13]    s1c       
-3 0 4; 	# [ -12]    s2w       
-1 -1 -4;	# [ -12]    s0l1c     
-1 1 -4; 	# [ -12]    s0r1c     
0 -1 -4; 	# [ -11]    n0l1c     
]


fv018 = Int8[       # another version of fv808 that splits token features to word + context
    0 0 4;
    0 0 -4;
    -1 0 4;
    -1 0 -4;
    -2 0 4;
    -2 0 -4;
    1 0 4;
    1 0 -4;
    0 -1 4;
    0 -1 -4;
    -1 1 4;
    -1 1 -4;
    -2 1 -2;
    -1 -1 4;
    -1 -1 -4;
    -1 1 -2;
    -3 0 4;
    -3 0 -4;
]


fv084 = zeros(Int8, 0, 3)
for a=-3:1
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   # no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   # no ldeps for buffer words other than n0
    for c=-9:9
      if ((c == 0) || (c == -3)) continue; end  # do not use the word+context combination, keep them separate
      if (abs(c) == 5 || abs(c) == 6) continue; end  # just use = encoding for child count
      if ((c == 3) || (c == -7)) continue; end  # just use >= encoding for distance
      if ((a > 0) && !in(c, [1,-1,4,-4])) continue; end      # no deps/dist/in-between/head for a>0
      if ((a == 0) && (b == 0) && !in(c, [1,-1,4,-4,-2,-5,-6])) continue; end  # no rdeps/dist/in-between/head for a=0
      if ((b != 0) && in(c, [3,-3,7,-7,8,-8,9,-9])) continue; end  # no dist/in-between/head for deps
      fv084 = [fv084; Int8[a b c]]
    end
  end
end


# Better initial starting point
fv008w = Int8[
    0  0 4; # n0w
    -1 0 4; # s0w
    1  0 4; # n1w
    -2 0 4; # s1w
    0 -1 4; # n0l1w
    -1 1 4; # s0r1w
    -2 1 4; # s1r1w
    -1 -1 4; # s0l1w
         ]

# Good initial starting point
fv008 = Int8[
    0 0 4;      # n0
    -1 0 4;     # s0
    1 0 4;      # n1
    -2 0 4;     # s1
    0 -1 4;     # n0l1
    -1 1 4;     # s0r1
    0 0 -4;     # n0c
    -1 0 -4;	# s0c
]

# Include the head feature for stack words:
fv136 = zeros(Int8, 0, 3)
for a=-3:2
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   # no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   # no ldeps for buffer words other than n0
    for c=-9:9
      if ((c == 0) || (c == -3)) continue; end  # do not use the word+context combination, keep them separate
      if ((a > 0) && !in(c, [1,-1,4,-4])) continue; end      # no deps/dist/in-between/head for a>0
      if ((a == 0) && (b == 0) && !in(c, [1,-1,4,-4,-2,-5,-6])) continue; end  # no rdeps/dist/in-between/head for a=0
      if ((b != 0) && in(c, [3,-3,7,-7,8,-8,9,-9])) continue; end  # no dist/in-between/head for deps
      fv136 = [fv136; Int8[a b c]]
    end
  end
end

# All legal features for s2..n2:
fv130 = zeros(Int8, 0, 3)
for a=-3:2
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   # no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   # no ldeps for a>0
    for c=-8:8
      if ((a > 0) && in(c, [2,-2,3,-3,5,-5,6,-6,7,-7,8,-8])) continue; end      # no deps/dist/in-between for a>0
      if ((a == 0) && (b == 0) && in(c, [2,3,-3,5,6,7,-7,8,-8])) continue; end  # no rdeps/dist/in-between for a=0
      if ((b != 0) && in(c, [3,-3,7,-7,8,-8])) continue; end  # no dist/in-between for deps
      if ((c == 0) || (c == -3)) continue; end  # do not use the word+context combination, keep them separate
      fv130 = [fv130; Int8[a b c]]
    end
  end
end

fv102 = zeros(Int8, 0, 3)
for a=-2:1
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   # no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   # no ldeps for a>0
    for c=-8:8
      if ((a > 0) && in(c, [2,-2,3,-3,5,-5,6,-6,7,-7,8,-8])) continue; end      # no deps/dist/in-between for a>0
      if ((a == 0) && (b == 0) && in(c, [2,3,-3,5,6,7,-7,8,-8])) continue; end  # no rdeps/dist/in-between for a=0
      if ((b != 0) && in(c, [3,-3,7,-7,8,-8])) continue; end  # no dist/in-between for deps
      fv102 = [fv102; Int8[a b c]]
    end
  end
end

# fv1768 = Int8[
# #n0            n1             n2             s0             s1             s2             s0l1           s0l2           s0r1           s0r2           s1l1           s1l2           s1r1           s1r2           n0l1           n0l2          n0s0 s0s1
# 0  0  0  0  0  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3 -3 -3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2  0  0  0  0  0  0  0  0  0  0 -1   -2;
# 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  0    0;
# 0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  3    3;
# ]';

fv1768 = 
Int8[0 0 0
     0 0 -2
#    0 0 2   # buffer words dont have right children
     0 0 1
     0 0 -1
     1 0 0
#    1 0 -2  # buffer words other than n0 do not have ldeps
#    1 0 2   # buffer words dont have right children
     1 0 1
     1 0 -1
     2 0 0
#    2 0 -2  # buffer words other than n0 do not have ldeps
#    2 0 2   # buffer words dont have right children
     2 0 1
     2 0 -1
     -1 0 0
     -1 0 -2
     -1 0 2
     -1 0 1
     -1 0 -1
     -2 0 0
     -2 0 -2
     -2 0 2
     -2 0 1
     -2 0 -1
     -3 0 0
     -3 0 -2
     -3 0 2
     -3 0 1
     -3 0 -1
     -1 -1 0
     -1 -1 -2
     -1 -1 2
     -1 -1 1
     -1 -1 -1
     -1 -2 0
     -1 -2 -2
     -1 -2 2
     -1 -2 1
     -1 -2 -1
     -1 1 0
     -1 1 -2
     -1 1 2
     -1 1 1
     -1 1 -1
     -1 2 0
     -1 2 -2
     -1 2 2
     -1 2 1
     -1 2 -1
     -2 -1 0
     -2 -1 -2
     -2 -1 2
     -2 -1 1
     -2 -1 -1
     -2 -2 0
     -2 -2 -2
     -2 -2 2
     -2 -2 1
     -2 -2 -1
     -2 1 0
     -2 1 -2
     -2 1 2
     -2 1 1
     -2 1 -1
     -2 2 0
     -2 2 -2
     -2 2 2
     -2 2 1
     -2 2 -1
     0 -1 0
     0 -1 -2
     0 -1 2
     0 -1 1
     0 -1 -1
     0 -2 0
     0 -2 -2
     0 -2 2
     0 -2 1
     0 -2 -1
     -1 0 3
     -2 0 3]

fv804 = Int8[
#n0 s0 s1 n1 n0l1 s0r1 s0l1 s1r1 s0r
  0 -1 -2  1  0   -1   -1   -2   -1;
  0  0  0  0 -1    1   -1    1    0;
  0  0  0  0  0    0    0    0    2;
]';

fv708 = Int8[
#n0 n0l1 n0l2 n1 s0 s0r s0r1 s0s1 s1
  0  0    0    1 -1 -1  -1   -2   -2;
  0 -1   -2    0  0  0   1    0    0;
  0  0    0    0  0  2   0    3    0;
]';

fv808 = Int8[
#n0 s0 s1 n1 n0l1 s0r1 s1r1l s0l1 s0r1l s2
  0 -1 -2  1  0   -1   -2    -1   -1    -3;
  0  0  0  0 -1    1    1    -1    1     0;
  0  0  0  0  0    0   -2     0   -2     0;
]';

end # module Flist
