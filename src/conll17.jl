# CONLL-U FORMAT:
# 1. ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
# 2. FORM: Word form or punctuation symbol.
# 3. LEMMA: Lemma or stem of word form.
# 4. UPOSTAG: Universal part-of-speech tag.
# 5. XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
# 6. FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
# 7. HEAD: Head of the current word, which is either a value of ID or zero (0).
# 8. DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
# 9. DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
# 10. MISC: Any other annotation.

function readconllu(filename::AbstractString)
    c = Corpus2()
    c.postags = UPOSTAG
    c.deprels = UDEPREL
    c.sentences = Sentence[]
    s = Sentence()
    nw = 0
    for line in eachline(filename)
        line = chomp(line)
        fields = split(line, '\t') # TODO: this makes fields[2] a substring preventing garbage collection.
        if line == ""
            if nw > 0
                s.wvec = Array(Float32,0,nw)
                push!(c.sentences,s)
                s = Sentence(); nw = 0
            end
        elseif line[1] == '#'
            continue
        elseif ismatch(r"^\d+$", fields[1])
            nw += 1
            if fields[1] != "$nw"; error("Token out of order in [$line]"); end
            push!(s.form, fields[2])
            hd = parse(Int,fields[7])
            if !(0 <= hd <= typemax(eltype(s.head))); Base.warn_once("Bad head: $hd"); hd=0; end
            push!(s.head, hd)
            pt = get(c.postags, fields[4], -1)
            if pt == -1; Base.warn_once("Bad postag: $(fields[4])"); end
            push!(s.postag, pt)
            dr = fields[8]
            # Get rid of language specific extensions
            drcolon = search(dr, ':')
            if drcolon > 0; dr = dr[1:drcolon-1]; end
            dr = get(c.deprels, dr, -1)
            if dr == -1; Base.warn_once("Bad deprel: $(fields[8])"); end
            push!(s.deprel, dr)
        elseif ismatch(r"^\d+[-.]\d+$", fields[1])
            # skip
        else
            error("Cannot parse [$line]")
        end
    end
    return c
end

# Universal POS tags (17)
const UPOSTAG = Dict{String,UInt8}(
"ADJ"   => 1, # adjective
"ADP"   => 2, # adposition
"ADV"   => 3, # adverb
"AUX"   => 4, # auxiliary
"CCONJ" => 5, # coordinating conjunction
"DET"   => 6, # determiner
"INTJ"  => 7, # interjection
"NOUN"  => 8, # noun
"NUM"   => 9, # numeral
"PART"  => 10, # particle
"PRON"  => 11, # pronoun
"PROPN" => 12, # proper noun
"PUNCT" => 13, # punctuation
"SCONJ" => 14, # subordinating conjunction
"SYM"   => 15, # symbol
"VERB"  => 16, # verb
"X"     => 17, # other
)

# Universal Dependency Relations (37)
const UDEPREL = Dict{String,UInt8}(
"root"       => 0, # root
"acl"        => 1, # clausal modifier of noun (adjectival clause)
"advcl"      => 2, # adverbial clause modifier
"advmod"     => 3, # adverbial modifier
"amod"       => 4, # adjectival modifier
"appos"      => 5, # appositional modifier
"aux"        => 6, # auxiliary
"case"       => 7, # case marking
"cc"         => 8, # coordinating conjunction
"ccomp"      => 9, # clausal complement
"clf"        => 10, # classifier
"compound"   => 11, # compound
"conj"       => 12, # conjunct
"cop"        => 13, # copula
"csubj"      => 14, # clausal subject
"dep"        => 15, # unspecified dependency
"det"        => 16, # determiner
"discourse"  => 17, # discourse element
"dislocated" => 18, # dislocated elements
"expl"       => 19, # expletive
"fixed"      => 20, # fixed multiword expression
"flat"       => 21, # flat multiword expression
"goeswith"   => 22, # goes with
"iobj"       => 23, # indirect object
"list"       => 24, # list
"mark"       => 25, # marker
"nmod"       => 26, # nominal modifier
"nsubj"      => 27, # nominal subject
"nummod"     => 28, # numeric modifier
"obj"        => 29, # object
"obl"        => 30, # oblique nominal
"orphan"     => 31, # orphan
"parataxis"  => 32, # parataxis
"punct"      => 33, # punctuation
"reparandum" => 34, # overridden disfluency
"vocative"   => 35, # vocative
"xcomp"      => 36, # open clausal complement
)

# Universal Features (21) with number of possible values
# Abbr: abbreviation (1) Yes (all features also have None as an extra choice)
# Animacy: animacy (4) Anim,Hum,Inan,Nhum
# Aspect: aspect (6) Hab,Imp,Iter,Perf,Prog,Prosp
# Case: case (32)
# Definite: definiteness or state (5) Com,Cons,Def,Ind,Spec
# Degree: degree of comparison (5) Abs,Cmp,Equ,Pos,Sup
# Evident: evidentiality (2) Fh,Nfh
# Foreign: is this a foreign word? (1) Yes
# Gender: gender (4) Com,Fem,Masc,Neut
# Mood: mood (12) Adm,Cnd,Des,Imp,Ind,Jus,Nec,Opt,Pot,Prp,Qot,Sub
# NumType: numeral type (7) Card,Dist,Frac,Mult,Ord,Range,Sets
# Number: number (10) CollCountDualGrpaGrplInvPaucPlurPtanSingTri
# Person: person (5) 0,1,2,3,4
# Polarity: polarity (2) Neg,Pos
# Polite: politeness (4) ElevFormHumbInfm
# Poss: possessive (1) Yes
# PronType: pronominal type (11) ArtDemEmpExcIndIntNegPrsRcpRelTot
# Reflex: reflexive (1) Yes
# Tense: tense (5) FutImpPastPqpPres
# VerbForm: form of verb or deverbative (8) ConvFinGdvGerInfPartSupVnoun
# Voice: voice (8) ActAntipCauDirInvMidPassRcp

nothing
