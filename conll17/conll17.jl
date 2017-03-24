using KUparser

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

# Only changes the HEAD and DEPREL columns of the inputfile
function writeconllu(sentences, parses, inputfile)
    if length(sentences) != length(parses); error(); end
    v = sentences[1].vocab
    s = p = nothing
    ns = nw = nl = 0
    for line in eachline(inputfile)
        nl += 1
        if ismatch(r"^\d+\t", line)
            # info("$nl word")
            if s == nothing
                s = sentences[ns+1]
                p = parses[ns+1]
            end
            f = split(line, '\t')
            nw += 1
            if f[1] != "$nw"; error(); end
            if f[2] != v.words[s.word[nw]]; error(); end
            f[7] = string(p.head[nw])
            f[8] = v.deprels[p.deprel[nw]]
            print(join(f, "\t"))
        else
            if line == "\n"
                # info("$nl blank")
                if s == nothing; error(); end
                if nw != length(s.word); error(); end
                ns += 1; nw = 0
                s = p = nothing
            else
                # info("$nl non-word")
            end
            print(line)
        end
    end
    if ns != length(sentences); error(); end
end

function readconllu(filename::AbstractString, vocab=nothing)
    if vocab == nothing
        vocab = Vocab()
        vocab.postags = UPOSTAG
        vocab.deprels = UDEPREL
    end
    words = Dict{String,WordId}()
    for id in 1:length(vocab.words)
        words[vocab.words[id]] = id
    end
    info(filename)
    sentences = Sentence[]
    s = Sentence(); s.vocab = vocab
    nw = 0
    for line in eachline(filename)
        line = chomp(line)
        fields = split(line, '\t') # TODO: this makes fields[2] a substring preventing garbage collection.
        if line == ""
            if nw > 0
                nw = 0
                push!(sentences,s)
                s = Sentence(); s.vocab = vocab
            end
        elseif line[1] == '#'
            continue
        elseif ismatch(r"^\d+$", fields[1])
            nw += 1
            if fields[1] != "$nw"; error("Token out of order in [$line]"); end

            if !haskey(words, fields[2])
                id = 1 + length(words)
                wform = String(fields[2])
                words[wform] = id
                push!(vocab.words, wform)
            end
            push!(s.word, words[fields[2]])

            if fields[4]=="_"
                pt = 0
            else
                pt = findfirst(vocab.postags, fields[4])
                if pt == 0; Base.warn_once("Bad postag: $(fields[4])"); end
            end
            push!(s.postag, pt)

            if fields[7]=="_"
                hd = 0
            else
                hd = parse(Position,fields[7])
                if !(0 <= hd <= typemax(eltype(s.head))); Base.warn_once("Bad head: $hd"); hd=0; end
            end
            push!(s.head, hd)

            if fields[8]=="_"
                dr = 0
            else
                dr = fields[8]
                # Get rid of language specific extensions
                drcolon = search(dr, ':')
                if drcolon > 0; dr = dr[1:drcolon-1]; end
                dr = findfirst(vocab.deprels, dr)
                if dr == 0; Base.warn_once("Bad deprel: $(fields[8])"); end
            end
            push!(s.deprel, dr)

        elseif ismatch(r"^\d+[-.]\d+$", fields[1])
            # skip
        else
            error("Cannot parse [$line]")
        end # if line == ""
    end # for line in eachline(filename)
    return sentences
end

function readvectors(filename::AbstractString, vocab::Vocab)
    lcwords = Dict{String,Vector{WordId}}()
    for id in 1:length(vocab.words)
        lcword = lowercase(vocab.words[id])
        if !haskey(lcwords, lcword); lcwords[lcword] = WordId[]; end
        push!(lcwords[lcword], id)
    end
    info(filename)
    open(`xzcat $filename`) do io
        firstline = split(readline(io)) # first line gives count and dim
        wdims = parse(Int,firstline[2])
        # TODO: We are leaving unknown word vectors as zero!
        vocab.wvecs = zeros(WVtype, wdims, length(vocab.words))
        for line in eachline(io)
            n = search(line, ' ')
            w = SubString(line,1,n-1)
            if !haskey(lcwords,w); continue; end
            a = map(s->parse(Float32,s), split(SubString(line,n+1)))
            if wdims != length(a); throw(DimensionMismatch()); end
            for i in lcwords[w]
                vocab.wvecs[:,i] = a
            end
        end
    end
end

# Universal POS tags (17)
const UPOSTAG = String[ # Dict{String,UInt8}(
"ADJ"   ,# => 1, # adjective
"ADP"   ,# => 2, # adposition
"ADV"   ,# => 3, # adverb
"AUX"   ,# => 4, # auxiliary
"CCONJ" ,# => 5, # coordinating conjunction
"DET"   ,# => 6, # determiner
"INTJ"  ,# => 7, # interjection
"NOUN"  ,# => 8, # noun
"NUM"   ,# => 9, # numeral
"PART"  ,# => 10, # particle
"PRON"  ,# => 11, # pronoun
"PROPN" ,# => 12, # proper noun
"PUNCT" ,# => 13, # punctuation
"SCONJ" ,# => 14, # subordinating conjunction
"SYM"   ,# => 15, # symbol
"VERB"  ,# => 16, # verb
"X"     ,# => 17, # other
]

# Universal Dependency Relations (37)
const UDEPREL = String[ # Dict{String,UInt8}(
"root"       ,# => 1, # root
"acl"        ,# => 2, # clausal modifier of noun (adjectival clause)
"advcl"      ,# => 3, # adverbial clause modifier
"advmod"     ,# => 4, # adverbial modifier
"amod"       ,# => 5, # adjectival modifier
"appos"      ,# => 6, # appositional modifier
"aux"        ,# => 7, # auxiliary
"case"       ,# => 8, # case marking
"cc"         ,# => 9, # coordinating conjunction
"ccomp"      ,# => 10,# clausal complement
"clf"        ,# => 11, # classifier
"compound"   ,# => 12, # compound
"conj"       ,# => 13, # conjunct
"cop"        ,# => 14, # copula
"csubj"      ,# => 15, # clausal subject
"dep"        ,# => 16, # unspecified dependency
"det"        ,# => 17, # determiner
"discourse"  ,# => 18, # discourse element
"dislocated" ,# => 19, # dislocated elements
"expl"       ,# => 20, # expletive
"fixed"      ,# => 21, # fixed multiword expression
"flat"       ,# => 22, # flat multiword expression
"goeswith"   ,# => 23, # goes with
"iobj"       ,# => 24, # indirect object
"list"       ,# => 25, # list
"mark"       ,# => 26, # marker
"nmod"       ,# => 27, # nominal modifier
"nsubj"      ,# => 28, # nominal subject
"nummod"     ,# => 29, # numeric modifier
"obj"        ,# => 30, # object
"obl"        ,# => 31, # oblique nominal
"orphan"     ,# => 32, # orphan
"parataxis"  ,# => 33, # parataxis
"punct"      ,# => 34, # punctuation
"reparandum" ,# => 35, # overridden disfluency
"vocative"   ,# => 36, # vocative
"xcomp"      ,# => 37, # open clausal complement
]

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

# conll17 word vectors
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Ancient_Greek/grc.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Arabic/ar.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Basque/eu.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Bulgarian/bg.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Catalan/ca.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/ChineseT/zh.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Croatian/hr.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Czech/cs.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Danish/da.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Dutch/nl.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/English/en.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Estonian/et.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Finnish/fi.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/French/fr.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Galician/gl.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/German/de.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Greek/el.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Hebrew/he.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Hindi/hi.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Hungarian/hu.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Indonesian/id.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Irish/ga.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Italian/it.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Japanese/ja.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Kazakh/kk.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Korean/ko.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Latin/la.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Latvian/lv.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Norwegian-Bokmaal/no_bokmaal.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Norwegian-Nynorsk/no_nynorsk.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Old_Church_Slavonic/cu.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Persian/fa.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Polish/pl.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Portuguese/pt.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Romanian/ro.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Russian/ru.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Slovak/sk.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Slovenian/sl.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Spanish/es.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Swedish/sv.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Turkish/tr.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Ukrainian/uk.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Urdu/ur.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Uyghur/ug.vectors.xz
# /mnt/ai/data/nlp/conll17/word-embeddings-conll17/Vietnamese/vi.vectors.xz

nothing
