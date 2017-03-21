typealias Position Int          # [1:nword]
typealias DepRel UInt8          # [1:ndeps]
typealias PosTag UInt8          # [1:npostag]
typealias Cost Int              # [0:nword]
typealias Move Int              # [1:nmove]
typealias SFtype Int32          # representing sparse feature in x
typealias WVtype Float32        # word vectors
typealias Pvec Vector{Position} # used for stack, head
typealias Dvec Vector{DepRel}   # used for deprel
Pzeros(n::Integer...)=zeros(Position, n...)
Dzeros(n::Integer...)=zeros(DepRel, n...)


