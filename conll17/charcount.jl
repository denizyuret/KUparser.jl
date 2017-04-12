chars = Set{Char}()
for line in eachline(STDIN)
    m = match(r"^\d+\t(.+?)\t", line)
    if m == nothing; continue; end
    w = m.captures[1]
    for c in w
        push!(chars, c)
    end
end
println(length(chars))
