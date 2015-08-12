pxyequal(a,b)=(isa(a,Vector) ? isequal(a,b) : (isequal(a[1],b[1]) && xyequal(a,b)))
xyequal(a,b)=myapprox(sortcols(vcat(hcat(a[2]...),hcat(a[3]...))), sortcols(vcat(hcat(b[2]...),hcat(b[3]...))))
function myapprox(x, y; 
                  maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                  rtol::Real=maxeps^(1/3), atol::Real=maxeps^(1/2))
    size(x) == size(y) || (warn("myapprox: $(size(x))!=$(size(y))"); return false)
    d = abs(x-y)
    s = abs(x)+abs(y)
    all(d .< (atol + rtol * s))
end

:ok
