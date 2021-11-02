using GaloisFields
function galoisrandtensor(v::Tuple,p::Int64,d::Int64) #need to add option to change name of primitive element to another character
    G=@GaloisField! p^d α
    T=rand(G,v)
    return T
end#end of galoisrandtensor