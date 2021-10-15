module TensorSpace

using Random #calling random numbers

function rand_mat(v,F) #Gives a real or complex l-valence tensor, expressed as an array
    #if typeof(v) # I want a check to mmake sure I get an array of positive integers
    if F==="C" #Complex
        T=rand(ComplexF64,v) #gives an array with valence size(v) and and dimensions equal to the values of v
        return T
    
    elseif F==="R" #Real
        T=rand(Float64,v)
        return T
    
    else
        print("Please specify your field as R (for real) or C(for complex)")
    end
end# End of rand-mat

end# End of module