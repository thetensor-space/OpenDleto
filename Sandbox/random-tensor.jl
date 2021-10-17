module TensorSpace

using Random #calling random numbers

function rand_tensor(v,F,I=-10:10) #Gives a real or complex length(v)-valence tensor, expressed as an array
    #v is a Tuple of positive integers i.e. (1,2,3)
    # F is either the string 'R', 'C', or 'Z'
    #If the ring is the integers, I determines the integers that will populate the random tensor. writing Int will draw randomly from all integers.
    #if typeof(v)=Tuple # I want a check to make sure I get a Tuple of positive integers
    if F==='C' #Complex
        T=rand(ComplexF64,v)
        return T
    
    elseif F==='R' #Real
        T=rand(Float64,v)
        return T
    
    elseif F==='Z' #Integers
        T=rand(I,v)
        return T
    else
        print("Please specify your field as 'R' (for real) or 'C' (for complex)")
    end
end# End of rand-mat

end# End of module