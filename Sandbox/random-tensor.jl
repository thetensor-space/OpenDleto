#module TensorSpace

using Random #calling random numbers

function randtensor(v,F='Z') #Gives a real or complex length(v)-valence tensor, expressed as an array
    #v is a Tuple of positive integers i.e. (1,2,3)
    # F is either the string 'R', 'C', or 'Z'
    #if typeof(v)=Tuple # I want a check to make sure I get a Tuple of positive integers
    if F==='C' #Complex
        T=rand(ComplexF8,v)
        return T
    
    elseif F==='R' #Real
        T=rand(Float8,v)
        return T
    
    elseif F==='Z' #Integers
        T=rand(Int8,v)
        return T
    else
        print("Please specify your field as 'R' (for real), 'C' (for complex), or 'Z' (for integers).")
    end
end# End of rand-mat

#end# End of module