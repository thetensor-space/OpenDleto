
function dot(x,y)
    a=size(x)[1]
    b=0
    for i = 1:a
        b += x[i]*y[i]
    end 
    return b
end

