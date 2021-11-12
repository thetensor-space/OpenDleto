
#############################################
## Examples.
#############################################

## Flat Genus 2 Indecomposable
function FlatGenus2(F,d)
    t = zeros(F, (2*d+1,2*d+1,2))
    for i = 1:d
        t[i,d+i,1] = 1;   t[d+i,i,1] = -1
        t[i,d+i+1,2] = 1; t[d+i+1,i,2] = -1
    end
    return t
end 

