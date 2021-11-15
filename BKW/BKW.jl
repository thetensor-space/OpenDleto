########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################



using LinearAlgebra
using SparseArrays
using Dates
using Random
#using Basic.FileSystem

### Used to save data
## Reload using 
## load("mydata.jld")["data"]
using Pkg
Pkg.add("JLD")
using JLD

include( "Tensors.jl" )
include( "Utils.jl"   )
include( "Tensor3D.jl")
include( "Laplace.jl" )


###############################################################################
# Tests 
###############################################################################


function testSurface(d, width, control=false)
    date = now()
    path = "../data/" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))
    mkdir( path )
    mkdir( path * "/data")
    mkdir( path * "/images")

    print("Creating original\n")
    t = MartiniT(d,d,d,width)    
    print("Saving original\n")
    save( path * "/data/original.jld", "data", t)

    if control 
        print("Startifying original.\n")
        @time pass, nt, mats = stratify(t)
        print("Saving original stratification.\n" )
        save( path * "/data/original-strat.jld", "data", nt)
    end 

    print( "Randomizing original.\n")
    @time rt = sprandomize(t,50)
    print( "Saving randomized version.\n")
    save( path * "/data/randomized.jld", "data", rt)

    print( "Stratifying ....\n")
    @time pass, nrt, mats = stratify(rt)
    print( "Saving stratification ...\n")
    save( path * "/data/randomized-start.jld", "data", nrt)

    print( "Rending in 3D...")
    save3D( path * "/images/plot-"*string(d)*"-org.ply", matround(t,3))
    if control
        save3D( path * "/images/plot-"*string(d)*"-org-recons.ply", matround(nt,3))
    end 
    save3D( path * "/images/plot-"*string(d)*"-rand.ply", matround(rt,3))
    save3D( path * "/images/plot-"*string(d)*"-rand-recons.ply", matround(nrt,3))
    return pass
end


####################################
# Test split
####################################

function test12Split(d, control=false)
    date = now()
    path = "../data/" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))
    mkdir(path)
    mkdir( path * "/data")
    mkdir( path * "/images")

    
    print("Creating original\n")
    dims1 = randPartition(d)
    dims2 = randPartition(d)
    print("\tDimensions\t")
    print( dims1)
    print( " x ")
    print(dims2)
    print("\n")
    t = makeSplit12(dims1,dims2,d)
    print("Saving original\n")
    save( path * "/data/original.jld", "data", t)

    if control 
        print("Blocking original.\n")
        @time pass, nt, mats = block12(t)
        print("Saving original blocking.\n" )
        save( path * "/data/original-block-12.jld", "data", nt)
    end 


    print( "Randomizing original.\n")
    @time rt = sprandomize(t,50)
    print( "Saving randomized version.\n")
    save( path * "/data/randomized.jld", "data", rt)

    print( "Blocking 12-face ...\n")
    @time pass, nrt, mats = block12(rt)
    print( "Saving blocking ...\n")
    save( path * "/data/randomized-start.jld", "data", nrt)

    print( "Rending in 3D...")
    save3D( path * "/images/plot-"*string(d)*"-org.ply", matround(t,3))
    if control
        save3D( path * "/images/plot-"*string(d)*"-org-recons.ply", matround(nt,3))
    end 
    save3D( path * "/images/plot-"*string(d)*"-rand.ply", matround(rt,3))
    save3D( path * "/images/plot-"*string(d)*"-rand-recons.ply", matround(nrt,3))
    return pass
end


function test123Split(d, control=false)
    date = now()
    path = "../data/" * string(year(date)) * "-" * string(month(date)) * "-" * string(day(date)) * "-time-" * string(hour(date)) * "-" * string(minute(date)) * "-" * string(second(date))
    mkdir(path)
    mkdir( path * "/data")
    mkdir( path * "/images")

    
    print("Creating original\n")
    dims1 = randPartition(d)
    dims2 = randPartition(d)
    dims3 = randPartition(d)
    print("\tDimensions\t")
    print( dims1)
    print( " x ")
    print(dims2)
    print( " x ")
    print(dims3)
    print("\n")
    t = makeSplit123(dims1,dims2,dims3)
    print("Saving original\n")
    save( path * "/data/original.jld", "data", t)

    if control 
        print("Blocking original.\n")
        @time pass, nt, mats = block123(t)
        print("Saving original blocking.\n" )
        save( path * "/data/original-block-12.jld", "data", nt)
    end 


    print( "Randomizing original.\n")
    @time rt = sprandomize(t,50)
    print( "Saving randomized version.\n")
    save( path * "/data/randomized.jld", "data", rt)

    print( "Blocking 12-face ...\n")
    @time pass, nrt, mats = block123(rt)
    print( "Saving blocking ...\n")
    save( path * "/data/randomized-start.jld", "data", nrt)

    print( "Rending in 3D...")
    save3D( path * "/images/plot-"*string(d)*"-org.ply", matround(t,3))
    if control
        save3D( path * "/images/plot-"*string(d)*"-org-recons.ply", matround(nt,3))
    end 
    save3D( path * "/images/plot-"*string(d)*"-rand.ply", matround(rt,3))
    save3D( path * "/images/plot-"*string(d)*"-rand-recons.ply", matround(nrt,3))
    return pass
end
