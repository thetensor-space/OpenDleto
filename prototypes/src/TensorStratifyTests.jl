########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################



using Dates

include("TensorRandomize.jl")
include("TensorStratify.jl")


using Pkg
Pkg.add("JLD")
using JLD


function printTimings(T)
	s=""
	for i in keys(T)
		for j in keys(T[i])
			s *= string(i) * "\t" * string(j) * "\t" * string(T[i][j]) * "\n"
		end
	end
	return s
end


###############################################################
# test for stratification

function transformTest(t,filename;ratio=1000,rounds=30,verbose=false,control=false,noise=10,relsize=0.001,toprint=10,type="normal",stratifyname="stratify",stratifyfunc=stratify)
    print("Transformation Testing with tensor of size " * string(size(t)) * " type: "* stratifyname*"\n\n")
    date = replace(string(now()), ':' => '.')

	timings = Dict{String,Any}(
		"original"	=> Dict{String,Any}(),
		"control"	=> Dict{String,Any}(),
		"randomized"	=> Dict{String,Any}(),
		stratifyname	=> Dict{String,Any}()
	)
	
	dataToSave=Dict{String,Any}(
		"filename"	=>	filename, 
		"ratio"		=>	ratio,
		"rounnds"	=>	rounds,
		"conrol"	=>	control,
		"noise"		=>	noise,
		"relsize"	=>	relsize,
		"noisetype"	=>	type,
		"orig-tensor"	=> t,
		"timestamp"	=>	date
	)

    print("Working on " * filename * "    dir: " * date *"\n")

    mkdir(date)
    mkdir( date * "/data")
    mkdir( date * "/images")

    print("Saving original\n")
	savetime = @timed ( save( date * "/data/original.jld", "data", t) )
	timings["original"]["save"] = savetime.time

    if control
        print("Startifying original.\n")
        controltime = @timed(stratifyfunc(t;toprint,verbose))
		st, D = controltime.value
		dataToSave["control-tensor"]=st
		dataToSave["control-result"]=D
		dataToSave["control-matrices"]=D["matrices"]
		dataToSave["control-derivation"]=D["derivation"]
        print("Saving original stratification.\n" )
		savetime = @timed ([ save( date * "/data/original-strat.jld", "data", st), save( date * "/data/original-strat-singularvalues.jld", "data", D["singularValues"])])
		timings["control"] = D["timings"]
		timings["control"]["save"] = savetime.time		
    end

    print( "Randomizing original.\n")
	rt,rm = tensorRandomize(t, [rounds,rounds,rounds];type)
	dataToSave["randomized-tensor"]=rt
	dataToSave["randomized-matrices"]=rm

	rtnoise = tensorAddNoise(rt,noise;relsize,type)
	dataToSave["randomized-tensor-noise"]=rtnoise

	print( "Saving randomized version.\n")
	savetime = @timed ([ save( date * "/data/randomized.jld", "data", rt), save( date * "/data/randomizednoise.jld", "data", rtnoise) ])
	timings["randomized"]["save"] = savetime.time


    print( "Stratifying randomized version, this may take a while....\n")
	startifytime = @timed(stratifyfunc(rtnoise;toprint,verbose))
	srt, D = startifytime.value
	dataToSave[stratifyname * "-tensor"]=srt
	dataToSave[stratifyname * "-result"]=D
	dataToSave[stratifyname * "-matrices"]=D["matrices"]
	dataToSave[stratifyname * "-derivation"]=D["derivation"]
    print( "Saving stratification of randomized.\n")
 	savetime = @timed ( [ save( date * "/data/randomized-start.jld", "data", srt), save( date * "/data/randomized-strat-singularvalues.jld", "data", D["singularValues"]) ])
	timings[stratifyname] = D["timings"]
	timings[stratifyname]["save"] = savetime.time

	Xmat=rm[1]*D["matrices"][1]
	Ymat=rm[2]*D["matrices"][2]
	Zmat=rm[3]*D["matrices"][3]
	dataToSave["combined-matrices"]=[Xmat,Ymat,Zmat]
	save( date * "/data/comibedTransformationsX.jld", "data", Xmat)
	save( date * "/data/comibedTransformationsY.jld", "data", Ymat)
	save( date * "/data/comibedTransformationsZ.jld", "data", Zmat)

    if verbose
        print( "Product of the X matrices\n")
        display( round.(Xmat, digits=3) )
        print( "\n")

        print( "Product of the Y matrices\n")
        display( round.(Ymat, digits=3) )
        print( "\n")

        print( "Product of the Z matrices\n")
       display( round.(Zmat, digits=3) )
        print( "\n")
    end

    print( "Generating images\n")
	name = date * "/images/" * filename 
 	savetime = @timed ( save3D(t, name * "-org.ply", name * "-org.dat"  ,ratio; factor=1))
	timings["original"]["3Dprint"] = savetime.time 
	if control
		savetime = @timed (save3D(st, name * "-org-recons.ply", name * "-org-recons.dat",ratio; factor=2))
		timings["control"]["3Dprint"] = savetime.time 
	end 
	savetime = @timed ( save3D(rtnoise,  name * "-rand.ply", date * "/images/" * filename * "-rand.dat",ratio; factor=1))
	timings["randomized"]["3Dprint"] = savetime.time 
	savetime = @timed ( save3D(srt, name * "-rand-recons.ply", name * "-rand-recons.dat",ratio; factor=2))
	timings[stratifyname]["3Dprint"] = savetime.time 

	save( date * "/data/allData.jld", dataToSave)
    file = open(date * "/data/timings.json", "w")
    write(file, string(timings)*"\n")
    close(file)
    file = open(date * "/data/timings.txt", "w")
    write(file, printTimings(timings)*"\n")
    close(file)

	return timings
end


function stratificationTest(t,filename;ratio=1000,rounds=30,verbose=false,control=false,noise=10,relsize=0.001,toprint=10,type="normal")
	return transformTest(t,filename;ratio,rounds,verbose,control,noise,relsize,toprint,type,stratifyname="stratify",stratifyfunc=stratify)
end

function curvificationTest(t,filename;ratio=1000,rounds=30,verbose=false,control=false,noise=10,relsize=0.001,toprint=10,type="normal")
	return transformTest(t,filename;ratio,rounds,verbose,control,noise,relsize,toprint,type,stratifyname="curvify",stratifyfunc=curvify)
end

function stratification12FaceTest(t,filename;ratio=1000,rounds=30,verbose=false,control=false,noise=10,relsize=0.001,toprint=10,type="normal")
	return transformTest(t,filename;ratio,rounds,verbose,control,noise,relsize,toprint,type,stratifyname="adjoint12",stratifyfunc=stratify12face)
end

nothing

