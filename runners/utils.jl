using POMDPs, Crux, Flux, BSON

include("../src/policy_gradient.jl")
include("../src/value_based.jl")
include("../src/mis_policy.jl")

# This function ensures that a random NN starts with a distribution that is close the nominal
function policy_match_logits(P)
    logps = log.(Float32.(P.distribution.p))
    (π, s) -> softmax(log.(softplus.(value(π, s))) .+ logps)
end


function run_experiment(𝒮fn, mdp, Ntrials, gt, dir, name)
	try mkdir(dir) catch end
	data = Dict(:est => [], :fs => [], :ws =>[])
	for i=1:Ntrials
		fs, ws = solve(𝒮fn(), mdp)
	    push!(data[:est], cumsum(fs .* ws) ./ (1:length(fs)))
	    push!(data[:fs], fs)
	    push!(data[:ws], ws)
	end
	BSON.@save "$dir/$name.bson" data
	Neps = length(data[:fs][1])
	plot(1:Neps, x->gt, linestyle=:dash, color=:black)
	plot!(1:Neps, mean(data[:est]), ribbon=std(data[:est]), label=name)
	savefig("$dir/$name.pdf")
end

