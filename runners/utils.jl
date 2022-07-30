using POMDPs, Crux, Flux, BSON

include("../src/policy_gradient.jl")
include("../src/value_based.jl")
include("../src/mis_policy.jl")

function pretrain_value(mdp, P; target, Ndata=10000, Nepochs=100, batchsize=1024, value_args=(D)->D[:s])
	(π) -> begin
		# Sample a bunch of data
		D = steps!(Sampler(mdp, P), explore=true, Nsteps=Ndata)
		
		# Put it into a flux dataloader
		v = value(trainable_policies(π)[1], value_args(D))
		v .= target
		d = Flux.Data.DataLoader((value_args(D), v), batchsize=batchsize)

		for pol in trainable_policies(π)
			# Use maximum likelihood estimation
			loss(x,y) = Flux.mse(pol(x), y)

			# Train
			Flux.@epochs Nepochs Flux.train!(loss, Flux.params(pol), d, Adam())
		end
	end
end

function pretrain_policy(mdp, P; Ndata=10000, Nepochs=100, batchsize=1024)
	(π) -> begin
		# Sample a bunch of data
		D = steps!(Sampler(mdp, P), explore=true, Nsteps=Ndata)
		
		# Put it into a flux dataloader
		d = Flux.Data.DataLoader((D[:s], D[:a]), batchsize=batchsize)

		for pol in trainable_policies(π)
			if actor(pol) isa MixtureNetwork
				N = length(actor(pol).networks)
				αtarget = 1f0 / N
				loss_mixture(x,y) = -mean(logpdf(pol, x, y)) + Flux.mse(actor(pol).weights(x), αtarget)
				Flux.@epochs Nepochs Flux.train!(loss_mixture, Flux.params(pol), d, Adam())			
			else
				loss(x,y) = -mean(logpdf(pol, x, y))
				Flux.@epochs Nepochs Flux.train!(loss, Flux.params(pol), d, Adam())			
			end
		end
	end
end

function pretrain_AV(mdp, P; v_target, kwargs...)
	vtrain = pretrain_value(mdp, P; target=v_target, kwargs...)
	poltrain = pretrain_policy(mdp, P; kwargs...)
	(π) -> begin
		vtrain(π)
		poltrain(π)
	end
end

function pretrain_AQ(mdp, P; v_target, kwargs...)
	vtrain = pretrain_value(mdp, P; target=v_target, value_args=(D)->vcat(D[:s], D[:a]), kwargs...)
	poltrain = pretrain_policy(mdp, P; kwargs...)
	(π) -> begin
		vtrain(π)
		poltrain(π)
	end
end


function experiment_setup(;mdp, Ntrials, dir, plot_init=()->plot())
	try mkdir(dir) catch end
	
	(𝒮fn, name=nothing) -> begin 
		data = Dict(:est => [], :fs => [], :ws =>[])
		failures = 0
		successes = 0
		while true
			S = 𝒮fn()
			try
				fs, ws = solve(S, mdp)
			    push!(data[:est], cumsum(fs .* ws) ./ (1:length(fs)))
			    push!(data[:fs], fs)
			    push!(data[:ws], ws)
				successes += 1
			catch e
				println(e)
				failures += 1
				!isnothing(name) && BSON.@save "$dir/$(name)_failure_$(failures)_solver.bson" S
			end
			if successes >= Ntrials || failures >= Ntrials
				break
			end
		end
		
		if successes > 0 && !isnothing(name)
			BSON.@save "$dir/$name.bson" data
			Neps = length(data[:fs][1])
			plot_init()
			plot!(1:Neps, mean(data[:est]), ribbon=std(data[:est]), label=name)
			savefig("$dir/$name.png")
		end
		return data
	end
end

