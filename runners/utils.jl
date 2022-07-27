using POMDPs, Crux, Flux, BSON

include("../src/policy_gradient.jl")
include("../src/value_based.jl")
include("../src/mis_policy.jl")

function pretrain_value_discrete(mdp, P; target, Ndata=10000, Nepochs=100, batchsize=1024)
	(π) -> begin
		# Sample a bunch of data
		D = steps!(Sampler(mdp, P), explore=true, Nsteps=Ndata)
		
		# Put it into a flux dataloader
		v = value(trainable_policies(π)[1], D[:s])
		v .= target
		d = Flux.Data.DataLoader((D[:s], v), batchsize=batchsize)

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
			# Use maximum likelihood estimation
			loss(x,y) = -mean(logpdf(pol, x, y))

			# Train
			Flux.@epochs Nepochs Flux.train!(loss, Flux.params(pol), d, Adam())
		end
	end
end


function experiment_setup(;mdp, Ntrials, dir, plot_init=()->plot())
	try mkdir(dir) catch end
	
	(𝒮fn, name) -> begin 
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
				BSON.@save "$dir/$(name)_failure_$(failures)_solver.bson" S
			end
			if successes >= Ntrials || failures >= Ntrials
				break
			end
		end
		
		if successes > 0
			BSON.@save "$dir/$name.bson" data
			Neps = length(data[:fs][1])
			plot_init()
			plot!(1:Neps, mean(data[:est]), ribbon=std(data[:est]), label=name)
			savefig("$dir/$name.pdf")
		end
	end
end

