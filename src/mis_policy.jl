trainable_indices(π::N) where N <: NetworkPolicy = [1]
all_policies(π::N) where N <: NetworkPolicy = [π]

## Multiple Importance Sampling distribution (Mixed per trajectory)
mutable struct MISPolicy <: NetworkPolicy
	distributions
	weights
	# Nsamps
	weight_style
	# i
	current_distribution
	MISPolicy(distributions, weights=ones(length(distributions)) ./ length(distributions); weight_style=:DM, current_distribution=1) = new(distributions, weights, weight_style, current_distribution)
	MISPolicy(distributions, weights, weight_style, current_distribution) = new(distributions, weights, weight_style, current_distribution)
end

function trainable_indices(π::MISPolicy)
	findall([d isa NetworkPolicy for d in π.distributions])
end

function all_policies(π::MISPolicy)
	π.distributions
end 

function Crux.new_ep_reset!(π::MISPolicy)
	# π.current_distribution = findfirst(round.(Int, cumsum(π.Nsamps)) .>= round(Int, π.i))
	# if isnothing(π.current_distribution)
	# 	println(π.Nsamps, π.i)
	# end
	# π.i = mod1(π.i+1, sum(π.Nsamps))
    π.current_distribution = rand(Categorical(π.weights))
end

Flux.@functor MISPolicy

Flux.trainable(π::MISPolicy) = Iterators.flatten([Flux.trainable(n) for n in π.distributions])

Crux.layers(π::MISPolicy) = Iterators.flatten([layers(n) for n in π.distributions])

function Crux.device(π::MISPolicy)
	d1 = device(π.distributions[1])
	@assert all(device.(π.distributions) .== [d1])
	d1
end

POMDPs.action(π::MISPolicy, s) = action(π.distributions[π.current_distribution], s)

Crux.exploration(π::MISPolicy, s; kwargs...) = exploration(π.distributions[π.current_distribution], s; kwargs...)

function Crux.action_space(π::MISPolicy)
	A1 = action_space(π.distributions[1])
	@assert all(action_space.(π.distributions) .== [A1])
	A1
end

function Crux.trajectory_pdf(π::MISPolicy, D...)
	# weights = π.Nsamps ./ sum(π.Nsamps)
	pdfs = [trajectory_pdf(d, D...) for d in π.distributions]
	
	return sum(π.weights .* pdfs)
end

