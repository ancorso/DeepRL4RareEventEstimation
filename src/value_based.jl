include("evaluation.jl")

# Sample from a discrete value function according to the optimal distribution
function estimator_logits(P)
	(π, s) -> begin
	    vals = value(π, s)
	    probs = Float32.(Crux.logits(P, s))
	    ps = vals .* probs
		sm = sum(ps, dims=1)
	    zeros = sm[:] .== 0
		sm[zeros] .= 1f0
		ps[:, zeros] .= probs
	    ps ./ sm
	end
end

# Estimate the value function for a discrete network
function value_estimate(π::DiscreteNetwork, s, 𝒫)
	pdfs = Float32.(Crux.logits(𝒫[:px], s))
    sum(value(π, s) .* pdfs, dims=1)
end

# Estimate the value function for a Continuous network
function value_estimate(π::ActorCritic, s, 𝒫)
	samps = []
	for i=1:𝒫[:N_samples]
		a, logqa = exploration(π, s)
		w = exp.(logpdf(𝒫[:px], s, a) .- logqa)
		push!(samps, value(π, s, a) .* w)
	end
	return mean(samps)
end

function Ef_target(π, 𝒫, 𝒟; kwargs...)
	πs = trainable_policies(π)
	if length(πs) > 1
		ids = 𝒟[:id][:]
		all_values = vcat([value_estimate(d, 𝒟[:sp], 𝒫) for d in πs]...)
		values = sum(all_values .* Flux.onehotbatch(ids, collect(1:length(πs))), dims=1)
		
		target = reshape(𝒫[:f_target_train][ids], 1, :)
	else
		values = value_estimate(πs[1], 𝒟[:sp], 𝒫)
		target = 𝒫[:f_target_train][1]
	end
	values[𝒟[:done]] .= 0f0
	return 𝒟[:done] .* (𝒟[:r] .> target) .+ (1.f0 .- 𝒟[:done]) .* values
end

function actor_loss_continuous_is(π, 𝒫, 𝒟; info=Dict())
	πs = trainable_policies(π)
	if length(πs) > 1
		ids = 𝒟[:id][:]
		oh = Flux.onehotbatch(ids, collect(1:length(πs)))
		expl = [exploration(d, 𝒟[:s]) for d in πs]
		a = sum(vcat([a for (a, _) in expl]...) .* oh, dims=1)
		logqa = sum(vcat([logqa for (_, logqa) in expl]...) .* oh, dims=1)

	    f_s = ignore_derivatives() do 
	        sum(vcat([value_estimate(d, 𝒟[:s], 𝒫) for d in πs]...) .* oh, dims=1) .+ eps(Float32)
	    end
		f_sa = sum(vcat([value(d, 𝒟[:s], a) for d in πs]...) .* oh, dims=1) .+ eps(Float32)
	else
		a, logqa = exploration(πs[1], 𝒟[:s])
	    f_s = ignore_derivatives() do 
	    	value_estimate(πs[1], 𝒟[:s], 𝒫) .+ eps(Float32)
	    end
		f_sa = value(πs[1], 𝒟[:s], a) .+ eps(Float32)
	end
	qa = exp.(logqa)
	pa = exp.(logpdf(𝒫[:px], 𝒟[:s], a)) 
	
	ws = ignore_derivatives() do
		ws = 𝒟[:fwd_importance_weight] ./ 𝒟[:importance_weight]
		ws ./ mean(ws)
	end
	
	-mean(ws .* f_sa .* pa .* logqa ./ (qa .* f_s))
end

function td_loss_mis(;loss=Flux.mse, name=:Qavg, s_key=:s, a_key=:a, weight=nothing)
    (π, 𝒫, 𝒟, y; info=Dict()) -> begin
		πs = trainable_policies(π)
		if length(πs) > 1
			ids = 𝒟[:id][:]
			Qs = vcat([value(d, 𝒟[s_key], 𝒟[a_key]) for d in πs]...)
			Q = sum(Qs .* Flux.onehotbatch(ids, collect(1:length(πs))), dims=1)
		else
			Q = value(πs[1], 𝒟[s_key], 𝒟[a_key])
		end
		
        # Store useful information
        ignore_derivatives() do
            info[name] = mean(Q)
        end
        
        loss(Q, y, agg = isnothing(weight) ? mean : Crux.weighted_mean(𝒟[weight] ./ mean(𝒟[weight])))
    end
end

function ValueBasedIS(;agent::PolicyParams,
			  training_buffer_size,
			  buffer_size,
			  train_actor=false,
			  elite_frac = 0.1,
			  N_samples=5, # Number of samples for a value estimate of the a continuous policy
			  f_target,
			  ΔN,
              a_opt::NamedTuple=(;), 
              c_opt::NamedTuple=(;), 
              log::NamedTuple=(;), 
              required_columns=[],
			  name = "value_based_is",
              kwargs...)
    if train_actor
		a_opt=TrainingParams(;loss=actor_loss_continuous_is, name="actor_", a_opt...)
	else
		a_opt = nothing
	end
	
	f_target_train = [f_target]
	
	# If MIS, make sure we record an ID. 
	if agent.π isa MISPolicy
		push!(required_columns, :id)
		f_target_train = fill(f_target, length(trainable_policies(agent.π)))
	end
	
    EvaluationSolver(;agent=PolicyParams(agent, π⁻=deepcopy(agent.π)),
                    𝒫=(;px=agent.pa, f_target=[f_target], f_target_train, elite_frac, N_samples),
					ΔN=ΔN,
					training_buffer_size,
					buffer_size,
					training_type=:value,
                    log=LoggerParams(;dir = "log/$name", log...),
                    a_opt=a_opt,
					c_opt=TrainingParams(;loss=td_loss_mis(weight=:fwd_importance_weight), name="critic_", epochs=ΔN, c_opt...),
                    target_fn=Ef_target,
                    required_columns=unique([required_columns..., :return, :traj_importance_weight, :fwd_importance_weight, :importance_weight]),
					pre_train_callback=gradual_target_increase,
                    kwargs...)
end
        
