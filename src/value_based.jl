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
function value_estimate(π::DiscreteNetwork, px, s)
	pdfs = Float32.(Crux.logits(px, s))
    sum(value(π, s) .* pdfs, dims=1)
end

function Ef_target(π, 𝒫, 𝒟; kwargs...)
	πs = trainable_policies(π)
	if length(πs) > 1
		ids = 𝒟[:id][:]
		all_values = vcat([value_estimate(d, 𝒫[:px], 𝒟[:sp]) for d in πs]...)
		values = sum(all_values .* Flux.onehotbatch(ids, collect(1:length(πs))), dims=1)
	else
		values = value_estimate(πs[1], 𝒫[:px], 𝒟[:sp])
	end
	return 𝒟[:done] .* (𝒟[:r] .> 𝒫[:f_target_train][1]) .+ (1.f0 .- 𝒟[:done]) .* values
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
        
        loss(Q, y, agg = isnothing(weight) ? mean : Crux.weighted_mean(𝒟[weight]))
    end
end

function ValueBasedIS(;agent::PolicyParams,
			  train_actor=false,
			  elite_frac = 0.1,
			  N_elite_candidate=100,
			  f_target,
			  ΔN,
              a_opt::NamedTuple=(;), 
              c_opt::NamedTuple=(;), 
              log::NamedTuple=(;), 
              required_columns=[],
			  recent_batch_only=false,
			  name = "value_based_is",
              kwargs...)
    if train_actor
		@error "not implemented"
	else
		a_opt = nothing
	end
	
	# If MIS, make sure we record an ID. 
	if agent.π isa MISPolicy
		push!(required_columns, :id)
	end
	
    EvaluationSolver(;agent=PolicyParams(agent, π⁻=deepcopy(agent.π)),
                    𝒫=(;px=agent.pa, f_target=[f_target], f_target_train=[f_target], elite_frac, N_elite_candidate),
					ΔN=ΔN,
					training_type=:value,
					recent_batch_only=recent_batch_only,
                    log=LoggerParams(;dir = "log/$name", log...),
                    a_opt=a_opt,
					c_opt=TrainingParams(;loss=td_loss_mis(weight=:fwd_importance_weight), name="critic_", epochs=ΔN, c_opt...),
                    target_fn=Ef_target,
                    required_columns=unique([required_columns..., :return, :traj_importance_weight, :fwd_importance_weight, :importance_weight]),
					pre_train_callback=gradual_target_increase,
                    kwargs...)
end
        
