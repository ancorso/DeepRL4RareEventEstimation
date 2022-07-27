include("evaluation.jl")

function pgis_loss(π, 𝒫, 𝒟; info = Dict())
	πs = trainable_policies(π)
	
	# Compute the baseline
	b = ignore_derivatives() do
		if 𝒫[:use_baseline]
			if length(πs) > 1
				bs = vcat([logpdf(d, 𝒟[:s], 𝒟[:a]) for d in πs]...)
				b = sum(bs .* oh, dims=1)
			else
				b = value(πs[1], 𝒟[:s])
			end
		else
			b = 0f0
		end
		b
	end
	
	# Compute the log probability
	if length(πs) > 1
		ids = 𝒟[:id][:]
		oh = Flux.onehotbatch(𝒟[:id][:], collect(1:length(πs)))
		logpdfs = vcat([logpdf(d, 𝒟[:s], 𝒟[:a]) for d in πs]...)
		new_probs = sum(logpdfs .* oh, dims=1)
	else
		new_probs = logpdf(πs[1], 𝒟[:s], 𝒟[:a])
	end
	
	# Log relevant parameters
	ignore_derivatives() do
		info[:kl] = mean(𝒟[:logprob] .- new_probs)
		info[:mean_baseline] = mean(b)
	end 
	
	-mean(new_probs .* ((𝒟[:return] .> 𝒫[:f_target_train][1]) .* 𝒟[:traj_importance_weight] .- b))
end

function value_loss(π, 𝒫, D; kwargs...)
	πs = trainable_policies(π)
	if length(πs) > 1
		ids = 𝒟[:id][:]
		valss = vcat([value(d, D[:s]) for d in πs]...)
		vals = sum(valss .* Flux.onehotbatch(ids, collect(1:length(πs))), dims=1)
	else
		vals = value(πs[1], D[:s])
	end
	
	returns = (D[:return] .> 𝒫[:f_target_train][1]) .* D[:traj_importance_weight]
	
	Flux.mse(vals, returns)
end

function PolicyGradientIS(;agent::PolicyParams,
			  ΔN,
			  N_elite_candidate=ΔN,
			  use_baseline=false,
			  elite_frac=0.1,
			  f_target,
              a_opt::NamedTuple=(;), 
              c_opt::NamedTuple=(;), 
              log::NamedTuple=(;), 
              required_columns=[],
			  buffer_size,
			  name = "pgis",
			  recent_batch_only=true,
              kwargs...)
    if use_baseline
		name = string(name, "_baseline")
		c_opt=TrainingParams(;loss=value_loss, name = "critic_", c_opt...)
	else
		c_opt = nothing
	end
	# If MIS, make sure we record an ID. 
	if agent.π isa MISPolicy
		push!(required_columns, :id)
	end
    EvaluationSolver(;agent=agent,
                    𝒫=(;use_baseline, elite_frac, f_target=[f_target], f_target_train=[f_target], N_elite_candidate),
					buffer_size,
					training_type=:policy_gradient,
					recent_batch_only=recent_batch_only,
                    log=LoggerParams(;dir = "log/$name", log...),
                    a_opt=TrainingParams(;loss=pgis_loss, early_stopping = (infos) -> (infos[end][:kl] > 0.015), name = "actor_", a_opt...),
                    c_opt=c_opt,
                    required_columns = unique([required_columns..., :logprob, :return, :traj_importance_weight]),
					pre_train_callback=gradual_target_increase,
                    kwargs...)
end
        
    



