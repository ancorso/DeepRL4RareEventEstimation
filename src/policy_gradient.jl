include("evaluation.jl")

function pgis_loss(π, 𝒫, 𝒟; info = Dict())
	act = actor(π)
	
	if act isa MISPolicy
		ids = 𝒟[:id][:]
		logpdfs = vcat([logpdf(d, 𝒟[:s], 𝒟[:a]) for d in trainable_policies(act)]...)
		new_probs = sum(logpdfs .* Flux.onehotbatch(𝒟[:id][:], collect(1:length(trainable_policies(act)))), dims=1)
	else
		new_probs = logpdf(act, 𝒟[:s], 𝒟[:a])
	end
	
	b = Flux.ignore_derivatives() do
		info[:kl] = mean(𝒟[:logprob] .- new_probs)
		info[:mean_weight] = mean(𝒟[:traj_importance_weight])
		b = 𝒫[:use_baseline] ? value(π, 𝒟[:s]) : 0f0
		info[:mean_baseline] = mean(b)
		if act isa MISPolicy
			for i=1:length(trainable_policies(act))
				info["index$(i)_frac"] = sum(𝒟[:id] .== i) / length(𝒟[:id])
			end
		end

		b
	end 
	
	-mean(new_probs .* ((𝒟[:return] .> 𝒫[:f_target_train][1]) .* 𝒟[:traj_importance_weight] .- b))
end

function value_loss(π, 𝒫, D; kwargs...)
	vals = value(π, D[:s]) #TODO: Maybe try with multiple
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
        
    



