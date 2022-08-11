include("evaluation.jl")

function pgis_loss(π, 𝒫, 𝒟; info = Dict())
	# Compute the baseline
	b = ignore_derivatives() do
		 𝒫[:use_baseline] ? value(π, 𝒟[:s]) : 0f0
	end
	
	# Compute the log probability
	new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
	
	# Log relevant parameters
	ignore_derivatives() do
		info[:kl] = mean(𝒟[:logprob] .- new_probs)
		info[:mean_baseline] = mean(b)
	end 
	
	-mean(new_probs .* ((𝒟[:return] .> 𝒫[:f_target_train_current][1]) .* 𝒟[:traj_importance_weight] .- b))
end

function value_loss(π, 𝒫, D; kwargs...)
	vals = value(π, D[:s])
	
	returns = (D[:return] .> 𝒫[:f_target_train_current][1]) .* D[:traj_importance_weight]
	
	Flux.mse(vals, returns)
end

function PolicyGradientIS(;agent::PolicyParams,
			  ΔN,
			  use_baseline=false,
			  elite_frac=0.1,
			  target_kl = 0.015,
			  f_target,
              a_opt::NamedTuple=(;), 
              c_opt::NamedTuple=(;), 
              log::NamedTuple=(;), 
              required_columns=[],
			  training_buffer_size,
			  buffer_size,
			  name = "pgis",
              kwargs...)
    if use_baseline
		name = string(name, "_baseline")
		c_opt=TrainingParams(;loss=value_loss, name="critic_", c_opt...)
	else
		c_opt = nothing
	end
	f_target_train = fill(0f0, length(all_policies(agent.π)))
	# If MIS, make sure we record an ID. 
	agent.π isa MISPolicy && push!(required_columns, :id)
	required_columns = unique([required_columns..., :logprob, :return, :traj_importance_weight])
    EvaluationSolver(;agent,
                    𝒫=(;use_baseline, elite_frac, f_target=[f_target], f_target_train, f_target_train_current=[0f0]),
					buffer_size,
					ΔN, 
					training_buffer_size,
					required_columns,
					training_type=:policy_gradient,
                    log=LoggerParams(;dir = "log/$name", log...),
                    a_opt=TrainingParams(;loss=pgis_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", max_batches=1000, a_opt...),
                    c_opt=c_opt,					
                    kwargs...)
end
        
    



