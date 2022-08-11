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
	# samps = []
	# for i=1:𝒫[:N_samples]
	# 	anom, _ = exploration(𝒫[:px], s)
	# 	# a, logqa = exploration(π, s)
	# 	# w = exp.(logpdf(𝒫[:px], s, a) .- logqa)
	# 	# push!(samps, value(π, s, a) .* w)
	# 	push!(samps, value(π, s, anom))
	# end
	# return mean(samps)
	samps = []
	for a in 𝒫[:xi]
		push!(samps, value(π, s, repeat(a, 1, size(s)[end])))
	end
	
	return sum(𝒫[:wi] .* samps)
end

function Ef_target(π, 𝒫, 𝒟; kwargs...)
	values = value_estimate(π, 𝒟[:sp], 𝒫)
	values[𝒟[:done]] .= 0f0
	
	return 𝒟[:done] .* (𝒟[:r] .> 𝒫[:f_target_train_current][1]) .+ (1.f0 .- 𝒟[:done]) .* values
end

function actor_loss_continuous_is(π, 𝒫, 𝒟; info=Dict())
	a, logqa = exploration(π, 𝒟[:s])
	
	# Compute the estimated value at each state and the fwd importance weight for visitation frequency
	f_s, ws = ignore_derivatives() do
		fs = value_estimate(π, 𝒟[:s], 𝒫) .+ 1f-20
		
		ws = 𝒟[:fwd_importance_weight] ./ 𝒟[:importance_weight]
		ws[𝒟[:importance_weight] .== 0f0] .= 0f0
		
		fs, ws
	end
	
	f_sa = value(π, 𝒟[:s], a) .+ 1f-20
	qa = exp.(logqa)
	pa = exp.(logpdf(𝒫[:px], 𝒟[:s], a)) 
	
	# Option 1: DKL(qθ || qstar)
	# qstar = f_sa .* pa ./ f_s
	# mean( ws .* (log.(qa ./ qstar)))
	
	# Option 2: DKL(qstar || qθ)
	-mean(ws .* f_sa .* pa .* logqa ./ (qa .* f_s))
end

function ValueBasedIS(;agent::PolicyParams,
			  prioritized = true,
			  training_buffer_size,
			  buffer_size,
			  priority_fn=Crux.td_error,
			  train_actor=false,
			  elite_frac = 0.1,
			  xi, # quadrature pts
			  wi, # quadrature weights
			  f_target,
			  ΔN,
              a_opt::NamedTuple=(;), 
              c_opt::NamedTuple=(;), 
              log::NamedTuple=(;), 
              required_columns=Symbol[],
			  name = "value_based_is",
              kwargs...)
    if train_actor
		a_opt=TrainingParams(;loss=actor_loss_continuous_is, name="actor_", a_opt...)
	else
		a_opt = nothing
	end
	
	f_target_train = fill(0f0, length(all_policies(agent.π)))
	
	# If MIS, make sure we record an ID. 
	if agent.π isa MISPolicy
		push!(required_columns, :id)
		
	end
	
	required_columns=unique([required_columns..., :return, :traj_importance_weight, :fwd_importance_weight, :importance_weight])
	
    EvaluationSolver(;agent=PolicyParams(agent, π⁻=deepcopy(agent.π)),
                    𝒫=(;px=agent.pa, elite_frac, f_target=[f_target], f_target_train, f_target_train_current=[0f0], wi, xi),
					ΔN,
					training_buffer_size,
					buffer_size,
					required_columns,
					buffer = ExperienceBuffer(S, agent.space, buffer_size, required_columns, prioritized=prioritized),
					training_type=:value,
                    log=LoggerParams(;dir = "log/$name", log...),
                    a_opt=a_opt,
					# c_opt=TrainingParams(;loss=Crux.td_loss(weight=:fwd_importance_weight), name="critic_", epochs=ΔN, c_opt...),
					c_opt=TrainingParams(;loss=Crux.td_loss(), name="critic_", epochs=ΔN, c_opt...),
                    target_fn=Ef_target,
					priority_fn,
                    kwargs...)
end
        
