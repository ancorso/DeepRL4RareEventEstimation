using Parameters, Random
import Zygote: ignore_derivatives

function safe_weight_fn(agent, data, ep)
	logp = trajectory_logpdf(agent.pa, data, ep)
	logq = trajectory_logpdf(agent.π, data, ep)
	if logq == -Inf
		return 0f0
	else
		return exp(logp - logq)
	end	
end

@with_kw mutable struct EvaluationSolver <: Solver
    agent::PolicyParams # Policy
    S::AbstractSpace # State space
    N::Int = 1000 # Number of episode samples
    ΔN::Int = 200 # Number of episode samples between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of episode interactions
    a_opt::Union{Nothing, TrainingParams} = nothing# Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the critic
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    post_sample_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling experience
    pre_train_callback = (𝒮; kwargs...) -> nothing # callback that gets called once prior to training
    required_columns = Symbol[:traj_importance_weight, :return]
	
	# Stuff specific to estimation
	training_type = :none # Type of training loop, either :policy_gradient, :value, :none
	training_buffer_size = ΔN*max_steps # Whether or not to train on all prior data or just the recent batch
	weight_fn = safe_weight_fn
	agent_pretrain=nothing # Function to pre-train the agent before any rollouts

	# Create buffer to store all of the samples
	buffer_size = N*max_steps
	buffer = ExperienceBuffer(S, agent.space, buffer_size, required_columns)
	𝒟 = nothing

    # Stuff for off policy update
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    target_fn=nothing # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
    priority_fn = Crux.td_error  # function for prioritized replay
end

MCSolver(args...; kwargs...) = EvaluationSolver(args...; kwargs...)
function CEMSolver(; agent, 
					 ΔN, 
					 log::NamedTuple=(;), 
					 required_columns=Symbol[], 
					 elite_frac=0.1, 
					 f_target,
					 buffer_size,
					 name="cem",
					 kwargs...)
					 
	f_target_train = fill(0f0, length(all_policies(agent.π)))
	# If MIS, make sure we record an ID. 
	agent.π isa MISPolicy && push!(required_columns, :id)
	required_columns = unique([required_columns..., :logprob, :return, :traj_importance_weight])
	𝒫=(;elite_frac, f_target=[f_target], f_target_train)
	
	EvaluationSolver(;agent,
					  ΔN,
					  buffer_size, 
					  required_columns, 
					  𝒫,
					  training_type=:cem, 
					  log=LoggerParams(;dir = "log/$name", log...),
					  kwargs...)
end

function gradual_target_increase(𝒮, 𝒟; info)
	fs = 𝒟[:return][𝒟[:episode_end] .== true][:]
	πt = trainable_indices(𝒮.agent.π)
	
	# In the MIS case, we have to set targets for different modes separately
	if 𝒮.agent.π isa MISPolicy
		ids = 𝒟[:id][𝒟[:episode_end] .== true][:]
		for (id, π) in enumerate(all_policies(𝒮.agent.π))
			fs_id = fs[ids .== id]
			if length(fs_id) == 0 # If there are no samples associated with a policy, set its target back to 0f0
				target = 0f0
			elseif id in πt # for trainable distributions, gradually increase the target		
				elite_target_min = sort(fs_id, rev=true)[max(1, floor(Int, length(fs_id) * 𝒮.𝒫[:elite_frac]))]
				target = min(elite_target_min, 𝒮.𝒫[:f_target][1])
			else # for non trainable distributions, just set the real target
				target = 𝒮.𝒫[:f_target][1]
			end
			info["f_target_train_$id"] = target
			𝒮.𝒫[:f_target_train][id] = target
		end
	else
		elite_target_min = sort(fs, rev=true)[max(1, floor(Int, length(fs) * 𝒮.𝒫[:elite_frac]))]
		target = min(elite_target_min, 𝒮.𝒫[:f_target][1])
		info[:f_target_train] = target
		𝒮.𝒫[:f_target_train][1] = target
	end
end

function assign_mode_ids(𝒮, 𝒟; info=Dict())
	πs = all_policies(𝒮.agent.π)
	length(πs) == 1 && return # If there is only one trainable policy,
	
	mis = 𝒮.agent.π
	
	weights = mis.weights
	logweights = log.(weights)
	new_weights = zeros(Float32, length(weights)) # new set of weights
	
	eps = episodes(𝒟)
	for ep_t in eps
		ep = ep_t[1]:ep_t[2]
		# update the failure mode id of each sample (this might change over time as well)
		
		# Get the likelihood that the sample "ep" comes from each policy
		pw = [trajectory_logpdf(d, 𝒟, ep) for d in πs] .+ logweights # likelihood times weight
		# γ = sum(pw) == 0 ? weights : pw ./ sum(pw) # normalize (safely)
		id = argmax(pw)
		𝒟[:id][1, ep] .= id
		new_weights[id] += 𝒟[:traj_importance_weight][1, ep_t[1]]*(𝒟[:return][1, ep_t[1]] .> 𝒮.𝒫[:f_target_train][id]) 
		
		# Update the logprobability of the (s,a) tuple for the new failure mode
		if haskey(𝒟, :logprob)
			𝒟[:logprob][:, ep] .= logpdf(πs[id], 𝒟[:s][:,ep], 𝒟[:a][:,ep])
		end
	end
	
	if sum(new_weights) > 0
		# Set a minimum fraction for each distribution
		new_weights =  (new_weights ./ sum(new_weights)) .+ 0.1
		
		# Compute the target weight values
		new_weights = new_weights ./ sum(new_weights)
		
		# Perform a moving average update
		mis.weights += 0.05*(new_weights .- mis.weights)
		
		# Normalize to make sure the total samples is always equal to the 
		mis.weights = mis.weights ./ sum(mis.weights)
	end
	
	
	# Record the fraction of samples from each
	for i=1:length(mis.weights)
		info["index$(i)_frac"] = mis.weights[i]
	end
end

function cem_training(𝒮::EvaluationSolver, 𝒟)
	info = Dict()
	πs = all_policies(𝒮.agent.π)
	for (id, π) in enumerate(πs)
		@assert π isa DistributionPolicy
		𝒟id = haskey(𝒟, :id) ? ExperienceBuffer(minibatch(𝒟, findall(𝒟[:id][:] .== id))) : 𝒟
		weights = ((𝒟id[:return] .> 𝒮.𝒫[:f_target_train][id]) .* 𝒟id[:traj_importance_weight])[:]
		(length(𝒟id) == 0 || sum(weights) == 0) && continue
		a = π.distribution isa Normal ? 𝒟id[:a][1, :] : 𝒟id[:a]
		
		if π.distribution isa ObjectCategorical
			π.distribution = Distributions.fit(typeof(π.distribution), Float64.(a), Float64.(weights), objs=π.distribution.objs)
		else
			π.distribution = Distributions.fit(typeof(π.distribution), Float64.(a), Float64.(weights))
		end
	end
	info
end

function policy_gradient_training(𝒮::EvaluationSolver, 𝒟)
    info = Dict()
	
	πt = trainable_indices(𝒮.agent.π)
	πs = all_policies(𝒮.agent.π)
	
	for id in πt
		π = πs[id]
		𝒟id = haskey(𝒟, :id) ? ExperienceBuffer(minibatch_copy(𝒟, findall(𝒟[:id][:] .== id))) : deepcopy(𝒟)
		𝒮.𝒫[:f_target_train_current][1] = 𝒮.𝒫[:f_target_train][id]
		
		length(𝒟id) == 0 && continue
	
		# Train Actor
		batch_train!(actor(π), 𝒮.a_opt, 𝒮.𝒫, 𝒟id, info=info, π_loss=π)
		
		# Optionally update critic
	    if !isnothing(𝒮.c_opt)
	        batch_train!(critic(π), 𝒮.c_opt, 𝒮.𝒫, 𝒟id, info=info, π_loss=π)
		end
	end
    
    info
end

function value_training(𝒮::EvaluationSolver, 𝒟)
	# Batch buffer
	𝒟batch = buffer_like(𝒟, capacity=𝒮.c_opt.batch_size, device=device(𝒮.agent.π))
    
    infos = []
	
	πt = trainable_indices(𝒮.agent.π)
	πs = all_policies(𝒮.agent.π)
	π⁻s = all_policies(𝒮.agent.π⁻)
	
	# Loop through policies and train 1 at a time
	for id in πt
		# extract the policy and target
		𝒮.𝒫[:f_target_train_current][1] = 𝒮.𝒫[:f_target_train][id]
		π = πs[id]
		π⁻ = π⁻s[id]
		
		𝒟id = haskey(𝒟, :id) ? ExperienceBuffer(minibatch(𝒟, findall(𝒟[:id][:] .== id))) : 𝒟
	
	    # Loop over the desired number of training steps
	    for epoch in 1:𝒮.c_opt.epochs
			
			# length(𝒟id) == 0 && continue
			# rand!(𝒟batch, 𝒟id, i=𝒮.i)
			
	        # Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
			rand!(𝒟batch, 𝒟, i=𝒮.i)
	        
			
	        # Dictionary to store info from the various optimization processes
	        info = Dict()
			
	        # Compute target
	        y = 𝒮.target_fn(π⁻, 𝒮.𝒫, 𝒟batch, i=𝒮.i)
			
	        # # Update priorities (for prioritized replay)
			err = cpu(𝒮.priority_fn(π, 𝒮.𝒫, 𝒟batch, y))
	        isprioritized(𝒟) && update_priorities!(𝒟, 𝒟batch.indices, err)
			
	        # Train the critic
	        if ((epoch-1) % 𝒮.c_opt.update_every) == 0
	            Crux.train!(critic(π), (;kwargs...) -> 𝒮.c_opt.loss(π, 𝒮.𝒫, 𝒟batch, y; kwargs...), 𝒮.c_opt, info=info)
	        end
			
			length(𝒟id) == 0 && continue
			# Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
	        uniform_sample!(𝒟batch, 𝒟id)
	        
	        # Train the actor 
	        if !isnothing(𝒮.a_opt) && ((epoch-1) % 𝒮.a_opt.update_every) == 0
	            Crux.train!(actor(π), (;kwargs...) -> 𝒮.a_opt.loss(π, 𝒮.𝒫, 𝒟batch; kwargs...), 𝒮.a_opt, info=info)
				
	            # Update the target network
	            𝒮.target_update(π⁻, π)
	        end
	        
	        # Store the training information
	        push!(infos, info)
	        
	    end
	    # If not using a separate actor, update target networks after critic training
	    isnothing(𝒮.a_opt) && 𝒮.target_update(π⁻, π, i=𝒮.i + 1:𝒮.i + 𝒮.ΔN)
	end
    
    aggregate_info(infos)
end

function POMDPs.solve(𝒮::EvaluationSolver, mdp)
	try mkdir("frames") catch end
	@assert haskey(𝒮.𝒫, :f_target)
	
	# Pre-train the policy if a function is provided
	if !isnothing(𝒮.agent_pretrain) && 𝒮.i == 0
		𝒮.agent_pretrain(𝒮.agent.π)
		if !isnothing(𝒮.agent.π⁻)
			𝒮.agent.π⁻=deepcopy(𝒮.agent.π)
		end 
	end
	
	# Construct the training buffer
	# TODO: consider different buffers for VB replay?
	𝒮.𝒟 = buffer_like(𝒮.buffer, capacity=𝒮.training_buffer_size, device=device(𝒮.agent.π))
	
    # Construct the training buffer, constants, and sampler
    s = Sampler(mdp, 𝒮.agent, S=𝒮.S, required_columns=𝒮.required_columns, max_steps=𝒮.max_steps, traj_weight_fn=𝒮.weight_fn)
    !isnothing(𝒮.log) && isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i, 𝒮=𝒮)

    # Loop over the desired number of environment interactions
    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Info to collect during training
        info = Dict()
        
        # Sample transitions into the batch buffer
		@assert length(𝒮.buffer) < capacity(𝒮.buffer) # Make sure we never overwrite
		start_index=length(𝒮.buffer) + 1
		𝒮.training_type in [:policy_gradient, :cem] && clear!(𝒮.𝒟)
		episodes!(s, 𝒮.𝒟, store=𝒮.buffer, Neps=𝒮.ΔN, explore=true, i=𝒮.i, cb=(D) -> 𝒮.post_sample_callback(D, info=info, 𝒮=𝒮))
		end_index=length(𝒮.buffer)
		
		# Record the average weight of the samples
		ep_ends = 𝒮.buffer[:episode_end][1,start_index:end_index]
		info[:mean_weight] = sum(𝒮.buffer[:traj_importance_weight][1,start_index:end_index][ep_ends]) / sum(ep_ends)
		@assert !isnan(info[:mean_weight])

		# If we are training then update required values and train
		training_info = Dict()
		if 𝒮.training_type != :none
		
			# Assign each datapoint in the training batch to one of the policies
			assign_mode_ids(𝒮, 𝒮.𝒟; info)
			
			# gradually increase the target
	        gradual_target_increase(𝒮, 𝒮.𝒟; info)
			
			# Plot training frames
			# try
			# 	if mod(𝒮.i, 100) == 0
			# 		function plot_traj(π; label, p=plot())
			# 			D = episodes!(Sampler(mdp, π), Neps=10)
			# 			scatter!(p, D[:s][1, :], D[:s][2, :], label=label)
			# 		end
			# 		ps = []
			# 
			# 		for (i,π) in enumerate(all_policies(𝒮.agent.π))
			# 			if π isa ActorCritic
			# 				if 𝒮.training_type == :policy_gradient
			# 					p = heatmap(0:0.1:2, -1.2:0.1:1.2, (t,θ) -> value(π, [t, θ, 0f0])[1], clims=(0,1))
			# 				else
			# 					p = heatmap(0:0.1:2, -1.2:0.1:1.2, (t,θ) -> value(π, [t, θ, 0f0, 0f0])[1], clims=(0,1))
			# 				end
			# 			else 
			# 				p = plot(ylims=(-1.2,1.2))
			# 			end
			# 			plot_traj(π, label="q$i", p=p)
			# 			push!(ps, p)
			# 		end
			# 
			# 		p=plot(ps..., layout=(length(ps), 1), size=(600, 200*length(ps)))
			# 		savefig(p, "frames/frame$(𝒮.i).png")
			# 	end
			# catch end
			
	        # Train the networks
	        if 𝒮.training_type == :policy_gradient
	            training_info = policy_gradient_training(𝒮, 𝒮.𝒟)
	        elseif 𝒮.training_type == :value
	            training_info = value_training(𝒮, 𝒮.𝒟)
			elseif 𝒮.training_type == :cem
				training_info = cem_training(𝒮, 𝒮.𝒟)
	        else
	            @error "uncregonized training type: $training_type"
	        end
		end
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info, training_info,  𝒮=𝒮)
    end
    𝒮.i += 𝒮.ΔN
	
	# Extract the samples
	eps = episodes(𝒮.buffer)
	fs = [𝒮.buffer[:return][1, ep[1]] > 𝒮.𝒫[:f_target][1] for ep in eps]
	ws = [𝒮.buffer[:traj_importance_weight][1, ep[1]] for ep in eps]
	
    fs, ws
end

