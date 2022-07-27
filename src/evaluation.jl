using Parameters, Random
import Zygote: ignore_derivatives

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
	recent_batch_only = true # Whether or not to train on all prior data or just the recent batch
	weight_fn = (agent, data, ep) -> trajectory_pdf(agent.pa, data, ep) / trajectory_pdf(agent.π, data, ep)
	agent_pretrain=nothing # Function to pre-train the agent before any rollouts

	# Create buffer to store all of the samples
	buffer_size = N*max_steps
	buffer = ExperienceBuffer(S, agent.space, buffer_size, required_columns)

    # Stuff for off policy update
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    target_fn=nothing # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
    priority_fn = Crux.td_error  # function for prioritized replay
end

MCSolver(args...; kwargs...) = EvaluationSolver(args...; kwargs...)

function gradual_target_increase(𝒮, 𝒟; info)
	fs = 𝒟[:return][𝒟[:episode_end] .== true][:]
	start = max(1, length(fs) - 𝒮.𝒫[:N_elite_candidate])
	fs = fs[start:end]

	elite_target_min = sort(fs, rev=true)[max(1, floor(Int, length(fs) * 𝒮.𝒫[:elite_frac]))]
	target = min(elite_target_min, 𝒮.𝒫[:f_target][1])
	info[:f_target_train] = target
	𝒮.𝒫[:f_target_train][1] = target
end

function assign_mode_ids(𝒮, 𝒟; info=Dict())
		πs = trainable_policies(𝒮.agent.π)
		length(πs) == 1 && return # If there is only one trainable policy, only it will be trained
		eps = episodes(𝒟)
		for ep_t in eps
			if 𝒟[:id][1, ep_t[1]] == 0
				ep = ep_t[1]:ep_t[2]
				# update the failure mode id of each sample (this might change over time as well)
				id = argmax([trajectory_pdf(d, 𝒟, ep) for d in πs])
				𝒟[:id][1, ep] .= id
		
				# Update the logprobabiliyt of the (s,a) tuple for the new failure mode
				if haskey(𝒟, :logprob)
					𝒟[:logprob][:, ep] .= logpdf(𝒮.agent.π.distributions[id], 𝒟[:s][:,ep], 𝒟[:a][:,ep])
				end
			end
		end
		
		# Record the fraction of samples from each
		for i=1:length(πs)
			info["index$(i)_frac"] = sum(𝒟[:id] .== i) / length(𝒟[:id])
		end
end

function policy_gradient_training(𝒮::EvaluationSolver, buffer)
    info = Dict()
    
    # Train Actor
    batch_train!(actor(𝒮.agent.π), 𝒮.a_opt, 𝒮.𝒫, buffer, info=info, π_loss=𝒮.agent.π)
    
    # Optionally update critic
    if !isnothing(𝒮.c_opt)
        batch_train!(critic(𝒮.agent.π), 𝒮.c_opt, 𝒮.𝒫, buffer, info=info, π_loss=𝒮.agent.π)
    end
    
    info
end

function value_training(𝒮::EvaluationSolver, buffer)
    𝒟 = buffer_like(buffer, capacity=𝒮.c_opt.batch_size, device=device(𝒮.agent.π))
    
    infos = []
	
    # Loop over the desired number of training steps
    for epoch in 1:𝒮.c_opt.epochs
        # Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
        rand!(𝒟, buffer, i=𝒮.i)
        
        # Dictionary to store info from the various optimization processes
        info = Dict()
        
        # Compute target
        y = 𝒮.target_fn(𝒮.agent.π⁻, 𝒮.𝒫, 𝒟, i=𝒮.i)
        
        # # Update priorities (for prioritized replay)
        # isprioritized(𝒮.buffer) && update_priorities!(𝒮.buffer, 𝒟.indices, cpu(𝒮.priority_fn(𝒮.agent.π, 𝒮.𝒫, 𝒟, y)))
        
        # Train the critic
        if ((epoch-1) % 𝒮.c_opt.update_every) == 0
            Crux.train!(critic(𝒮.agent.π), (;kwargs...) -> 𝒮.c_opt.loss(𝒮.agent.π, 𝒮.𝒫, 𝒟, y; kwargs...), 𝒮.c_opt, info=info)
        end
        
        # Train the actor 
        if !isnothing(𝒮.a_opt) && ((epoch-1) % 𝒮.a_opt.update_every) == 0
            Crux.train!(actor(𝒮.agent.π), (;kwargs...) -> 𝒮.a_opt.loss(𝒮.agent.π, 𝒮.𝒫, 𝒟; kwargs...), 𝒮.a_opt, info=info)
        
            # Update the target network
            𝒮.target_update(𝒮.agent.π⁻, 𝒮.agent.π)
        end
        
        # Store the training information
        push!(infos, info)
        
    end
    # If not using a separate actor, update target networks after critic training
    isnothing(𝒮.a_opt) && 𝒮.target_update(𝒮.agent.π⁻, 𝒮.agent.π, i=𝒮.i + 1:𝒮.i + 𝒮.ΔN)
    
    aggregate_info(infos)
end

function POMDPs.solve(𝒮::EvaluationSolver, mdp)
	@assert haskey(𝒮.𝒫, :f_target)
	
	# Pre-train the policy if a function is provided
	if !isnothing(𝒮.agent_pretrain) && 𝒮.i == 0
		𝒮.agent_pretrain(𝒮.agent.π)
		if !isnothing(𝒮.agent.π⁻)
			𝒮.agent.π⁻=deepcopy(𝒮.agent.π)
		end 
	end
	
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
		episodes!(s, 𝒮.buffer, Neps=𝒮.ΔN, explore=true, i=𝒮.i, cb=(D) -> 𝒮.post_sample_callback(D, info=info, 𝒮=𝒮))
		end_index=length(𝒮.buffer)
		
		# Record the average weight of the samples
		ep_ends = 𝒮.buffer[:episode_end][1,start_index:end_index]
		info[:mean_weight] = sum(𝒮.buffer[:traj_importance_weight][1,start_index:end_index][ep_ends]) / sum(ep_ends)
		@assert !isnan(info[:mean_weight])

		# If we are training then update required values and train
		training_info = Dict()
		if 𝒮.training_type != :none
		
			𝒟 = 𝒮.recent_batch_only ? ExperienceBuffer(minibatch(𝒮.buffer, start_index:end_index)) : 𝒮.buffer
			
			# Pre-train callback, used to make changes to the buffer and update f
	        𝒮.pre_train_callback(𝒮, 𝒟; info)
			
			𝒮.agent.π isa MISPolicy && assign_mode_ids(𝒮, 𝒟; info)
			
			
	        # Train the networks
	        if 𝒮.training_type == :policy_gradient
	            training_info = policy_gradient_training(𝒮, 𝒟)
	        elseif 𝒮.training_type == :value
	            training_info = value_training(𝒮, 𝒟)
	        else
	            @error "uncregonized training type: $training_type"
	        end
		end
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info, training_info, 𝒮=𝒮)
    end
    𝒮.i += 𝒮.ΔN
	
	# Extract the samples
	eps = episodes(𝒮.buffer)
	fs = [𝒮.buffer[:return][1, ep[1]] > 𝒮.𝒫[:f_target][1] for ep in eps]
	ws = [𝒮.buffer[:traj_importance_weight][1, ep[1]] for ep in eps]
	
    fs, ws
end

