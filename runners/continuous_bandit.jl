using Crux, Distributions, POMDPGym, Plots
include("utils.jl")

## Parameters
# Experiment params
Neps=20000

# Problem setup params
failure_target=0.7
mdp = ContinuousBanditMDP(0, 1) #NOTE: To get multimodal behavior, need to modify POMDPGym impl (add a 1- to the reward)
Px = DistributionPolicy(Normal(0, 0.5))

xi, wi = quadrature_pts(Px.distribution.μ, Px.distribution.σ)

S = state_space(mdp)
A = action_space(Px)

# Show returns
D = episodes!(Sampler(mdp, Px), Neps=1000)
histogram(D[:r][:])

# Networks
net(out_act...;Nin=1, Nout=1) = Chain(Dense(Nin, 12, relu), Dense(12,12, relu), Dense(12, Nout, out_act...)) # Basic architecture
V() = ContinuousNetwork(net(sigmoid)) # Value network
function Π()
	base = net(Nout=2)
	μ = ContinuousNetwork(Chain(base..., Dense(2, 1)))
	logΣ = ContinuousNetwork(Chain(base..., Dense(2, 1)))
	GaussianPolicy(μ, logΣ, true)
end
function Π_mixture()
	base = net(Nout=2)
	
	mu1 = ContinuousNetwork(Chain(base..., Dense(2, 1)))
    logΣ1 = ContinuousNetwork(Chain(base..., Dense(2, 1)))
	
    mu2 = ContinuousNetwork(Chain(base..., Dense(2, 1)))
    logΣ2 = ContinuousNetwork(Chain(base..., Dense(2, 1)))
	
    αs = ContinuousNetwork(Chain(base..., Dense(2, 2), softmax), 2)
    MixtureNetwork([GaussianPolicy(mu1, logΣ1, true), GaussianPolicy(mu2, logΣ2, true)], αs)
end
AC(A=Π()) = ActorCritic(A, V()) # Actor-crtic for baseline approaches
QSA() = ContinuousNetwork(net(sigmoid, Nin=2)) # Q network for value-based
AQ(A=Π()) = ActorCritic(A, QSA()) # Actor-critic for continuous value-based approach




# Solver parameters
Nbuff = Neps
shared_params(name, π) = (agent=PolicyParams(;π, pa=Px), 
                          N=Neps, 
                          S,
                          f_target=failure_target,
                          buffer_size=Nbuff,
                          log=(dir="log/$name", period=1000))
						  
mc_params(N=Neps) = (agent=PolicyParams(;π=Px, pa=Px), 
			 N=N, 
			 S, 
			 buffer_size=N, 
			 𝒫=(;f_target=failure_target))
			 
pg_params(name, π, use_baseline=false) = (ΔN=200, 
										  use_baseline,
										  training_buffer_size=200,
										  agent_pretrain=pretrain_policy(mdp, Px),
										  shared_params(name, π)...)
vb_params(name, π) = (ΔN=4,
					  train_actor=true,
					  c_opt=(epochs=4,), 
					  agent_pretrain=pretrain_AQ(mdp, Px, v_target=0.1),
					  training_buffer_size=5000,
					  xi, 
					  wi,
					  shared_params(name, π)...)


## Policy gradient MIS approach 
𝒮 = PolicyGradientIS(;pg_params("PG", Π())...)
solve(𝒮, mdp)
plot(-4:0.02:4, (a) -> reward(mdp, [0], a))
hline!([failure_target])
a1 = action(𝒮.agent.π, 𝒮.buffer[:s])
histogram!(a1[:], normalize=true)


𝒮 = ValueBasedIS(;vb_params("VB", AQ())...)
solve(𝒮, mdp)
plot(-4:0.02:4, (a) -> reward(mdp, [0], a))
hline!([failure_target])
plot!(-4:0.02:4, (a) -> value(𝒮.agent.π, [-1.], [a])[1])
a1 = action(𝒮.agent.π, 𝒮.buffer[:s])
histogram!(a1[:], normalize=true)


𝒮 = PolicyGradientIS(;pg_params("PG_MIS", MISPolicy([Π(), Π(), Π(), Π()], [1/4, 1/4, 1/4, 1/4]))...)
fs, ws = solve(𝒮, mdp)

plot(-4:0.02:4, (a) -> reward(mdp, [0], a))
hline!([failure_target])
a1 = action(𝒮.agent.π.distributions[1], 𝒮.buffer[:s])
a2 = action(𝒮.agent.π.distributions[2], 𝒮.buffer[:s])
a3 = action(𝒮.agent.π.distributions[3], 𝒮.buffer[:s])
a4 = action(𝒮.agent.π.distributions[4], 𝒮.buffer[:s])
histogram!(a1[:], normalize=true)
histogram!(a2[:], normalize=true)
histogram!(a3[:], normalize=true)
histogram!(a4[:], normalize=true)

## Policy gradient Mixture approach 
𝒮 = PolicyGradientIS(;pg_params("PG_mixture", Π_mixture())..., agent_pretrain=nothing)
fs, ws = solve(𝒮, mdp)

plot(-4:0.02:4, (a) -> reward(mdp, [0], a))
hline!([failure_target])
a1 = action(𝒮.agent.π.networks[1], 𝒮.buffer[:s])
a2 = action(𝒮.agent.π.networks[2], 𝒮.buffer[:s])
histogram!(a1[:], normalize=true)
histogram!(a2[:], normalize=true)

## Value based MIS approach
𝒮 = ValueBasedIS(;vb_params("VB_MIS", MISPolicy([AQ(), AQ()], [0.5, 0.5]))...)
# 𝒮 = ValueBasedIS(;vb_params("VB", AQ())...)
fs, ws = solve(𝒮, mdp)


plot(-4:0.02:4, (a) -> reward(mdp, [0], a))
hline!([failure_target])
plot!(-4:0.02:4, (a) -> value(𝒮.agent.π.distributions[1], [-1.], [a])[1])
plot!(-4:0.02:4, (a) -> value(𝒮.agent.π.distributions[2], [-1.], [a])[1])
a1 = action(actor(𝒮.agent.π).distributions[1], zeros(1,1000))
a2 = action(actor(𝒮.agent.π).distributions[2], zeros(1,1000))
histogram!(a1[:], normalize=true)
histogram!(a2[:], normalize=true)


## Value-based mixture approach
𝒮 = ValueBasedIS(;vb_params("VB_Mixture", AQ(Π_mixture()))...)
fs, ws = solve(𝒮, mdp)

plot(-4:0.02:4, (a) -> reward(mdp, [0], a))
hline!([failure_target])
plot!(-4:0.02:4, (a) -> value(𝒮.agent.π, [-1.], [a])[1])
a1 = action(𝒮.agent.π.A.networks[1], zeros(1,1000))
a2 = action(𝒮.agent.π.A.networks[2], zeros(1,1000))
histogram!(a1[:], normalize=true)
histogram!(a2[:], normalize=true)

