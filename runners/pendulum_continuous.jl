using Crux, Distributions
include("utils.jl")
include("../environments/pendulum_problem.jl")

## Parameters
# Experiment params
Neps=50_000
Neps_gt=10_000_00
Ntrials=10
dir="results/pendulum_continuous/"

# Problem setup params
failure_target=π/4
dt = 0.1
Nsteps_per_episode = 20
noise_dist=Normal(0, 0.3)

# quadrature points -> https://jblevins.org/notes/quadrature
xi, wi = quadrature_pts(noise_dist.μ, noise_dist.σ)

Px, mdp = gen_topple_mdp(px=noise_dist, Nsteps=Nsteps_per_episode, dt=dt, failure_thresh=failure_target, discrete=false)
S = state_space(mdp)
A = action_space(Px)

# Networks
function net(out_act...; Nin=3, Nout=1, Nhiddens=[32, 32])
    hiddens = [Dense(Nhiddens[idx], Nhiddens[idx+1], relu) for idx in 1:length(Nhiddens)-1]
    Chain(Dense(Nin, Nhiddens[begin], relu), hiddens..., Dense(Nhiddens[end], Nout, out_act...)) # Basic architecture
end
V() = ContinuousNetwork(net(sigmoid)) # Value network
function Π(Nhiddens=[32, 32])
	base = net(Nout=4, Nhiddens=Nhiddens)
	μ = ContinuousNetwork(Chain(base..., Dense(4, 1)))
	logΣ = ContinuousNetwork(Chain(base..., Dense(4, 1)))
	GaussianPolicy(μ, logΣ, true)
end
AC(A=Π()) = ActorCritic(A, V()) # Actor-crtic for baseline approaches
QSA() = ContinuousNetwork(net(sigmoid, Nin=4)) # Q network for value-based
AQ(A=Π()) = ActorCritic(A, QSA()) # Actor-critic for continuous value-based approach
function Π_CEM(Npolicies; dμ = Distributions.Uniform(-0.2, 0.2))
	if Npolicies == 1
		return DistributionPolicy(Normal(rand(dμ), 0.3))
	else
		return MISPolicy([DistributionPolicy(Normal(rand(dμ), 0.3)) for _ in 1:Npolicies])
	end
end

# Solver parameters
Nbuff = Neps*Nsteps_per_episode
Npretrain=100
shared_params(name, π) = (
	agent=PolicyParams(;π, pa=Px), 
    N=Neps, 
    S,
    f_target=failure_target,
    buffer_size=Nbuff,
    log=(dir="log/$name", period=1000)
)
						  
mc_params(N=Neps) = (
	agent=PolicyParams(;π=Px, pa=Px), 
	N=N, 
	S, 
	buffer_size=N*Nsteps_per_episode, 
	𝒫=(;f_target=failure_target)
)
			 
cem_params(name, π) = (
	ΔN=200, 
	training_buffer_size=200*Nsteps_per_episode, 
	shared_params(name, π)...
)
			 
pg_params(name, π; use_baseline=false, kwargs...) = (
	ΔN=200, 
	use_baseline,
	c_opt=(;max_batches=200, batch_size=1024),
	training_buffer_size=200*Nsteps_per_episode,
	agent_pretrain=use_baseline ? pretrain_AV(mdp, Px, v_target=0.1, Nepochs=Npretrain) : pretrain_policy(mdp, Px, Nepochs=Npretrain),
	shared_params(name, π)...,
	kwargs...
)

vb_params(name, π; kwargs...) = (
	ΔN=20,
	train_actor=true,
	training_buffer_size=3200*Nsteps_per_episode,
	c_opt=(epochs=20,), 
	agent_pretrain=pretrain_AQ(mdp, Px, v_target=0.1, Nepochs=Npretrain),
	xi,
	wi,
	shared_params(name, π)...,
	kwargs...
)
					  


## Get the ground truth comparison
# data = experiment_setup(;mdp, Ntrials, dir)(()->MCSolver(;mc_params(Neps_gt)...), "gt")
# gt = mean(data[:est])[end]
# gt_std = std(data[:est])[end]

# Ground truth experiment
# D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps = 1000)
# vals = D[:r][:][D[:done][:] .== 1]
# histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# sum(D[:r] .> π/4) / sum(D[:done][:] .== 1)

gt = 1.96f-5
gt_std = 9.823442f-7
plot_init = ()->plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))

# Create our "run_experiment function"
run_experiment = experiment_setup(;mdp, Ntrials, dir, plot_init)

## Experiments: 
standard_exps = [
	(()->MCSolver(;mc_params()...), "MC"),
	(()->CEMSolver(;cem_params("CEM_1", Π_CEM(1))...), "CEM_1"),
	(()->CEMSolver(;cem_params("CEM_2", Π_CEM(2))...), "CEM_2"),
    (()->CEMSolver(;cem_params("CEM_4", Π_CEM(4))...), "CEM_4"),
	(()->PolicyGradientIS(;pg_params("PG_nopretrain", Π())..., agent_pretrain=nothing), "PG_nopretrain"),
	(()->PolicyGradientIS(;pg_params("PG", Π())...), "PG"),
	(()->PolicyGradientIS(;pg_params("PG_defensive", MISPolicy([Π(), Px]))...), "PG_defensive"),
    (()->PolicyGradientIS(;pg_params("PG_baseline", AC(), use_baseline=true)...), "PG_baseline"),
    (()->PolicyGradientIS(;pg_params("PG_MIS2_nopretrain", MISPolicy([Π(), Π()]))..., agent_pretrain=nothing), "PG_MIS2_nopretrain"),
	(()->PolicyGradientIS(;pg_params("PG_MIS2", MISPolicy([Π(), Π()]))...), "PG_MIS2"),
	(()->PolicyGradientIS(;pg_params("PG_MIS2_defense", MISPolicy([Π(), Π(), Px]))...), "PG_MIS2_defense"),
	(()->PolicyGradientIS(;pg_params("PG_MIS2_baseline", MISPolicy([AC(), AC()]), use_baseline=true)...), "PG_MIS2_baseline"),
	(()->PolicyGradientIS(;pg_params("PG_MIS4_nopretrain", MISPolicy([Π(), Π(), Π(), Π()]))..., agent_pretrain=nothing), "PG_MIS4_nopretrain"),
	(()->PolicyGradientIS(;pg_params("PG_MIS4", MISPolicy([Π(), Π(), Π(), Π()]))...), "PG_MIS4"),
	(()->PolicyGradientIS(;pg_params("PG_MIS4_defense", MISPolicy([Π(), Π(), Π(), Π(), Px]))...), "PG_MIS4_defense"),
	(()->PolicyGradientIS(;pg_params("PG_MIS4_baseline", MISPolicy([AC(), AC(), AC(), AC()]), use_baseline=true)...), "PG_MIS4_baseline"),
	(()->ValueBasedIS(;vb_params("VB_nopretrain", AQ())..., agent_pretrain=nothing),  "VB_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB", AQ())...),  "VB"),
	(()->ValueBasedIS(;vb_params("VB_defensive", MISPolicy([AQ(), Px]))...), "VB_defensive"),
	(()->ValueBasedIS(;vb_params("VB_MIS2_nopretrain", MISPolicy([AQ(), AQ()]))..., agent_pretrain=nothing),  "VB_MIS2_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB_MIS2", MISPolicy([AQ(), AQ()]))...),  "VB_MIS2"),
	(()->ValueBasedIS(;vb_params("VB_MIS2_defensive", MISPolicy([AQ(), AQ(), Px]))...),  "VB_MIS2_defensive"),
	(()->ValueBasedIS(;vb_params("VB_MIS4_nopretrain", MISPolicy([AQ(), AQ(), AQ(), AQ()]))..., agent_pretrain=nothing),  "VB_MIS4_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB_MIS4", MISPolicy([AQ(), AQ(), AQ(), AQ()]))...),  "VB_MIS4"),
	(()->ValueBasedIS(;vb_params("VB_MIS4_defensive", MISPolicy([AQ(), AQ(), AQ(), AQ(), Px]))...),  "VB_MIS4_defensive")
]

Threads.@threads for (𝒮fn, name) in standard_exps; run_experiment(𝒮fn, name); end
# for (𝒮fn, name) in standard_exps; run_experiment(𝒮fn, name); end

# # TODO Things to look into more
# # The main issue with the value based method is that each policy doesn't want to specialize. Can we combine the actor losses to encourage specialization?
# # Can we include exploration for the value based method?
