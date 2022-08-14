using Crux, Distributions, LinearAlgebra
include("utils.jl")
include("../environments/pendulum_problem.jl")

## Parameters
test = false

# Experiment params
if test
	Neps=500
	Ntrials=1
	Npretrain=1
else
	Neps=50_000
	Neps_gt=1_000_000
	Ntrials=10
	Npretrain=100
end
dir="results/pendulum_discrete"

# Problem setup params
failure_target=π/4
dt = 0.1
Nsteps_per_episode = 20
noise_dist=Normal(0, 0.4)
Px, mdp = gen_topple_mdp(px=noise_dist, Nsteps=Nsteps_per_episode, dt=dt, failure_thresh=failure_target)
xs = Px.distribution.support
S = state_space(mdp)
A = action_space(Px)

## Networks
function net(out_act...; Nin=3, Nout=A.N, Nhiddens=[32, 32])
    hiddens = [Dense(Nhiddens[idx], Nhiddens[idx+1], relu) for idx in 1:length(Nhiddens)-1]
    Chain(Dense(Nin, Nhiddens[begin], relu), hiddens..., Dense(Nhiddens[end], Nout, out_act...)) # Basic architecture
end
V() = ContinuousNetwork(net(sigmoid, Nout=1)) # Value network
Π(Nhiddens=[32, 32]) = DiscreteNetwork(net(;Nhiddens), xs, always_stochastic=true)
AC(A=Π()) = ActorCritic(A, V()) # Actor-crtic for baseline approaches
Q() = DiscreteNetwork(net(sigmoid, Nout=A.N), xs, always_stochastic=true, logit_conversion=estimator_logits(Px)) # Q network for value-based
function Π_CEM(Npolicies; d = ()->ObjectCategorical(xs, normalize(rand(length(xs)),1)))
	if Npolicies == 1
		return DistributionPolicy(d())
	else
		return MISPolicy([DistributionPolicy(d()) for _ in 1:Npolicies], trainable_indices=collect(1:Npolicies))
	end
end

## Algorithm parameter setup
Nbuff = Neps*Nsteps_per_episode
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
	a_opt=(;optimizer=Flux.Optimiser(Flux.ClipValue(1f0), Adam(3f-4)), batch_size=200*Nsteps_per_episode),
	c_opt=(;max_batches=200, batch_size=1024),
	training_buffer_size=200*Nsteps_per_episode,
	agent_pretrain=use_baseline ? pretrain_AV(mdp, Px, v_target=0.1, Nepochs=Npretrain) : pretrain_policy(mdp, Px, Nepochs=Npretrain),
	shared_params(name, π)...,
	kwargs...
)

vb_params(name, π; kwargs...) = (
	ΔN=20,
	training_buffer_size=3200*Nsteps_per_episode,
	a_opt=(optimizer=Flux.Optimiser(Flux.ClipValue(1f0), Adam(3f-4)), batch_size=1024),
	c_opt=(epochs=20, batch_size=1024, optimizer=Flux.Optimiser(Flux.ClipValue(1f0), Adam(3f-4))), 
	agent_pretrain=pretrain_value(mdp, Px, target=0.1, Nepochs=Npretrain),
	shared_params(name, π)...,
	kwargs...
)
					  

## Get the ground truth comparison
# data = experiment_setup(;mdp, Ntrials, dir)(()->MCSolver(;mc_params(Neps_gt)...), "gt")
# gt = mean(data[:est])[end]
# gt_std = std(data[:est])[end]

# Ground truth experiment
# D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps = Neps)
# vals = D[:r][:][D[:done][:] .== 1]
# histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# sum(D[:r] .> π/4) / sum(D[:done][:] .== 1)

gt = 2.5333333333333334e-5
gt_std = 4.163331998932266e-6
plot_init = ()->plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))

# Create our "run_experiment function"
run_experiment = experiment_setup(;mdp, Ntrials, dir, plot_init)

## Experiments
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
	(()->ValueBasedIS(;vb_params("VB_nopretrain", Q())..., agent_pretrain=nothing),  "VB_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB", Q())...),  "VB"),
	(()->ValueBasedIS(;vb_params("VB_defensive", MISPolicy([Q(), Px]))...), "VB_defensive"),
	(()->ValueBasedIS(;vb_params("VB_MIS2_nopretrain", MISPolicy([Q(), Q()]))..., agent_pretrain=nothing),  "VB_MIS2_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB_MIS2", MISPolicy([Q(), Q()]))...),  "VB_MIS2"),
	(()->ValueBasedIS(;vb_params("VB_MIS2_defensive", MISPolicy([Q(), Q(), Px]))...),  "VB_MIS2_defensive"),
	(()->ValueBasedIS(;vb_params("VB_MIS4_nopretrain", MISPolicy([Q(), Q(), Q(), Q()]))..., agent_pretrain=nothing),  "VB_MIS4_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB_MIS4", MISPolicy([Q(), Q(), Q(), Q()]))...),  "VB_MIS4"),
	(()->ValueBasedIS(;vb_params("VB_MIS4_defensive", MISPolicy([Q(), Q(), Q(), Q(), Px]))...),  "VB_MIS4_defensive")
]

if test
	for (𝒮fn, name) in standard_exps; run_experiment(𝒮fn, name); end
else
	Threads.@threads for (𝒮fn, name) in shuffle(standard_exps); run_experiment(𝒮fn, name); end
end

## Quick test:
# 𝒮 = PolicyGradientIS(;pg_params("PG", Π())...)
# 𝒮 = PolicyGradientIS(;pg_params("PG_MIS2", MISPolicy([Π(), Π()]))...)
# fs, ws = solve(𝒮, mdp)
# plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))
# plot!(cumsum(fs .* ws) ./ (1:length(ws)))

