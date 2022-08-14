using Crux, Distributions, LinearAlgebra, Plots, DistributionsAD
include("utils.jl")
include("../environments/T-intersection_problem.jl")

## false
test = true

# Experiment params
if test
	Neps=500
	Ntrials=1
	Neps_gt=100
	Npretrain=1
else
	Neps=50_000
	Neps_gt=1_000_000
	Ntrials=3
	Npretrain=100
end
dir="results/t_intersection"

# Problem setup params
failure_target = 0.99f0
Px, mdp = gen_T_intersection_problem()

# Crux.gif(mdp, Px, "out.gif", Neps=10)

# D = episodes!(Sampler(mdp, Px, max_steps=100), Neps=1000)
# vals = D[:r][:][D[:done][:] .== 1]
# histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# sum(vals .>= 1f0)


S = state_space(mdp, μ=Float32[0.22, 25.0, -1, 1.5, 1.0, 16.0, 5.5, 1.0, 0.0], σ=Float32[.15f0, .56, 3.0, 0.2, 0.3, 7, 6, 1f0, 1f0])
A = action_space(Px)
Nsteps_per_episode=30

## Networks
function net(out_act...; Nin=S.dims[1], Nout=1, Nhiddens=[64, 32], act=tanh)
    hiddens = [Dense(Nhiddens[idx], Nhiddens[idx+1], relu) for idx in 1:length(Nhiddens)-1]
    Chain(Dense(Nin, Nhiddens[begin], act), hiddens..., Dense(Nhiddens[end], Nout, out_act...)) # Basic architecture
end

# Networks for on-policy algorithms
function Π(Nhiddens=[64, 32])
	base = net(Nout=32, Nhiddens=Nhiddens)
	μ = ContinuousNetwork(Chain(base..., Dense(32, A.dims[1])))
	logΣ = ContinuousNetwork(Chain(base..., Dense(32, A.dims[1])))
	GaussianPolicy(μ, logΣ, true)
end
V(Nhiddens=[64, 32]) = ContinuousNetwork(net(sigmoid; Nhiddens))
AC(Nhiddens=[64, 32]) = ActorCritic(Π(Nhiddens), V(Nhiddens))

# Networks for off-policy algorithms
QSA(Nhiddens=[256,256]) = ContinuousNetwork(net(sigmoid, Nin=S.dims[1] + A.dims[1], Nhiddens=Nhiddens, act=relu))
AQ(Nhiddens=[256,256]) = ActorCritic(Π(Nhiddens), QSA(Nhiddens))

function Π_CEM(Npolicies; d=()->product_distribution([Normal(rand(Distributions.Uniform(-0.1, 0.1)), 1.0) for _=1:A.dims[1]]))
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
	max_steps=Nsteps_per_episode,
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
	a_opt=(;optimizer=Flux.Optimiser(Flux.ClipValue(1f0), Adam(3f-4)), batch_size=200*60),
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
	a_opt=(optimizer=Flux.Optimiser(Flux.ClipValue(1f0), Adam(3f-4)), batch_size=1024),
	c_opt=(epochs=10, ), 
	agent_pretrain=pretrain_AQ(mdp, Px, v_target=0.1, Nepochs=Npretrain),
	shared_params(name, π)...,
	kwargs...
)
					  

## Get the ground truth comparison
# data = experiment_setup(;mdp, Ntrials=1, dir)(()->MCSolver(;mc_params(Neps_gt)...), "gt")
# gt = mean(data[:est])[end]
# gt_std = std(data[:est])[end]

# Ground truth experiment
# D = episodes!(Sampler(mdp, Px), Neps=Neps_gt)
# vals = D[:r][:][D[:done][:] .== 1]
# # histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# gt = sum(D[:r] .> failure_target) / sum(D[:done][:] .== 1)
# io = open("gt_tintersection.txt", "w")
# write(io, string(gt))
# close(io)

gt = 2.5333333333333334e-5
# gt_std = 4.163331998932266e-6
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
	(()->ValueBasedIS(;vb_params("VB_nopretrain", AQ())..., agent_pretrain=nothing),  "VB_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB", AQ())...),  "VB"),
	(()->ValueBasedIS(;vb_params("VB_defensive", MISPolicy([AQ(), Px]))...), "VB_defensive"),
	(()->ValueBasedIS(;vb_params("VB_MIS2_nopretrain", MISPolicy([AQ(), AQ()]))..., agent_pretrain=nothing),  "VB_MIS2_nopretrain"),
	(()->ValueBasedIS(;vb_params("VB_MIS2", MISPolicy([AQ(), AQ()]))...),  "VB_MIS2"),
	(()->ValueBasedIS(;vb_params("VB_MIS2_defensive", MISPolicy([AQ(), AQ(), Px]))...),  "VB_MIS2_defensive"),
]

if test
	for (𝒮fn, name) in standard_exps; run_experiment(𝒮fn, name); end
else
	Threads.@threads for (𝒮fn, name) in shuffle(standard_exps); run_experiment(𝒮fn, name); end
end

## Quick test:
# 𝒮 = MCSolver(;mc_params()...)
# 𝒮 = CEMSolver(;cem_params("CEM_1", Π_CEM(1))...)
# 𝒮 = CEMSolver(;cem_params("CEM_2", Π_CEM(2))...)
# 𝒮 = PolicyGradientIS(;pg_params("PG", Π())...)
# 𝒮 = PolicyGradientIS(;pg_params("PG_MIS2", MISPolicy([Π(), Π()]))...)
# 𝒮 = ValueBasedIS(;vb_params("VB", AQ())...)
# 
# fs, ws = solve(𝒮, mdp)
# plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))
# plot!(cumsum(fs .* ws) ./ (1:length(ws)))

