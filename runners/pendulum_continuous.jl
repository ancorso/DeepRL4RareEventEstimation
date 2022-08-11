using Crux, Distributions
include("utils.jl")
include("../environments/pendulum_problem.jl")

## Parameters
# Experiment params
Neps=50_0
Neps_gt=10_000_00
Ntrials=1
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
net(out_act...;Nin=3, Nout=1) = Chain(Dense(Nin, 32, relu), Dense(32, 32, relu), Dense(32, Nout, out_act...)) # Basic architecture
V() = ContinuousNetwork(net(sigmoid)) # Value network
function Π()
	base = net(Nout=4)
	μ = ContinuousNetwork(Chain(base..., Dense(4, 1)))
	logΣ = ContinuousNetwork(Chain(base..., Dense(4, 1)))
	GaussianPolicy(μ, logΣ, true)
end
function Π_mixture()
	base = net(Nout=4)
	
	mu1 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
    logΣ1 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
	
    mu2 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
    logΣ2 = ContinuousNetwork(Chain(base..., Dense(4, 1)))
	
    αs = ContinuousNetwork(Chain(base..., Dense(4, 2), softmax), 2)
    MixtureNetwork([GaussianPolicy(mu1, logΣ1, true), GaussianPolicy(mu2, logΣ2, true)], αs)
end
AC(A=Π()) = ActorCritic(A, V()) # Actor-crtic for baseline approaches
QSA() = ContinuousNetwork(net(sigmoid, Nin=4)) # Q network for value-based
AQ(A=Π()) = ActorCritic(A, QSA()) # Actor-critic for continuous value-based approach

# Solver parameters
Nbuff = Neps*Nsteps_per_episode
shared_params(name, π) = (agent=PolicyParams(;π, pa=Px), 
                          N=Neps, 
                          S,
                          f_target=failure_target,
                          buffer_size=Nbuff,
                          log=(dir="log/$name", period=1000))
						  
mc_params(N=Neps) = (agent=PolicyParams(;π=Px, pa=Px), 
			 N=N, 
			 S, 
			 buffer_size=N*Nsteps_per_episode, 
			 𝒫=(;f_target=failure_target))
			 
pg_params(name, π, use_baseline=false) = (ΔN=200, 
										  use_baseline,
										  c_opt=(;max_batches=200, batch_size=1024),
										  training_buffer_size=200*Nsteps_per_episode,
										  agent_pretrain=use_baseline ? pretrain_AV(mdp, Px, v_target=0.1) : pretrain_policy(mdp, Px),
										  shared_params(name, π)...)
vb_params(name, π) = (ΔN=20,
					  train_actor=true,
					  training_buffer_size=3200*Nsteps_per_episode,
					  c_opt=(epochs=20,), 
					  agent_pretrain=pretrain_AQ(mdp, Px, v_target=0.1),
					  xi,
					  wi,
					  shared_params(name, π)...)
					  


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

## Experiments: Compare a bunch of different variance reduction techniques
run_experiment(()->MCSolver(;mc_params()...), "MC")

run_experiment(()->PolicyGradientIS(;pg_params("PG_nopretrain", Π())..., agent_pretrain=nothing), "PG_nopretrain")
run_experiment(()->PolicyGradientIS(;pg_params("PG", Π())...), "PG")
run_experiment(()->PolicyGradientIS(;pg_params("PG_baseline", AC(), true)...), "PG_baseline")
run_experiment(()->PolicyGradientIS(;pg_params("PG_defensive", MISPolicy([Π(), Px]))...), "PG_defensive")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS2_nopretrain", MISPolicy([Π(), Π()]))..., agent_pretrain=nothing), "PG_MIS2_nopretrain")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS2", MISPolicy([Π(), Π()]))...), "PG_MIS2")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS2_defense", MISPolicy([Π(), Π(), Px]))...), "PG_MIS2_defense")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS2_baseline", MISPolicy([AC(), AC()]), true)...), "PG_MIS2_baseline")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS4_nopretrain", MISPolicy([Π(), Π(), Π(), Π()]))..., agent_pretrain=nothing), "PG_MIS4_nopretrain")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS4", MISPolicy([Π(), Π(), Π(), Π()]))...), "PG_MIS4")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS4_baseline", MISPolicy([AC(), AC(), AC(), AC()]), true)...), "PG_MIS4_baseline")
# run_experiment(()->PolicyGradientIS(;pg_params("PG_mixture", Π_mixture())...), "PG_mixture")
# run_experiment(()->PolicyGradientIS(;pg_params("PG_mixture_baseline", AC(Π_mixture()), true)...), "PG_mixture_baseline")
# 
run_experiment(()->ValueBasedIS(;vb_params("VB_nopretrain", AQ())..., agent_pretrain=nothing),  "VB_nopretrain")
run_experiment(()->ValueBasedIS(;vb_params("VB", AQ())...),  "VB")
run_experiment(()->ValueBasedIS(;vb_params("VB_defensive", MISPolicy([AQ(), Px]))...), "VB_defensive")
run_experiment(()->ValueBasedIS(;vb_params("VB_MIS2", MISPolicy([AQ(), AQ()]))...),  "VB_MIS2")
run_experiment(()->ValueBasedIS(;vb_params("VB_MIS4", MISPolicy([AQ(), AQ(), AQ(), AQ()]))...),  "VB_MIS4")
# run_experiment(()->ValueBasedIS(;vb_params("VB_mixture", MISPolicy([AQ(Π_mixture()), Px], [3, 1]))...),  "VB_mixture")

# function plot_results(fs, ws; p=plot(), kwargs...)
# 	est = cumsum(fs .* ws) ./ (1:length(fs))
# 	plot!(p, est; kwargs...)
# end
# 
# p = plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))
# S1 = PolicyGradientIS(;pg_params("PG", Π())...)
# fs1, ws1 = solve(S1, mdp)
# plot_results(fs1, ws1, label="PG", p=p)
# 
# S2 = PolicyGradientIS(;pg_params("PG baseline", AC(), true)...)
# fs2, ws2 = solve(S2, mdp)
# plot_results(fs2, ws2, label="PG baseline ", p=p)
# 
# 
# S3 = PolicyGradientIS(;pg_params("PG_MIS", MISPolicy([Π(), Π()], [0.5, 0.5]))...)
# fs3, ws3 = solve(S3, mdp)
# plot_results(fs3, ws3, label="PG_MIS", p=p)
# 
# 
# S4 = PolicyGradientIS(;pg_params("PG_MIS_defense", MISPolicy([Π(), Π(), Px], [1/3,1/3,1/3]))...)
# fs4, ws4 = solve(S4, mdp)
# plot_results(fs4, ws4, label="PG_MIS_defense", p=p)
# 
# 
# S5 = PolicyGradientIS(;pg_params("PG_MIS_baseline", MISV(), true)...)
# fs5, ws5 = solve(S5, mdp)
# plot_results(fs5, ws5, label="PG_MIS_baseline", p=p)
# 
# 
# S7 = ValueBasedIS(;vb_params("VB_MIS", MISQ())...)
# fs7, ws7 = solve(S7, mdp)
# plot_results(fs7, ws7, label="VB_MIS", p=p)
# 
# 
# S_VB = ValueBasedIS(;vb_params("VB", AQ())...)
# fs, ws, = solve(S_VB, mdp)
# 
# # TODO Things to look into more
# # The main issue with the value based method is that each policy doesn't want to specialize. Can we combine the actor losses to encourage specialization?
# # Can we include exploration for the value based method?
# 
# 
