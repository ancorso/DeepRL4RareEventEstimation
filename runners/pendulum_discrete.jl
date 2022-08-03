using Crux, Distributions
include("utils.jl")
include("../environments/pendulum_problem.jl")

## Parameters
# Experiment params
Neps=100_000
Neps_gt=1_000_000
Ntrials=5
dir="results/pendulum_discrete_v2/"

# Problem setup params
failure_target=π/4
dt = 0.1
Nsteps_per_episode = 20
noise_dist=Normal(0, 0.4)
Px, mdp = gen_topple_mdp(px=noise_dist, Nsteps=Nsteps_per_episode, dt=dt, failure_thresh=failure_target)
xs = Px.distribution.support
S = state_space(mdp)
A = action_space(Px)

# Networks
net(out_act...;Nout=A.N) = Chain(Dense(3, 32, relu), Dense(32, 32, relu), Dense(32, Nout, out_act...)) # Basic architecture
V() = ContinuousNetwork(net(sigmoid, Nout=1)) # Value network
Π() = DiscreteNetwork(net(), xs, always_stochastic=true) # Actor network (Policy gradient)
AC() = ActorCritic(Π(), V()) # Actor-crtic for baseline approaches
Q() = DiscreteNetwork(net(sigmoid, Nout=A.N), xs, always_stochastic=true, logit_conversion=estimator_logits(Px)) # Q network for value-based


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
										  training_buffer_size=200*Nsteps_per_episode,
										  c_opt=(;max_batches=1000),
										  agent_pretrain=use_baseline ? pretrain_AV(mdp, Px, v_target=0.1) : pretrain_policy(mdp, Px),
										  shared_params(name, π)...)
vb_params(name, π) = (ΔN=4,
					  training_buffer_size=40000, 
					  c_opt=(epochs=20,), 
					  agent_pretrain=pretrain_value(mdp, Px, target=0.1),
					  shared_params(name, π)...)


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

## Experiments: Compare a bunch of different variance reduction techniques
run_experiment(()->MCSolver(;mc_params()...), "MC")

run_experiment(()->PolicyGradientIS(;pg_params("PG_nopretrain", Π())..., agent_pretrain=nothing), "PG_nopretrain")
run_experiment(()->PolicyGradientIS(;pg_params("PG", Π())...), "PG")
run_experiment(()->PolicyGradientIS(;pg_params("PG_baseline", AC(), true)...), "PG_baseline")
run_experiment(()->PolicyGradientIS(;pg_params("PG_defensive", MISPolicy([Π(), Px], [3, 1]))...), "PG_defensive")
run_experiment(()->PolicyGradientIS(;pg_params("PG_defensive_baseline", MISPolicy([AC(), Px], [3, 1]), true)...), "PG_defensive_baseline")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS", MISPolicy([Π(), Π(), Px], [1, 1, 1]))...), "PG_MIS")


run_experiment(()->ValueBasedIS(;vb_params("VB_nopretrain", Q())..., agent_pretrain=nothing),  "VB_nopretrain")
run_experiment(()->ValueBasedIS(;vb_params("VB", Q())...),  "VB")
run_experiment(()->ValueBasedIS(;vb_params("VB_defensive", MISPolicy([Q(), Px], [3, 1]))...), "VB_defensive")
run_experiment(()->ValueBasedIS(;vb_params("VB_MIS", MISPolicy([Q(), Q(), Px], [1, 1, 1]))...),  "VB_MIS")