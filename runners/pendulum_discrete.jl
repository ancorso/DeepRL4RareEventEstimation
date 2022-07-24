using Crux, Distributions
include("utils.jl")
include("../environments/pendulum_problem.jl")

## Parameters
# Experiment params
Neps=10000
Neps_gt=1_000_000
Ntrials=3
dir="results/pendulum_discrete/"

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
V() = ContinuousNetwork(net(softplus, Nout=1)) # Value network
Π() = DiscreteNetwork(net(), xs, always_stochastic=true, logit_conversion=policy_match_logits(Px)) # Actor network (Policy gradient)
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
mc_params = (agent=PolicyParams(;π=Px, pa=Px), N=Neps, S, buffer_size=Nbuff, 𝒫=(;f_target=failure_target))
pg_params(name, π, use_baseline=false) = (ΔN=200, use_baseline, shared_params(name, π)...)
vb_params(name, π) = (ΔN=4, N_elite_candidate=100, c_opt=(epochs=4*Nsteps_per_episode,), shared_params(name, π)...)


## Experiments 

# Ground truth experiment
# D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps = Neps)
# vals = D[:r][:][D[:done][:] .== 1]
# histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# sum(D[:r] .> π/4) / sum(D[:done][:] .== 1)

gt = 2.5333333333333334e-5
gt_std = 4.163331998932266e-6


run_experiment(()->MCSolver(;mc_params...), mdp, Ntrials, gt, dir, "MC")
run_experiment(()->PolicyGradientIS(;pg_params("PG", Π())...), mdp, Ntrials, gt, dir, "PG")
run_experiment(()->PolicyGradientIS(;pg_params("PG_baseline", AC(), true)...), mdp, Ntrials, gt, dir, "PG_baseline")
run_experiment(()->PolicyGradientIS(;pg_params("defensive", MISPolicy([Π(), Px], [10, 1]))...), mdp, Ntrials, gt, dir, "defensive")
# run_experiment(()->PolicyGradientIS(;pg_params("defensive_baseline", MISPolicy([Π(), Px], [10, 1]))...), mdp, Ntrials, gt, dir, "defensive")
run_experiment(()->PolicyGradientIS(;pg_params("MIS", MISPolicy([Π(), Π(), Π(), Π()], [1, 1, 1, 1]))...), mdp, Ntrials, gt, dir, "MIS")
run_experiment(()->ValueBasedIS(;vb_params("VB", Q())...), mdp, Ntrials, gt, dir, "VB")

