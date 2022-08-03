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
<<<<<<< HEAD
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS", MISPolicy([Π(), Π(), Π(), Px], [1, 1, 1, 1]))...), "PG_MIS")
=======
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS", MISPolicy([Π(), Π(), Px], [1, 1, 1]))...), "PG_MIS")
>>>>>>> reseting to remove big files

run_experiment(()->ValueBasedIS(;vb_params("VB_nopretrain", Q())..., agent_pretrain=nothing),  "VB_nopretrain")
run_experiment(()->ValueBasedIS(;vb_params("VB", Q())...),  "VB")
run_experiment(()->ValueBasedIS(;vb_params("VB_defensive", MISPolicy([Q(), Px], [3, 1]))...), "VB_defensive")
<<<<<<< HEAD
run_experiment(()->ValueBasedIS(;vb_params("VB_MIS", MISPolicy([Q(), Q(), Q(), Px], [1, 1, 1, 1]))...),  "VB_MIS")
=======
run_experiment(()->ValueBasedIS(;vb_params("VB_MIS", MISPolicy([Q(), Q(), Px], [1, 1, 1]))...),  "VB_MIS")

## Stuff to generate intro figures
using BSON, Plots, Distributions
pgfplotsx()


MC_data = BSON.load("results/pendulum_discrete_v2/MC.bson")[:data]
PG_baseline_data = BSON.load("results/pendulum_discrete_v2/PG_baseline.bson")[:data]

Neps = length(MC_data[:ws][1])
indices = 1:100:100000
plot([1,Neps], x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001), label="Ground Truth", xlabel="Episodes", ylabel="Rare Event Probability", color_palette=:Dark2_3, legend=:topright, xticks=[5e4, 1e5], yticks=[0, 5e-5, 1e-4])
plot!(collect(1:Neps)[indices], mean(MC_data[:est])[indices], ribbon=std(MC_data[:est])[indices], label="MC")
plot!(collect(1:Neps)[indices], mean(PG_baseline_data[:est])[indices], ribbon=std(PG_baseline_data[:est])[indices], label="PG-AIS (ours)")


savefig("comparison.tex")

# Train one:

𝒮_ppo = DQN(;π=DiscreteNetwork(net(Nout=A.N), xs), S=S, ΔN=4, N=100000)
solve(𝒮_ppo, mdp)

𝒮_pgais = PolicyGradientIS(;pg_params("PG_baseline", AC(), true)...)
solve(𝒮_pgais, mdp)


D_pgais = ExperienceBuffer(episodes!(Sampler(mdp, 𝒮_pgais.agent.π, episode_checker=(b, s, e) -> sum(b[:r][1,s:e]) > π/4), Neps=50))

D_ppo = ExperienceBuffer(episodes!(Sampler(mdp, 𝒮_ppo.agent.π, episode_checker=(b, s, e) -> sum(b[:r][1,s:e]) > π/4), Neps=50))



p = plot(xlabel="Time", ylabel="Angle", color_palette=:Dark2_3)
hline!([π/4, -π/4], color=:black, linestyle=:dash, label="Failure Threshold", legend=:topleft)

epis_ppo = episodes(D_ppo)
isfirst=true
for ep in epis_ppo
	plot!(D_ppo[:s][1,ep[1]:ep[2]], D_ppo[:s][2,ep[1]:ep[2]], label=isfirst ? "AST" : "", alpha=0.3, color=2)
	isfirst=false
end
epis_pgais = episodes(D_pgais)
isfirst=true
for ep in epis_pgais
	plot!(D_pgais[:s][1,ep[1]:ep[2]], D_pgais[:s][2,ep[1]:ep[2]], label=isfirst ? "PG-AIS (ours)" : "", alpha=0.3, color=3)
	isfirst=false
end
p

savefig("estimation.tex")

epis_pgais = episodes(D_pgais)
isfirst=true
for ep in epis_pgais
	plot!(D_pgais[:s][1,ep[1]:ep[2]], D_pgais[:s][2,ep[1]:ep[2]], label=isfirst ? "PG-AIS (ours)" : "", alpha=0.3, color=3)
	isfirst=false
end
p

savefig("estimation.tex")
>>>>>>> reseting to remove big files

