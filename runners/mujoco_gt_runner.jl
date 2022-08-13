using Crux, Distributions
include("utils.jl")
include("../environments/mujoco_problem.jl")

## Parameters
# Experiment params
Neps_gt=1_000_000
Ntrials=5

# Problem setup params
dt = 0.1
Nsteps_per_episode = 50
xscale=0.04f0
failure_target=200f0

Px, mdp = gen_halfcheetah(;dt, Nsteps=Nsteps_per_episode, xscale)
S = state_space(mdp)
A = action_space(Px)
dir="results/halfcheetah"

# Ground truth experiment
# D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps=1000)
# vals = D[:r][:][D[:done][:] .== 1]
# histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# sum(D[:r] .> π/4) / sum(D[:done][:] .== 1)
# 
# Crux.gif(mdp, Px, "out.gif", Neps=1, max_steps=50)

mc_params(N=Neps) = (
	agent=PolicyParams(;π=Px, pa=Px), 
	N=N, 
	S, 
	buffer_size=N*Nsteps_per_episode, 
	𝒫=(;f_target=failure_target)
)
			 
## Ground truth
data = experiment_setup(;mdp, Ntrials, dir)(()->MCSolver(;mc_params(Neps_gt)...), "gt")
gt = mean(data[:est])[end]
gt_std = std(data[:est])[end]

println("gt: $gt, std: $std")

