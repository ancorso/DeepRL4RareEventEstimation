using Crux, Flux, POMDPs, POMDPGym, Distributions
import POMDPPolicies:FunctionPolicy
using Random
init_mujoco_render() # Required for visualization

# Construct the Mujoco environment
noise_model = product_distribution([Normal(0, 0.05) for _=1:6])
# mdp = NoisyPOMDP(GymPOMDP(:HalfCheetah, version = :v3), action_noise_model=noise_model)
mdp = GymPOMDP(:HalfCheetah, version = :v3)
S = state_space(mdp)
adim = length(POMDPs.actions(mdp)[1])
amin = -1*ones(Float32, adim)
amax = 1*ones(Float32, adim)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

# Initializations that match the default PyTorch initializations
Winit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out, in))
binit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))

# Build the networks
idim = S.dims[1] + adim

# Networks for off-policy algorithms
Q() = ContinuousNetwork(Chain(Dense(idim, 256, relu, init=Winit, bias=binit(256, idim)), 
            Dense(256, 256, relu, init=Winit, bias=binit(256, 256)), 
            Dense(256, 1, init=Winit, bias=binit(1,256)))) 
A() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu, init=Winit, bias=binit(256, S.dims[1])), 
            Dense(256, 256, relu, init=Winit, bias=binit(256, 256)), 
            Dense(256, 6, tanh, init=Winit, bias=binit(6, 256)), (x)->x *0.9f0), 6)
            
Π() = ActorCritic(A(), DoubleNetwork(Q(), Q()))

function SAC_A()
    base = Chain(Dense(S.dims[1], 256, relu, init=Winit, bias=binit(256, S.dims[1])), 
                Dense(256, 256, relu, init=Winit, bias=binit(256, 256)))
    mu = ContinuousNetwork(Chain(base..., Dense(256, 6, init=Winit, bias=binit(6, 256))))
    logΣ = ContinuousNetwork(Chain(base..., Dense(256, 6, init=Winit, bias=binit(6, 256))))
    SquashedGaussianPolicy(mu, logΣ)
end

## Setup params
params =     (ΔN=50,
              S=S,
              max_steps=1000,
              N=Int(3e6),
              log=(period=4000, fns=[log_undiscounted_return(3)]),
              buffer_size=Int(1e6), 
              buffer_init=1000, 
              c_opt=(batch_size=100, optimizer=Adam(1e-3)),
              a_opt=(batch_size=100, optimizer=Adam(1e-3)), 
              π_explore=FirstExplorePolicy(10000, rand_policy, GaussianNoiseExplorationPolicy(0.1f0, a_min=amin, a_max=amax)),
              π_smooth = GaussianNoiseExplorationPolicy(0.2f0, ϵ_min=-0.5f0, ϵ_max=0.5f0, a_min=amin, a_max=amax))

# Solve with TD3
𝒮_td3 = TD3(;π=Π(), params...,)
solve(𝒮_td3, mdp)

policy = 𝒮_td3.agent.π

using BSON
BSON.@save "environments/halfcheetah_policy_noisy.bson" policy

undiscounted_return(Sampler(mdp, policy, max_steps=1000), Neps=10)

