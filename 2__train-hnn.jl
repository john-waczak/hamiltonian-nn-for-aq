using DifferentialEquations
using DiffEqFlux
using SciMLSensitivity
using Flux, ReverseDiff
using CSV, DataFrames # , DTables
using StableRNGs, Random
using Plots
using StatsBase, Statistics
using ProgressMeter

include("nn_models.jl")


rng = StableRNG(42)  # for reproducible random number generation

# -----------------------------------------------------------
# 1. Collect Data
# -----------------------------------------------------------

function get_data(path)
    df = CSV.File(path) |> DataFrame

    cols_now = [name for name ∈ names(df) if !(name ∈ ["ts", "dateTime", "device_name"]) && !occursin("next", name)]
    cols_next = [name for name ∈ names(df) if occursin("next", name)]

    times = df.dateTime
    Xnow = Float32.(collect(Matrix(df[:, cols_now])'))
    Xnext = Float32.(collect(Matrix(df[:,cols_next])'))

    return times, Xnow, Xnext
end


basepath = "data/combined"
times, Xnow, Xnext = get_data(joinpath(basepath, "central-node-1_Nwindow-2.csv"));

# set up data loader for mini-batching and shuffling
dataloader = Flux.Data.DataLoader((Xnow, Xnext); batchsize=256, shuffle=true)


# -----------------------------------------------------------
# 2. Define Models
# -----------------------------------------------------------


# now define the actual encoder model
encoder = Encoder(
    Flux.Chain(
        Dense(80, 40, tanh),
        Dense(40, 20, tanh),
        Dense(20, 10, tanh),
        Dense(10, 2)
    )
)

p_enc = encoder.p
# test encoder on first record
encoder(Xnow[:,1], p_enc)


decoder = Decoder(
    Flux.Chain(
        Dense(2, 10, tanh),
        Dense(10, 20, tanh),
        Dense(20, 40, tanh),
        Dense(40, 80)
    )
)


p_dec = decoder.p
# test decoder on simulated pq vals
decoder(rand(Float32, 2), p_dec)


hamiltonian = Flux.Chain(
    Dense(2,200,tanh),
    Dense(200,1)
)

hamiltonian(rand(Float32, 2))

hnn = HamiltonianNN(
    hamiltonian
)

p_hnn = hnn.p

hnn(rand(Float32, 2), p_hnn)


params = [p_enc..., p_dec..., p_hnn...]

Np_enc = length(p_enc)
Np_dec = length(p_dec)

@assert all(params[1:Np_enc] .== p_enc)
@assert all(params[Np_enc+1:Np_enc+Np_dec] .== p_dec)
@assert all(params[Np_enc+ Np_dec + 1:end] .== p_hnn)

# -----------------------------------------------------------
# 3. Define Hamiltonain ODE Problem
# -----------------------------------------------------------

Δt = 15.0f0
integrate_forward = NeuralHamiltonianDE(
    hnn, (0.0f0, Δt),
    Tsit5(),
    save_everystep = false,
    save_start = false,
    save_end=true,
    sensealg=ReverseDiffAdjoint()
)

# integrate forward takes a set of initial conditions p₀q₀ and
# integrates them forward by Δt

# this could be problematic... need to make sure this is
# doing what we intend and actually integrating each column separately. 
@time Array(integrate_forward(rand(Float32, 2,10), p_hnn))



function loss(x_now, x_next, params)
    p_enc = params[1:Np_enc]
    p_dec = params[Np_enc+1:Np_enc+Np_dec]
    p_hnn = params[Np_enc+ Np_dec + 1:end]


    # compute generalized coordinates from encoder
    qp = encoder(x_now, p_enc)
    qp_next = encoder(x_next, p_enc)

    # compute the auto-encoder loss
    x̂_now = decoder(qp, p_dec)
    ℓ_ae = mean((x_now .- x̂_now).^2)

    # now integrate our coordinates forward using
    # Hamilton's equations
    qp_forward = Array(integrate_forward(qp, p_hnn))
    ℓ_hnn = mean((qp_forward .- qp_next).^2)

    # finally compute a coordinate loss that tries
    # to force the momentum to be related to
    # the velocity

    q_now = qp[1]
    p_now = qp[2]

    q_next = qp_next[1]
    p_next = qp_next[2]

    ℓ_coord = mean((p_now .- (q_next .- q_now)).^2)

    # include knob for hnn loss
    return ℓ_ae + ℓ_coord + 0.1*ℓ_hnn
end


# test loss function and gradient calculation
loss(Xnow[:,1:5], Xnext[:,1:5], params)
ReverseDiff.gradient(p -> loss(Xnow[:,1:5], Xnext[:,1:5], p), params)


# -----------------------------------------------------------
# 4. Perform Training Loop
# -----------------------------------------------------------
callback() = println("Loss Neural Hamiltonian DE = $(loss(Xnow[:,1:500:end], Xnext[:,1:500:end], params))")

@time callback()

dataloader

opt = ADAM(0.01)

epochs = 10
for epoch in 1:epochs
    println("epoch: ", epoch)

    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), params)
        Flux.Optimise.update!(opt, params, gs)
        callback()
    end
    # if epoch % 100 == 1
    #     callback()
    # end
    # callback()
end
callback()

