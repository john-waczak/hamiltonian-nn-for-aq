using DifferentialEquations
using DiffEqFlux
using SciMLSensitivity
using Flux, ReverseDiff
using CSV, DataFrames # , DTables
using StableRNGs, Random
using Plots
using StatsBase, Statistics
using ProgressMeter
using Dates




include("nn_models.jl")


rng = StableRNG(42)  # for reproducible random number generation

# -----------------------------------------------------------
# 1. Collect Data
# -----------------------------------------------------------


function standard_scale(X::AbstractMatrix, μ::AbstractMatrix, σ::AbstractMatrix)
    X̃ = similar(X)
    for j ∈ axes(X, 2)
        X̃[:,j] .= (X[:,j] .- μ) ./ σ
    end
    return X̃
end

function inv_scale(X̃::AbstractMatrix, μ::AbstractMatrix, σ::AbstractMatrix)
    X = similar(X̃)
    for j ∈ axes(X̃, 2)
        X[:,j] .= (X̃[:,j] .* σ) .+ μ
    end
    return X
end


function get_data(path; npoints=10_000, npoints_test=500)
    df = CSV.File(path) |> DataFrame

    names_to_use = [n for n ∈ names(df) if !occursin("pc", n)]

    df = df[year.(df.dateTime) .== 2022, names_to_use]

    cols_now = [name for name ∈ names(df) if !(name ∈ ["ts", "dateTime", "device_name"]) && !occursin("next", name)]
    cols_next = [name for name ∈ names(df) if occursin("next", name)]

    for i ∈ 1:length(cols_now)
        println(i, "\t", cols_now[i])
    end

    times = df.dateTime
    Xnow = Float32.(collect(Matrix(df[:, cols_now])'))
    Xnext = Float32.(collect(Matrix(df[:,cols_next])'))

    # compute mean and std rowwise
    μ = mean(Xnow, dims=2)
    σ = std(Xnow, dims=2)

    Xnow = standard_scale(Xnow, μ, σ)
    Xnext = standard_scale(Xnext, μ, σ)

    idxs = shuffle(1:nrow(df))

    idxs_train = idxs[1:npoints]
    idxs_test = idxs[npoints+1: npoints+npoints_test]
    return times[idxs_train], Xnow[:, idxs_train], Xnext[:, idxs_train], times[idxs_test], Xnow[:, idxs_test], Xnext[:, idxs_test], μ, σ
end

basepath = "data/combined"
nwindow = 2
times_train, Xnow_train, Xnext_train, times_test, Xnow_test , Xnext_test, μ, σ = get_data(joinpath(basepath, "central-node-1_Nwindow-$(nwindow).csv"));



size(times_train)
size(times_test)

# set up data loader for mini-batching and shuffling
dataloader = Flux.Data.DataLoader((Xnow_train, Xnext_train); batchsize=64, shuffle=true)


# -----------------------------------------------------------
# 2. Define Models
# -----------------------------------------------------------
input_dim = size(Xnow_test, 1)
hidden_dim = 50
n_manifold = 1
output_dim = 2*n_manifold

# now define the actual encoder model
encoder = Encoder(
    Flux.Chain(
        Dense(input_dim, hidden_dim, tanh; init=Flux.orthogonal),
        Dense(hidden_dim, hidden_dim, tanh; init=Flux.orthogonal),
        Dense(hidden_dim, hidden_dim, tanh; init=Flux.orthogonal),
        Dense(hidden_dim, output_dim; init=Flux.orthogonal)
    )
)

p_enc = encoder.p
# test encoder on first record
encoder(Xnow_train[:,1], p_enc)


decoder = Decoder(
    Flux.Chain(
        Dense(output_dim, hidden_dim, tanh; init=Flux.orthogonal),
        Dense(hidden_dim, hidden_dim, tanh; init=Flux.orthogonal),
        Dense(hidden_dim, hidden_dim, tanh; init=Flux.orthogonal),
        Dense(hidden_dim, input_dim; init=Flux.orthogonal)
    )
)


p_dec = decoder.p
# test decoder on simulated pq vals
decoder(rand(Float32, output_dim), p_dec)



hamiltonian = Flux.Chain(
    Dense(output_dim,hidden_dim, tanh; init=Flux.orthogonal),
    Dense(hidden_dim,1; init=Flux.orthogonal)
)


hamiltonian(rand(Float32, output_dim))

hnn = HamiltonianNN(
    hamiltonian
)

p_hnn = hnn.p

hnn(rand(Float32, output_dim), p_hnn)


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
@time Array(integrate_forward(rand(Float32, output_dim,10), p_hnn))


function loss(x_now, x_next, params)
    # x_now = Xnow_train[:,1:10]
    # x_next = Xnext_train[:,1:10]

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
    # simple hnn loss estimate
    #ℓ_hnn = mean((hnn(qp, p_hnn) .- (qp_next .- qp)./15.0).^2)


    # finally compute a coordinate loss that tries
    # to force the momentum to be related to
    # the velocity

    q_now = qp[1:n_manifold,:]
    p_now = qp[n_mHamilton's equations
    qp_forward = Array(integrate_forward(qp, p_hnn))
    ℓ_hnn = mean((qp_forward .- qp_next).^2)
    return ℓ_hnn
end



# test loss function and gradient calculation
loss(Xnow_train[:,1:5], Xnext_train[:,1:5], params)
ReverseDiff.gradient(p -> loss(Xnow_train[:,1:16], Xnext_train[:,1:16], p), params)


# maybe the problem is the batch size is too big

# -----------------------------------------------------------
# 4. Perform Training Loop
# -----------------------------------------------------------
losses = []
function callback()
    ℓ = loss(Xnow_test, Xnext_test, params)
    println("Loss Neural Hamiltonian DE = $(ℓ)")
    push!(losses, ℓ)
end


@time callback()

opt = ADAM(0.001)  # we should set the decay rate too

epochs = 1
for epoch in 1:epochs
    println("epoch: ", epoch)

    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), params)
        Flux.Optimise.update!(opt, params, gs)
        callback()
    end
end
callback()


# do second loop with just hnn loss

opt = ADAM(0.005)  # we should set the decay rate too

epochs = 1
for epoch in 1:epochs
    println("epoch: ", epoch)

    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss2(x, y, p), params)
        Flux.Optimise.update!(opt, params, gs)
        callback()
    end
end
callback()



plot(losses, ylabel="loss", xlabel="iteration", label="", lw=3)


# now let's visualize the Hamiltonian
p

Xfinal = hcat(Xnow_train, Xnow_test)

qps = encoder(Xfinal)

q_min = minimum(qps[1,:])
q_max = maximum(qps[1,:])
p_min = minimum(qps[2,:])
p_max = maximum(qps[2,:])

q_min
q_max

npoints = 500
qs = Float32.(range(3*q_min, stop=3*q_max, length=npoints))
ps = range(3 * p_min, stop=3*p_max, length=npoints)
Hs = [Float64(hamiltonian([q, p])[1]) for q ∈ qs, p ∈ ps]


p1 = contour(qs,
             ps,
             Hs,
             fill=true,
             #color=:black,
             clabels=true,
             xlabel="q",
             ylabel="p",
             title="fitted Hamiltonian",
             #legend=false
             )


savefig("./hamiltonian_cn_1--2022.pdf")
savefig("./hamiltonian_cn_1--2022.png")

# times_tot = [times_train..., times_test...]
# tstart = minimum(times_tot)
# tend = maximum(times_tot)

# times_as_ints = [(t - tstart).value for t ∈ times_tot]

# scatter!(
#     p1,
#     Xnow_test[1,1:10],
#     Xnow_test[2,1:10],
#     zcolor=times_as_ints[1:10]
# )




# times_test
# for some reason the parameters aren't updating.
