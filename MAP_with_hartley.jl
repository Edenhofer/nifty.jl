using ForwardDiff
using FFTW

function hartley(u::Vector{T}) where T<:Real
	v = fft(real.(u))
	return real.(v) .+ imag.(v)
end

function hartley(u::Vector{T}) where T<:Complex
	v_r = fft(real.(u))
	v_i = fft(imag.(u))
	return real.(v_r) .+ imag.(v_r) .+ (real.(v_i) .+ imag.(v_i)) * im
end

function hartley!(o::Vector{T}, u::Vector{T}) where T<:Real
	v = fft(real.(u))
	o .= real.(v) .+ imag.(v)
end

function hartley!(o::Vector{T}, u::Vector{T}) where T<:Complex
	v_r = fft(real.(u))
	v_i = fft(imag.(u))
	o .= real.(v_r) .+ imag.(v_r) .+ (real.(v_i) .+ imag.(v_i)) * im
end

function hartley(u::Vector{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
	# Unpack AoS -> SoA
	vs = vec(ForwardDiff.value.(u))
	ps = vec(mapreduce(ForwardDiff.partials, hcat, u))
	# Actual computations
	val = hartley(vs)
	jvp = hartley(ps)
	# Pack SoA -> AoS (depending on jvp, might need `eachrow`)
	return map((v, p) -> ForwardDiff.Dual{T}(v, p...), val, jvp)
end


import IterativeSolvers: cg
import Random: randn
import ForwardDiff
using LinearMaps
using LinearAlgebra
using Optim
using Zygote
using Plots

dims = (1024)

# ξ := latent variables
# Get hermitian symmetry in Fourier space
ξ_truth = randn(dims)
k = [i < dims / 2 ? i :  dims-i for i = 0:dims-1]

power = @. 50 / (k^2.5 + 1)
# Modulo some factors hartley = ihartley
ihartley(u) = 1 ./ dims .* hartley(u)
draw_sample(ξ) = real(ihartley(power .* hartley(ξ)))
signal(ξ) = exp.(draw_sample(ξ))

ss = signal(ξ_truth)

plot(ss, color=:red, label="ground truth", linewidt=5)

N = Diagonal(0.01^2 * ones(dims))
R = ones(dims)
R[10:20] .= 0
R = Diagonal(R)
d = R * ss .+ R * sqrt(N) * randn(dims)

plot!(d, seriestype=:scatter, marker=:x, color=:black)

signal_response(ξ) = R * signal(ξ)
nll(ξ) = sum(inv(N) * (d .- signal_response(ξ)).^2)
ham(ξ) = nll(ξ) + sum(ξ.^2)

function ∂ham!(ξ_storage::T, ξ::T) where T
	ξ_storage .= first(Zygote.gradient(ham, ξ))
end

init_pos = 0.1 * randn(dims)
opt = optimize(ham, ∂ham!, init_pos, LBFGS())
opt_min = Optim.minimizer(opt)
plot!(signal(opt_min), label="reconstruction", color=:orange)
