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
	# Actual computation
	val = hartley(vs)
	jvp = hartley(ps)
	# Pack SoA -> AoS (depending on jvp, might need `eachrow`)
	return map((v, p) -> ForwardDiff.Dual{T}(v, p...), val, jvp)
end

# Modulo some factors hartley = ihartley
ihartley(u) = 1 ./ dims .* hartley(u)


import IterativeSolvers: cg
import Random: randn
import ForwardDiff
using LinearMaps
using LinearAlgebra
using Zygote
using Plots
using Statistics: mean

dims = (1024)

# ξ := latent variables
ξ_truth = randn(dims)
k = [i < dims / 2 ? i :  dims-i for i = 0:dims-1]

power = @. 50 / (k^2.5 + 1)
draw_sample(ξ) = real(ihartley(power .* hartley(ξ)))
signal(ξ) = exp.(draw_sample(ξ))

N = Diagonal(0.1^2 * ones(dims))
R = ones(dims)
R[100:200] .= 0
R = Diagonal(R)

# Generate synthetic signal and data
ss = signal(ξ_truth)
d = R * ss .+ R * sqrt(N) * randn(dims)
plot(ss, color=:red, label="ground truth", linewidt=5)
#plot!(d, seriestype=:scatter, marker=:x, color=:black)

signal_response(ξ) = R * signal(ξ)
# Negative log-likelihood assuming a Gaussian energy
nll(ξ) = sum(N^-1 * (d .- signal_response(ξ)).^2)
# Standard-Hamiltonian
ham(ξ) = nll(ξ) + sum(ξ.^2)
# Metric of the given energy (here a Gaussian one)
M = N^-1

function val_vjp(ξ, δ_ξ)
	val, back = Zygote.pullback(signal_response, ξ)
	return val, first(back(δ_ξ))
end

function val_jvp(ξ, δ_ξ)
	dls = map((v, p) -> ForwardDiff.Dual(v, p...), ξ, δ_ξ)
	v_jvp = signal_response(dls)
	vs = ForwardDiff.value.(v_jvp)
	ps = mapreduce(ForwardDiff.partials, hcat, v_jvp)
	return vs, vec(ps)
end

function lcl_fisher(ξ_bar)
	nll_fish(ξ) = transpose(val_vjp(ξ_bar, M * val_jvp(ξ_bar, ξ)[2])[2])
	return LinearMap(nll_fish, dims) + I
end

function implicit_covariance_sample(cov_inv, ξ_bar)
    ξ_new = randn(dims)
	d_new = val_jvp(ξ_bar, ξ_new)[2] .+ R * sqrt(M^-1) * randn(dims)
	j_new = val_vjp(ξ_bar, M * d_new)[2]
	m_new = cg(cov_inv, j_new, log=true)[1]
    return ξ_new - m_new
end

function mgvi!(pos, n_smpls; nat_grad_scl=1.)
	fish = lcl_fisher(pos)
	println("sampling...")
	smpls = [implicit_covariance_sample(fish, pos) for i = 1 : n_smpls]
	println("minimizing...")
	kl(ξ) = mean(ham(ξ + s) for s in smpls)
	Δξ = cg(fish, first(gradient(kl, pos)), log=true)[1]
	println("computing natural gradient...")
	pos .-= nat_grad_scl * Δξ
	return pos, smpls
end

init_pos = 0.1 * randn(dims)

n_samples = 3
pos = copy(init_pos)
for i in 1:10
	global pos, samples
	samples = mgvi!(pos, 4; nat_grad_scl=1.)[2]
	#plot!(signal(pos), label="it. " * string(i))
end

for (i, s) in enumerate(samples)
	plot!(signal(pos + s), label="Post. Sample " * string(i), color=:gray)
end
plot!(signal(pos), label="Post. Mean", color=:orange)
savefig("mgvi_example.pdf")
