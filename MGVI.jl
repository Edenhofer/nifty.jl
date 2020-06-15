import IterativeSolvers: cg
import Random: randn
import ForwardDiff
import FFTW: plan_r2r, DHT
import Base: *
using ForwardDiff
using Zygote
using LinearAlgebra
using LinearMaps
using Statistics: mean
using Plots

dims = (1024)

# ξ := latent variables
ξ_truth = randn(dims)
k = [i < dims / 2 ? i :  dims-i for i = 0:dims-1]

# Define the harmonic transform operator as a matrix-like object
ht = plan_r2r(zeros(dims), DHT)
# Unfortunately neither Zygote nor ForwardDiff support planned Hartley
# transformations. While Zygote does not support AbstractFFTs.ScaledPlan,
# ForwardDiff does not overload the appropriate methods from AbstractFFTs.
# TODO: Push those changes to upstream. At the very least, Zygote is open to it
function *(trafo::typeof(ht), u::Vector{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
	# Unpack AoS -> SoA
	vs = ForwardDiff.value.(u)
	ps = mapreduce(ForwardDiff.partials, vcat, u)
	# Actual computation
	val = trafo * vs
	jvp = trafo * ps
	# Pack SoA -> AoS (depending on jvp, might need `eachrow`)
	return map((v, p) -> ForwardDiff.Dual{T}(v, p...), val, jvp)
end
Zygote.@adjoint function *(trafo::typeof(inv(ht)), xs)
	return trafo * xs, Δ -> (nothing, trafo * Δ)
end
Zygote.@adjoint function inv(trafo::typeof(ht))
	inv_t = inv(trafo)
	return inv_t, function (Δ)
		adj_inv_t = adjoint(inv_t)
		return (- adj_inv_t * Δ * adj_inv_t, )
	end
end

power = @. 50 / (k^2.5 + 1)
draw_sample(ξ) = inv(ht) * (power .* (ht * ξ))
signal(ξ) = exp.(draw_sample(ξ))

N = Diagonal(0.1^2 * ones(dims))
R = ones(dims)
#R[100:200] .= 0
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

function jacobian(f, ξ)
	to_dual(δ) = map((v, p) -> ForwardDiff.Dual(v, p...), ξ, δ)
	jvp(δ) = mapreduce(ForwardDiff.partials, vcat, f(to_dual(δ)))

	vjp(δ) = first(Zygote.pullback(f, ξ)[2](δ))

	return LinearMap{eltype(ξ)}(jvp, vjp, first(size(ξ)))
end

function local_fisher(jac)
	nll_fish(ξ) = adjoint(jac) * M * jac * ξ
	return LinearMap(nll_fish, dims) + I
end

function implicit_covariance_sample(cov_inv, jac)
    ξ_new = randn(dims)
	d_new = jac * ξ_new .+ R * sqrt(M^-1) * randn(dims)
	j_new = adjoint(jac) * M * d_new
	m_new = cg(cov_inv, j_new, log=true)[1]
    return ξ_new - m_new
end

function mgvi!(pos, n_samples; nat_grad_scl=1., mirror_samples=false)
	jac = jacobian(signal_response, pos)
	fish = local_fisher(jac)
	println("sampling...")
	smpls = [implicit_covariance_sample(fish, jac) for i = 1 : n_samples]

	println("minimizing...")
	if mirror_samples
		kl(ξ) = reduce(+, ham(ξ + s) + ham(ξ - s) for s in smpls) / (2 * n_samples)
		n_eff_smpls = 2 * n_samples
	else
		kl(ξ) = reduce(+, ham(ξ + s) for s in smpls) / n_samples
		n_eff_smpls = n_samples
	end

	# TODO: take metric of KL itself, i.e. the averaged one
	Δξ = cg(fish, first(gradient(kl, pos)), log=true)[1]
	println("computing natural gradient...")
	pos .-= nat_grad_scl * Δξ
	return pos, smpls
end

init_pos = 0.1 * randn(dims)

n_samples = 3
pos = copy(init_pos)
# Warm up in a region where the gradient is not yet very meaningful
samples = mgvi!(pos, 1; nat_grad_scl=1e-2, mirror_samples=true)[2]
samples = mgvi!(pos, 1; nat_grad_scl=1e-1, mirror_samples=true)[2]
for i in 1:10
	global pos, samples
	samples = mgvi!(pos, n_samples; nat_grad_scl=1.; mirror_samples=true)[2]
	#plot!(signal(pos), label="it. " * string(i))
end

for (i, s) in enumerate(samples)
	plot!(signal(pos + s), label="Post. Sample " * string(i), color=:gray)
end
plot!(signal(pos), label="Post. Mean", color=:orange)
savefig("mgvi_example.pdf")
