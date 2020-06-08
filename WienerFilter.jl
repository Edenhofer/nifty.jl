using IterativeSolvers
using FFTW
using Plots
using Random
using LinearMaps
using LinearAlgebra

dims = (1024)

# get hermitian symmetry in fourier space
xxii = randn(dims)
xii = fft(xxii)
# k values
k = [ i<dims/2 ? i :  dims-i for i=0:dims-1]

power = 50 ./(k.^1.5 .+ 1)

function draw_sample(xi)
    return real(ifft(power .* xi))
end
ss = draw_sample(xii)

plot(ss, color=:red, label="ground truth",linewidt=5)

N = ones(dims)

R = ones(dims)
R[512:742] .= 0
d = R .* ss + R .*N.^0.5 .* randn(dims)

plot!(d,seriestype = :scatter, marker = :x, color = :black)

function D_inv(x::Array{Float64,1})
    y = R .* 1 ./ N .* R.* x + real(ifft((1 ./power.^2) .* fft(x)))
    return y
end

Dinv = LinearMap(D_inv, dims;issymmetric=true)
j = R .* 1 ./ N .* d
m = cg(Dinv,j,log=true)[1]

function draw_posterior_sample(m)
    xxii = randn(dims)
    xii = fft(xxii)
    ss = draw_sample(xii)
    d = R .* ss + R .*N.^0.5 .* randn(dims)
    j = R .* 1 ./ N .* d
    m_p = cg(Dinv,j,log=true)[1]
    return ss - m_p + m
end

samps = []
for i in 1:10
    samp = draw_posterior_sample(m)
    append!(samps,[samp])
    plot!(samp,color=:black, linewidth=0.5, label="",alpha=0.3)
end

plot!(m, label="reconstruction", color=:black, linewidth=3)
