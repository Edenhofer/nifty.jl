using ForwardDiff

goo((x, y, z),) = [x^2, x*y*z, abs(z)]
foo((x, y, z),) = [x^2, x*y*z, abs(z)]

function foo(u::Vector{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # unpack: AoS -> SoA
    vs = ForwardDiff.value.(u)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ForwardDiff.partials, hcat, u)
    # get f(vs)
    val = foo(vs)
    # get J(f, vs) * ps (cheating). Write your custom rule here
    jvp = ForwardDiff.jacobian(goo, vs) * ps
    # pack: SoA -> AoS
    return map(val, eachrow(jvp)) do v, p
        ForwardDiff.Dual{T}(v, p...) # T is the tag
    end
end

ForwardDiff.gradient(u->sum(cumsum(foo(u))), [1, 2, 3]) == ForwardDiff.gradient(u->sum(cumsum(goo(u))), [1, 2, 3])


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
