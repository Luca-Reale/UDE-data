#UDE's with known part
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots

function data(du, u, p, t)
  α,β,γ,δ = p
  du[1] = sin(t)+1
  du[2] = cos(t)+1
end

#here we initialise our parameters
tspan = (0.0f0, 3.0f0)
u0 = Float32[0, 0]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(data, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol = 1e-12, saveat = 0.1)

scatter(solution, alpha = 0.25)
plot!(solution, alpha = 0.5)

tsdata = Array(solution)

# We add some stachasticity with noisy data

#noisy_data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))
#plot(abs.(tsdata-noisy_data)')

ann = FastChain(FastDense(2, 32, tanh), FastDense(32, 32, tanh), FastDense(32, 2))
p = initial_params(ann)

function dudt_(u, p,t)
  x, y = u
  z = ann(u,p)
  [exp(t)+z[1], t+z[2]]
end

prob_nn = ODEProblem(dudt_, u0, tspan, p)
s = concrete_solve(prob_nn, Tsit5(), u0, p, saveat = solution.t)

plot(solution)
plot!(s)


running_total1 = 0
running_total2 = 0
for i in 1:31
  running_total1 = running_total1 + abs(solution[i][1]-s[i][1])
  running_total2 = running_total2 + abs(solution[i][2]-s[i][2])
end
println(running_total1)
println(running_total2)





ddprob = ContinuousDataDrivenProblem(s)

@variables t x(t) y(t) z(t)
u = [x;y;z]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, normalize = true)
print(ddsol, Val{true})