#UDE's with known part

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots

function data(du, u, p, t)
  α,β,γ,δ = p
  du[1] = t
end

#here we initialise our parameters
tspan = (0.0f0, 3.0f0)
u0 = Float32[0]
p_ = Float32[1, 1, 1, 1]
prob = ODEProblem(data, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol = 1e-12, saveat = 0.1)
tsteps = range(0.0f0, 1.5f0, length = datasize)

scatter(solution, alpha = 0.25)
plot!(solution, alpha = 0.5)

tsdata = Array(solution)'
ann = FastChain(FastDense(1, 32, tanh), FastDense(32, 32, tanh), FastDense(32, 1))
p = initial_params(ann)

function dudt(u,p,t)
  x = u
  z = ann(u,p)
  [0+z[1]]
end

prob_nn = ODEProblem(dudt, u0, tspan, p)
s = concrete_solve(prob_nn, Tsit5(), u0, p, saveat = solution.t)
Array(s)
tsdata'

plot(solution)
plot!(s)

function predict_neuralode(p)
  tmp_prob = remake(prob_nn,p=p)
  Array(solve(tmp_prob,Tsit5(),saveat=0.1,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(p)
  tmp_prob = remake(prob_nn, p=p)
  tmp_sol = solve(tmp_prob,Tsit5(), saveat = 0.1)
  sum(abs2, Array(tmp_sol)-tsdata')
end

loss(p)

function neuralode_callback(p,l)
  @show l
  tmp_prob = remake(prob_nn, p=p)
  tmp_sol = solve(tmp_prob,Tsit5(), saveat = tsteps)
  fig = plot(tmp_sol, label="UDE Solution", lw=3)
  scatter!(fig)
  scatter!(solution, label = "ODE Data")
  display(fig)
  false
end

res1 = DiffEqFlux.sciml_train(loss,p, ADAM(0.05), maxiters=100, cb=neuralode_callback)

sol = predict_neuralode(res1)'

running_total = 0
for i in 1:31
  running_total += abs(solution[i][1] - sol[i])
end
println(running_total)
x = 1.5:0.1:3
plot!(x, sol[16:31], label="UDE Prediction", lw=3)
