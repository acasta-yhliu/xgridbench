using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets


@parameters x y t
@variables u(..) v(..) p(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

∇²(z) = Dxx(z) + Dyy(z)

ρ = 1
ν = 0.1

eq = [Dt(u(x, y, t)) + u(x, y, t) * Dx(u(x, y, t)) + v(x, y, t) * Dy(u(x, y, t)) + Dx(p(x, y, t)) / ρ - ν * ∇²(u(x, y, t)) ~ 0,
    Dt(v(x, y, t)) + u(x, y, t) * Dx(v(x, y, t)) + v(x, y, t) * Dy(v(x, y, t)) + Dy(p(x, y, t)) / ρ - ν * ∇²(v(x, y, t)) ~ 0,
    Dxx(p(x, y, t)) + Dyy(p(x, y, t)) + ρ * (Dx(u(x, y, t)) * Dx(u(x, y, t)) + 2 * Dy(u(x, y, t)) * Dx(v(x, y, t)) + Dy(v(x, y, t)) * Dy(v(x, y, t))) ~ 0]

domains = [x ∈ Interval(0, 2),
    y ∈ Interval(0, 2),
    t ∈ Interval(0, 1)]

# Periodic BCs
bcs = [u(x, y, 0) ~ 0, u(0, y, t) ~ 0, u(2, y, t) ~ 0, u(x, 0, t) ~ 0, u(x, 2, t) ~ 1,
    v(x, y, 0) ~ 0, v(0, y, t) ~ 0, v(2, y, t) ~ 0, v(x, 0, t) ~ 0, v(x, 2, t) ~ 0,
    p(x, y, 0) ~ 0, Dy(p(x, 0, t)) ~ 0, p(x, 2, t) ~ 0, Dx(p(0, y, t)) ~ 0, Dx(p(2, y, t)) ~ 0]

@named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t), p(x, y, t)])

N = 41

order = 2 # This may be increased to improve accuracy of some schemes

# Integers for x and y are interpreted as number of points. Use a Float to directtly specify stepsizes dx and dy.
discretization = MOLFiniteDifference([x => N, y => N], t, approx_order=order)

# Convert the PDE problem into an ODE problem
@time "discretization" prob = discretize(pdesys, discretization)
@time "solve" sol = solve(prob, Euler(), saveat=0.1)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]

solu = sol[u(x, y, t)]
solv = sol[v(x, y, t)]
solp = sol[p(x, y, t)]

using Plots
anim = @animate for k in 1:length(discrete_t)
    heatmap(solu[k, 2:end, 2:end], title="$(discrete_t[k])") # 2:end since end = 1, periodic condition
end
gif(anim, "plots/Brusselator2Dsol_u.gif", fps=8)