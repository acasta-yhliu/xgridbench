import matplotlib.pyplot as plt
plt.style.use("classic")

x = ["WithoutCache", "WithCache", "Following"]
y = [0.29425716400146484, 0.014646530151367188, 0.0013039328835227273]
plt.bar(x, y, color=["gray", "gray", "lightgray"])
plt.grid(axis="y", ls="--")
plt.ylabel("time/s")
plt.title("First-Time Kernel Invocation Time")
plt.savefig("benchmark_firstcall.png")
plt.clf()

invoke_t = y[2]
link_t = y[1] - y[2]
compile_t = y[0] - y[1]
ttl_t = invoke_t + link_t + compile_t
plt.pie([compile_t, link_t, invoke_t], labels=[
        f"Compiling ({compile_t * 100 /ttl_t:.2f}%)", f"Linking ({link_t * 100/ttl_t:.2f}%)", f"Running ({invoke_t * 100 /ttl_t:.2f}%)"], explode=(0.0, 0.05, 0.1), colors=["gray", "lightgray", "lightgray"])
plt.title("First-Time Kernel Invocation Ratio")
plt.savefig("benchmark_firstcall_pie.png")
plt.clf()
