import torch
import matplotlib.pyplot as plt

rate = 1.5
sample_n = 10000

# Inverse transform sampling for an exponential distribution.
u = torch.rand(sample_n)
samples = -torch.log(1 - u) / rate

plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']

counts, bins, patches = plt.hist(
    samples.numpy(),bins = 50, density = True, alpha = 0.7, label = "Sample Histogram"
)

plt.xlabel('x(等待时间)')
plt.ylabel('概率密度')
plt.title('逆变换采样验证：直方图 vs 真实指数分布')
plt.legend()
plt.grid(True)
plt.show()

# x = torch.linspace(0, samples.max().item(), 300)
# theoretical_pdf = rate * torch.exp(-rate * x)

# plt.figure(figsize=(10, 6))
# plt.hist(samples.numpy(), bins=60, density=True, alpha=0.65, label="Sample histogram")
# plt.plot(x.numpy(), theoretical_pdf.numpy(), color="red", linewidth=2, label="Theoretical PDF")
# plt.title(f"Exponential Distribution Sampling (rate={rate})")
# plt.xlabel("x")
# plt.ylabel("density")
# plt.legend()
# plt.tight_layout()
# plt.savefig("exponential_distribution.png", dpi=150)
# plt.show()

# print(f"sample count: {sample_n}")
# print(f"sample mean: {samples.mean().item():.4f}")
# print(f"theoretical mean: {1 / rate:.4f}")
# print(f"sample variance: {samples.var(unbiased=True).item():.4f}")
# print(f"theoretical variance: {1 / rate ** 2:.4f}")