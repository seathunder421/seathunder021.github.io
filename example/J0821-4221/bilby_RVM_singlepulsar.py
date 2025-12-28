import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
import seaborn as sns
import csv
import json
import pickle
import math
import glob
import warnings
import psrqpy
import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
import bilby
import matplotlib.patches as patches
# 设置 Pandas 配置
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', None)   # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
# 设置 NumPy 配置
np.set_printoptions(suppress=True)
np.random.seed(666)
# 设置 Matplotlib 配置
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 忽略警告
warnings.filterwarnings("ignore")


def plot_errorbar(x1, y1, xerr1, yerr1, x2, y2, xerr2, yerr2, color1, color2, title, xlabel, ylabel, xlim, ylim, figsize=(9, 3), dpi=600):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.errorbar(x1, y1, xerr=xerr1, yerr=yerr1, fmt=".k", markersize=4, elinewidth=1, color=color1, label="psi_fit_chi")
    plt.errorbar(x2, y2, xerr=xerr2, yerr=yerr2, fmt=".k", markersize=4, elinewidth=1, color=color2, label="PA_filtered")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("比较图.png")

def process_data(file):
    data = pd.read_csv(file, header=None, delimiter=',', on_bad_lines='skip')
    data.columns = ['Angle','I_normalized','Linear_Polarization', 'V_normalized', 'PA','PA_error','PA_filtered',"PA_error_filtered","SN"]
    # 将列转换为数值类型，处理可能存在的非数值数据
    for col in ['Angle','I_normalized','Linear_Polarization', 'V_normalized', 'PA','PA_error','PA_filtered',"PA_error_filtered","SN"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data   



print("Current working directory:", os.getcwd())
# 匹配所有以 filtered_data.csv 结尾的文件
file = glob.glob("*_filtered_data.csv")[0]
print("读取当前文件：:",file)
print("===========================================================================================") 
print("===========================================================================================") 
psr = file.split("_")[0]
all_min, all_max = map(float, [file.split("_")[1], file.split("_")[2]])  # 转换为浮点数
data = process_data(file)   
# 打印分离结果
print("脉冲星：", psr)
print("左边界：", all_min)
print("右边界：", all_max)
print("===========================================================================================") 

Angle = data["Angle"]
PA = data["PA_filtered"]
PA_error = data["PA_error_filtered"]
PA[PA == 0] = np.nan  # 将 PA 中的 0 替换为 NaN
PA_error[PA_error == 0] = np.nan
mask = (Angle >= all_min) & (Angle <= all_max)
Angle_filtered = Angle[mask]
PA_filtered = PA[mask]
PA_error = PA_error[mask]
valid_mask = ~np.isnan(PA_filtered)  # 生成非 NaN 掩码

Angle_filtered = Angle_filtered[valid_mask]
PA_filtered = PA_filtered[valid_mask]
PA_err_filtered =  PA_error[valid_mask]


# 创建一个 DataFrame
df = pd.DataFrame({'Angle': Angle_filtered,'PA': PA_filtered,'PA_error': PA_err_filtered})
# 保存为 CSV 文件
df.to_csv(f'{psr}_data.csv', index=False)


print("======================================MCMC=================================================") 
def plot_distribution(data, label=None, best_val=None):
    if label is None:
        label = r"3σ"  # Use LaTeX format
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=50, alpha=0.6, color='g', edgecolor='black', label='Data')
    if best_val is not None:
        plt.axvline(best_val, color='r', linestyle='--', linewidth=2, label=f"Best K")  # Mark the optimal value
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title(f'K Data Distribution ({label})')  # Add label in the title
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"K_{label}_Distribution.png") 
def distribution(data, label=None):
    if label is None:
        label = r"3sigma"  # 使用 LaTeX 格式
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=50, alpha=0.6, color='g', edgecolor='black', label='Data')
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.yscale('log')  # 设置 y 轴对数刻度
    plt.title(f'Chi Data Distribution ({label})')  # 在标题中添加标签
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"Chi_{label}_Distribution.png")  # 文件名包含 σ 值
def rvm_model(phi, alpha, beta, phase_offset, psi_offset):
    # 计算 RVM
    alpha = np.radians(180 - alpha)
    beta = np.radians(-beta)
    phi = np.radians(phi)
    phase_offset = np.radians(phase_offset)
    zeta = alpha + beta
    numerator = np.sin(alpha) * np.sin(phi - phase_offset)
    denominator = np.sin(zeta) * np.cos(alpha) - np.cos(zeta) * np.sin(alpha) * np.cos(phi - phase_offset)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.degrees(np.arctan(numerator / denominator)) + psi_offset
# 定义似然函数
likelihood = bilby.likelihood.GaussianLikelihood(x=Angle_filtered,y=PA_filtered,func=rvm_model,sigma=PA_err_filtered)
# 定义先验
priors = bilby.prior.PriorDict()
priors['alpha'] = bilby.prior.Uniform(0, 180, 'alpha')
priors['beta'] = bilby.prior.Uniform(-90, 90, 'beta')
priors['phase_offset'] = bilby.prior.Uniform(-150, 150, 'phase_offset')
priors['psi_offset'] = bilby.prior.Uniform(-150,150, 'psi_offset')

if __name__ == '__main__':   
    quantile = [0.00135, 0.5, 0.99865]            

    result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='dynesty',
                                     nlive = 2600,
                                     sample="rwalk",
                                     dlogz =0.01,
                                     walks=200,
                                     print_progress=True,
                                     write_progress=True,
                                     bootstrap=0,
                                     nthreads=10,
                                     burnin=500,
                                     outdir='./',
                                     label=psr + f"_{0}_RVM_fit",
                                     verbose=True)   

    truths = [result.posterior['phase_offset'].mean(),result.posterior['psi_offset'].mean()]
    fig = result.plot_corner(
        parameters=['phase_offset', 'psi_offset'],  # 指定采样参数
        labels=[r'$\phi$', r'$\psi$'],  # 自定义轴标签
        show_titles=True,  # 显示每个参数的标题
        title_fmt='.2f',  # 标题的格式化方式
        quantiles=quantile,  # 显示分位点
        color='grey',  # 线和点的颜色
        smooth=1,  # 平滑度
        truths=truths,  # 设置真实值为拟合参数均值
        truth_color='red', )
    



    # 添加自定义分位点线条
    quantiles = [0.00135, 0.025, 0.16, 0.5, 0.84, 0.975, 0.99865]
    colors = ["red", "orange", "blue", "green", "blue", "orange", "red"]
    sigma_labels = ["-3σ", "-2σ", "-1σ", "Median", "+1σ", "+2σ", "+3σ"]
    # 获取角点图的所有子图
    axes = np.array(fig.axes).reshape(len(truths), len(truths))
    # 遍历对角线上的直方图，绘制参考线
    for i, ax in enumerate(axes.diagonal()):
        data1 = result.posterior[result.posterior.columns[i]].values  # 当前参数的采样值
        for q, color, label in zip(quantiles, colors, sigma_labels):
            quantile_value = np.percentile(data1, q * 100)  # 计算百分位数
            ax.axvline(quantile_value, color=color, linestyle="--", linewidth=0.8)
            ax.text(quantile_value, ax.get_ylim()[1] * 0.8,  label, color=color, fontsize=7, rotation=90, verticalalignment="center", horizontalalignment="right" )
   # 显示图像
    fig.suptitle("Corner Plot", fontsize=16, y=1.05)
    fig.savefig(psr + "_phase_psi_corner_plot.png")



    posterior_samples = result.posterior
    phase_samples = np.array(posterior_samples['phase_offset'])
    psi_samples = np.array(posterior_samples['psi_offset'])
    # 计算二维 KDE
    samples = np.vstack([phase_samples, psi_samples])
    kde = gaussian_kde(samples)
    kde.set_bandwidth(bw_method='scott')  # 自动调整带宽
    # 生成网格点
    phase_range = np.linspace(min(phase_samples), max(phase_samples), 400)
    psi_range = np.linspace(min(psi_samples), max(psi_samples), 400)
    phase_grid, psi_grid = np.meshgrid(phase_range, psi_range)
    grid_points = np.vstack([phase_grid.ravel(), psi_grid.ravel()])
    # 计算密度
    density = kde(grid_points)
    density_grid = density.reshape(phase_grid.shape)
    # 找到密度最大的位置
    max_density_idx = np.argmax(density)
    best_phase = phase_grid.ravel()[max_density_idx]
    best_psi = psi_grid.ravel()[max_density_idx]
    print(f"密度最大点: phase = {best_phase}, psi = {best_psi}")

    # 创建子图
    fig, ax = plt.subplots(figsize=(8, 7),dpi=600)
    # 假设 phase_grid, psi_grid, density_grid 已经被定义并且是合适的二维数组
    # 定义密度图的等高线级别
    levels = np.linspace(np.min(density_grid), np.max(density_grid), 20)  # 更高的等高线密度
    # 绘制 2D KDE 等高线图，使用原始 density_grid 数据
    contour = ax.contourf(phase_grid, psi_grid, density_grid, levels=levels, cmap="viridis")
    # 绘制最密集点
    ax.scatter(best_phase, best_psi, color="red", marker="*", s=150, label="Max Probability", edgecolor='black', linewidth=1.5)
    # 设置轴标签和标题，增强可读性
    ax.set_xlabel(r"$\varphi$ (deg)", fontsize=14)
    ax.set_ylabel(r"$\psi$ (deg)", fontsize=14)
    # 设置图例
    ax.legend(fontsize=10)
    # 创建归一化对象，将颜色条范围设置为 [0, 1]
    norm = Normalize(vmin=0, vmax=1)
    # 使用 ScalarMappable 创建颜色条，并将颜色条归一化
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])  # 需要设置一个空数组
    # 添加颜色条，确保颜色条被归一化到 [0, 1]
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Probability Density", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    # 调整图形布局
    plt.tight_layout()
    # 保存图形
    plt.savefig("2D_KDE_phase_psi_density_normalized.png")
    plt.savefig("2D_KDE_phase_psi_density_normalized.pdf")
    # 归一化密度到 [0,1]
    density_min, density_max = np.min(density), np.max(density)
    density_norm = (density - density_min) / (density_max - density_min)
    density_grid_norm = density_norm.reshape(phase_grid.shape)
    # 设置颜色归一化
    norm = mcolors.Normalize(vmin=0, vmax=1)
    levels = np.linspace(0, 1, 5)  # 10 个等间隔等级
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    im = ax.imshow(density_grid_norm, cmap="viridis", origin="lower", aspect="auto", 
                extent=[min(phase_samples), max(phase_samples), min(psi_samples), max(psi_samples)], 
                interpolation='bilinear', norm=norm)
    ax.scatter(best_phase, best_psi, color="red", marker="x", s=20, label="best parameter")
    ax.text(best_phase + 0.02, best_psi + 0.02, f"({best_phase:.3f}, {best_psi:.3f})", color="red", fontsize=8)
    # 设置坐标轴标签
    ax.set_ylabel(r"$\psi$ (deg)",fontsize=12)
    ax.set_xlabel(r"$\varphi$ (deg)",fontsize=12)
    ax.legend(fontsize=10)
    # 添加颜色条，归一化到 [0,1] 并保留一位小数
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, ticks=levels)
    cbar.set_label("Normalized Density")
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in levels])  # 保留 1 位小数
    # 保存图像
    fig.savefig("2D_phsae_psi_density_plot.png", dpi=300, bbox_inches='tight')

    phase_quantiles = posterior_samples["phase_offset"].quantile(quantile)
    phase_lower = phase_quantiles[quantile[0]]
    phase_upper = phase_quantiles[quantile[2]]
    # 计算 psi 的分位点
    psi_quantiles = posterior_samples["psi_offset"].quantile(quantiles)
    psi_lower = psi_quantiles[quantile[0]]
    psi_upper = psi_quantiles[quantile[2]]
    # 输出结果
    print(f"Phase : {phase_lower},  {phase_upper}")
    print(f"Psi : {psi_lower}, : {psi_upper}")


    if os.path.exists("phi_psi.csv"):
        os.remove("phi_psi.csv")
        print("成功删除:phi_psi.csv")
    # 生成 CSV 文件路径
    for i in range(1):
      #  df = pd.DataFrame(pickle.load(open(f"{psr}_{i}_RVM_fit_dynesty.pickle", "rb")).samples, columns=['alpha', 'beta'])   
        df = pd.DataFrame(json.load(open(f"{psr}_{0}_RVM_fit_result.json"))["posterior"]["content"])[['phase_offset', 'psi_offset']]
        # 追加数据
        if os.path.exists("phi_psi.csv"):
            df.to_csv("phi_psi.csv", mode='a', header=False, index=False)  # 追加模式，不写表头
        else:
            df.to_csv("phi_psi.csv", index=False)   # 写入模式，写表头

    posterior1 = pd.read_csv("phi_psi.csv")
    posterior_psi_phi = posterior1[(posterior1["phase_offset"] < phase_upper) & (posterior1["phase_offset"] > phase_lower) & \
                                        (posterior1["psi_offset"] < psi_upper) & (posterior1["psi_offset"] > psi_lower)] #
                                        


    likelihood = bilby.likelihood.GaussianLikelihood(x=Angle_filtered,y=PA_filtered,func=rvm_model,sigma=PA_err_filtered)
    priors = bilby.prior.PriorDict()
    priors['alpha'] = bilby.prior.Uniform(0, 180, 'alpha')
    priors['beta'] = bilby.prior.Uniform(-90, 90, 'beta')
    priors['phase_offset'] = bilby.prior.DeltaFunction(best_phase) 
    priors['psi_offset'] = bilby.prior.DeltaFunction(best_psi)   
    for i in range(1):
        result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='dynesty',
                                     nlive=2500,
                                     sample="rwalk",
                                     dlogz =0.01,
                                     walks=200,
                                     bootstrap=0,
                                     nthreads=10,
                                     burnin=500,
                                     write_progress=True,
                                     outdir='./',
                                    label=psr + f"_{1}_RVM_{i}_fit",
                                    verbose=True)   


    
    truths = [result.posterior['alpha'].mean(),result.posterior['beta'].mean()]
    fig = result.plot_corner(
        parameters=['alpha', 'beta'],  # 指定采样参数
        labels=[r'$\alpha$', r'$\beta$'],  # 自定义轴标签
        show_titles=True,  # 显示每个参数的标题
        title_fmt='.2f',  # 标题的格式化方式
        quantiles= quantile,  # 显示分位点
        color='grey',  # 线和点的颜色
        smooth=0,  # 平滑度
        truths=truths,  # 设置真实值为拟合参数均值
        truth_color='red', )
    # 添加自定义分位点线条
    quantiles = [0.00135, 0.025, 0.16, 0.5, 0.84, 0.975, 0.99865]
    colors = ["red", "orange", "blue", "green", "blue", "orange", "red"]
    sigma_labels = ["-3σ", "-2σ", "-1σ", "Median", "+1σ", "+2σ", "+3σ"]
    # 获取角点图的所有子图
    axes = np.array(fig.axes).reshape(len(truths), len(truths))
    # 遍历对角线上的直方图，绘制参考线
    for i, ax in enumerate(axes.diagonal()):
        data1 = result.posterior[result.posterior.columns[i]].values  # 当前参数的采样值
        for q, color, label in zip(quantiles, colors, sigma_labels):
            quantile_value = np.percentile(data1, q * 100)  # 计算百分位数
            ax.axvline(quantile_value, color=color, linestyle="--", linewidth=0.8)
            ax.text(quantile_value, ax.get_ylim()[1] * 0.8,  label, color=color, fontsize=7, rotation=90, verticalalignment="center", horizontalalignment="right" )
    # 显示图像
    fig.suptitle("Corner Plot", fontsize=16, y=1.05)
    fig.savefig(psr + "_alpha_beta_corner_plot.png")

    if os.path.exists("alpha_beta.csv"):
        os.remove("alpha_beta.csv")
        print("成功删除:alpha_beta.csv")
    # 生成 CSV 文件路径
    for i in range(1):
      #  df = pd.DataFrame(pickle.load(open(f"{psr}_{i}_RVM_fit_dynesty.pickle", "rb")).samples, columns=['alpha', 'beta'])   
        df = pd.DataFrame(json.load(open(f"{psr}_{1}_RVM_{i}_fit_result.json"))["posterior"]["content"])[['alpha', 'beta']]
        # 追加数据
        if os.path.exists("alpha_beta.csv"):
            df.to_csv("alpha_beta.csv", mode='a', header=False, index=False)  # 追加模式，不写表头
        else:
            df.to_csv("alpha_beta.csv", index=False)   # 写入模式，写表头
# 读取数据
    posterior_samples = pd.read_csv("alpha_beta.csv")
    alpha_samples = np.array(posterior_samples['alpha'])
    beta_samples = np.array(posterior_samples['beta'])
    # 计算二维 KDE
    samples = np.vstack([alpha_samples, beta_samples])
    kde = gaussian_kde(samples)
    kde.set_bandwidth(bw_method='scott')  # 自动调整带宽
    # 生成高分辨率网格
    alpha_range = np.linspace(min(alpha_samples), max(alpha_samples), 200)
    beta_range = np.linspace(min(beta_samples), max(beta_samples), 200)
    alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)
    grid_points = np.vstack([alpha_grid.ravel(), beta_grid.ravel()])
    # 计算二维密度
    density = kde(grid_points)
    density_grid = density.reshape(alpha_grid.shape)

    # 找到密度最高点
    max_idx = np.unravel_index(np.argmax(density_grid), density_grid.shape)
    best_alpha = alpha_range[max_idx[1]]
    best_beta = beta_range[max_idx[0]]
    print(f"最优 alpha = {best_alpha}, 最优 beta = {best_beta}")
    print(f"最优 alpha = {best_alpha}, 最优 beta = {best_beta}")
    # 设置 Seaborn 风格，并去掉网格
        # 归一化密度到 [0,1]
    density_min, density_max = np.min(density), np.max(density)
    density_norm = (density - density_min) / (density_max - density_min)
    density_grid_norm = density_norm.reshape(alpha_grid.shape)
    # 设置颜色归一化
    norm = mcolors.Normalize(vmin=0, vmax=1)
    levels = np.linspace(0, 1, 5)  # 10 个等间隔等级
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    im = ax.imshow(density_grid_norm, cmap="viridis", origin="lower", aspect="auto", 
                extent=[min(alpha_samples), max(alpha_samples), min(beta_samples), max(beta_samples)], 
                interpolation='bilinear', norm=norm)
    # 标出密度最大点
    ax.scatter(best_alpha, best_beta, color="red", marker="x", s=20, label="best parameter")
    ax.text(best_alpha + 0.02, best_beta + 0.02, f"({best_alpha:.3f}, {best_beta:.3f})", color="red", fontsize=8)
    # 设置坐标轴标签]

    print("================================================")



    N = len(Angle_filtered) 
    a_rad = np.radians(180 - best_alpha)
    b_rad = np.radians(-best_beta)
    best_K = np.sin(a_rad) / np.sin(b_rad)
    psi_fit_chi = rvm_model(Angle_filtered, alpha=best_alpha, beta=best_beta, phase_offset=best_phase, psi_offset=best_psi)
   
   
    plot_errorbar(
        x1=Angle_filtered, y1=psi_fit_chi, xerr1=0, yerr1=0, 
        x2=Angle_filtered, y2=PA_filtered, xerr2=0, yerr2=PA_err_filtered, 
        color1='purple', color2='blue', 
        title='Fitting Data', xlabel='Pulse Phase', ylabel='PA (degree)', 
        xlim=(all_min, all_max), ylim=(-180, 180)
    )
   
    squared = np.sum(((PA_filtered - psi_fit_chi) / PA_err_filtered) ** 2) 
    best_chi_red = squared / (N - 4) 
    print(best_chi_red)
    ax.text(0.02, 0.98, f"$\\chi_{{red}}$ = {best_chi_red:.2f}", 
            color="white", fontsize=8, fontweight="bold", ha="left", va="top", 
            transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    ax.set_xlabel(r"$\alpha$ (deg)",fontsize=10)
    ax.set_ylabel(r"$\beta$ (deg)",fontsize=10)
    ax.legend(fontsize=10)
    # 添加颜色条，归一化到 [0,1] 并保留一位小数
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, ticks=levels)
    cbar.set_label("Probability Density", fontsize=14)
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in levels])  # 保留 1 位小数
    cbar.ax.tick_params(labelsize=14)
    # 保存图像
    fig.savefig("2D_alpha_beta_density_plot.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 7),dpi=600)
    # 假设 alpha_grid, beta_grid, density_grid 已经被定义并且是合适的二维数组
    # 定义密度图的等高线级别
    levels = np.linspace(np.min(density_grid), np.max(density_grid), 20)  # 更高的等高线密度
    # 绘制 2D KDE 等高线图，使用原始 density_grid 数据
    contour = ax.contourf(alpha_grid, beta_grid, density_grid, levels=levels, cmap="viridis")
    # 绘制最密集点
   # ax.scatter(best_alpha, best_beta, color="red", marker="*", s=150, label=f"$\chi^2_{{\nu}}$ = {best_chi_red:.2f}", edgecolor='black', linewidth=1.5)
    ax.scatter(best_alpha, best_beta, color="red", marker="*", s=150, label="Max Probability", edgecolor='black', linewidth=1.5)
 
   # ax.scatter(best_alpha, best_beta, color="red", marker="*", s=150, label=f"$\\chi_{{red}}$ = {best_chi_red:.2f}", edgecolor='black', linewidth=1.5)
    # 设置轴标签和标题，增强可读性
    ax.set_xlabel(r"$\alpha$ (deg)", fontsize=18)
    ax.set_ylabel(r"$\beta$  (deg)", fontsize=18)
    # 设置图例
   # ax.set_ylim(1.7,2.4)
    ax.legend(fontsize=10)
    # 创建归一化对象，将颜色条范围设置为 [0, 1]
    norm = Normalize(vmin=0, vmax=1)
    # 使用 ScalarMappable 创建颜色条，并将颜色条归一化
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])  # 需要设置一个空数组
    # 添加颜色条，确保颜色条被归一化到 [0, 1]
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Probability Density', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    # 调整图形布局
    plt.tight_layout()
    plt.savefig("2D_KDE_alpha_beta_density_normalized.png")
    plt.savefig("2D_KDE_alpha_beta_density_normalized.pdf")



    # 过的alpha 和 beta 最优解 以及误差
    quantile = [0.00135, 0.5, 0.99865]
    #quantile = [0.025, 0.5, 0.975]
    # 计算 alpha 的分位点
    alpha_quantiles = posterior_samples["alpha"].quantile(quantile)
    alpha_lower = alpha_quantiles[quantile[0]]
    alpha_median = alpha_quantiles[quantile[1]]
    alpha_upper = alpha_quantiles[quantile[2]]
    # 计算 beta 的分位点
    beta_quantiles = posterior_samples["beta"].quantile(quantile)
    beta_lower = beta_quantiles[quantile[0]]
    beta_median = beta_quantiles[quantile[1]]
    beta_upper = beta_quantiles[quantile[2]]
    # 输出结果
    print(f"Alpha 16%: {alpha_lower}, Median: {alpha_median}, 84%: {alpha_upper}")
    print(f"Beta 16%: {beta_lower}, Median: {beta_median}, 84%: {beta_upper}")

    # 设置 Seaborn 风格
    # 选择四个参数及其最优值
    params = ["alpha", "beta"]
    best_values = [best_alpha, best_beta]  # 请替换为你的最优值
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # 遍历参数并绘制直方图1
    for ax, param, best_val in zip(axes.flatten(), params, best_values):
        sns.histplot(posterior_samples[param], bins=200, kde=True, ax=ax)
        ax.axvline(best_val, color='r', linestyle='--', linewidth=2, label=f"Best {param}")  # 标注最优值
        ax.set_xlabel(param)
        ax.set_ylabel("Density")
        ax.legend()
    # 调整子图间距
    fig.suptitle(f"Parameter Distributions of {psr}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Distribution of parameter")


    # 计算 phase 的分位点
        # 上面把 位置定出来了 现在开始计算误差
    def compute_K_and_chi(posterior_samples, best_alpha=None, best_beta=None, best_phase=None, best_psi=None):
        K_slopes = []
        Chi = []
        N = len(Angle_filtered)  # 数据点数量

        for i in range(len(posterior_samples)):  # 只绘制符合条件的样本
            alpha = posterior_samples.iloc[i]['alpha'] if best_alpha is None else best_alpha
            beta = posterior_samples.iloc[i]['beta'] if best_beta is None else best_beta
            phase_offset = best_phase if best_phase is not None else posterior_samples.iloc[i]['phase_offset']
            psi_offset = best_psi if best_psi is not None else posterior_samples.iloc[i]['psi_offset']

            a_rad = np.radians(180 - alpha)
            b_rad = np.radians(-beta)
            K = np.sin(a_rad) / np.sin(b_rad)
            K_slopes.append(K)

            # 计算每组样本的拟合曲线
            psi_fit_chi = rvm_model(Angle_filtered, alpha=alpha, beta=beta, phase_offset=phase_offset, psi_offset=psi_offset)
            squared = np.sum(((PA_filtered - psi_fit_chi) / PA_err_filtered) ** 2)
            chi_red = squared / (N - 4)
            Chi.append(chi_red)

        return K_slopes, Chi

    # 处理 alpha_beta.csv
    alpha_beta = pd.read_csv("alpha_beta.csv")
    K_slopes, Chi = compute_K_and_chi(alpha_beta, best_alpha=None, best_beta=None, best_phase=best_phase, best_psi=best_psi)
    alpha_beta["phase_offset"] = best_phase
    alpha_beta["psi_offset"] = best_psi
    alpha_beta["chi_red"] = Chi
    alpha_beta["K"] = K_slopes
    alpha_beta.to_csv("alpha_beta.csv", index=False)
    # 合并后只保留 alpha_beta.csv
    combined_df = alpha_beta
    combined_df.to_csv("combined_data.csv", index=False)
    posterior_sample = pd.read_csv("combined_data.csv")
    posterior_sample = posterior_sample[(posterior_sample["alpha"] < alpha_upper) & (posterior_sample["alpha"] > alpha_lower) & \
                                        (posterior_sample["beta"] < beta_upper) & (posterior_sample["beta"] > beta_lower)] #
                                            

            

    print("===========================================================================================")
    K_max_value = max(posterior_sample["K"])
    K_min_value = min(posterior_sample["K"])
    print(K_max_value, K_min_value,best_K)
    K_lower_error = abs(best_K - K_min_value)
    K_upper_error = abs(K_max_value - best_K)
    # Calculate Chi_max_value and Chi_min_value
    Chi_max_value = max(posterior_sample["chi_red"])
    Chi_min_value = min(posterior_sample["chi_red"])
    # Calculate the Chi errors
    Chi_lower_error = abs(best_chi_red - Chi_min_value)
    Chi_upper_error = abs(Chi_max_value - best_chi_red)

    # Output the results
    print("Best K:", best_K)
    print("best chi",best_chi_red)
    print("K_max_value:", K_max_value, "K_min_value:", K_min_value)
    print("K_lower_error:", K_lower_error, "K_upper_error:", K_upper_error)
    print("Chi_max_value:", Chi_max_value, "Chi_min_value:", Chi_min_value)
    print("Chi_lower_error:", Chi_lower_error, "Chi_upper_error:", Chi_upper_error)

    phi_fit = np.linspace(all_min, all_max, 1000)
    plt.figure(figsize=(8, 5), dpi=500)
    print(len(posterior_sample))
    for i in range(len(posterior_sample)):  # 只绘制符合条件的样本
        alpha = posterior_sample.iloc[i]['alpha']
        beta = posterior_sample.iloc[i]['beta']
        phase_offset = posterior_sample.iloc[i]['phase_offset']
        psi_offset = posterior_sample.iloc[i]['psi_offset']
        # 计算每组样本的拟合曲线
        psi_fit = rvm_model(phi_fit, alpha=alpha, beta=beta, phase_offset=phase_offset, psi_offset=psi_offset)
        plt.plot(phi_fit, psi_fit, color='grey', linewidth=4, alpha=0.4)  # 透明度调整
    # 解包参数
    psi_fit = rvm_model(phi_fit, alpha=best_alpha, beta=best_beta, phase_offset=best_phase, psi_offset=best_psi)
    plt.plot(phi_fit, psi_fit, 
         label=f"$\\chi^2_{{\\nu}}$ = {best_chi_red:.2f}$^{{+{Chi_upper_error:.2f}}}_{{-{Chi_lower_error:.2f}}}$", 
         color='green', linewidth=2)
    # 绘制观测数据点
    plt.errorbar(Angle_filtered, PA_filtered, yerr=PA_err_filtered, fmt='.', color="black", markersize=4, capsize=2, elinewidth=1)
    # 图形标签和样式
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))

    plt.title(f"PSR {psr}", fontsize=16, pad=12, fontweight='bold')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Longitud(deg)', fontsize=14)  # 横坐标字体大小
    plt.ylabel('PA (deg)', fontsize=14)  # 纵坐标字体大小
    plt.legend(fontsize=12)
    # 设置刻度线的长度（横纵坐标的刻度线长度）
    plt.xlim(all_min, all_max)
    plt.ylim(min(psi_fit)-30, max(psi_fit)+30)
    # 启用网格线
    plt.grid(False)
    plt.savefig(f"{psr}_fiting.png")
    plt.savefig(f"{psr}_fiting.pdf")


    # 读取数据
    data1 = pd.read_csv("combined_data.csv")
    alpha = data1["alpha"].values
    beta = data1["beta"].values
    chi_red = data1["chi_red"].values
    # 找到最小的卡方值
    # 过滤数据（仅保留 chi_red 在 best_chi_red+10 以内的点）
    mask = chi_red < best_chi_red + 10
    alpha_filtered = alpha[mask]
    beta_filtered = beta[mask]
    chi_red_filtered = chi_red[mask]
    print("有效数据点：", len(alpha_filtered))
    # 计算二维直方图
    bins = 300
    density, xedges, yedges = np.histogram2d(alpha_filtered, beta_filtered, bins=bins, density=True)
    # 生成像素左下角坐标
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    # 计算符合条件的点并累积到 density
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            # 计算卡方值（这里需要 rvm_model 和相关变量）
            psi_fit_chi = rvm_model(Angle_filtered, alpha=x, beta=y, 
                                    phase_offset=phase_offset, psi_offset=psi_offset)
            squared = np.sum(((PA_filtered - psi_fit_chi) / PA_err_filtered) ** 2)
            chi_red_value = squared / (N - 4)
            if Chi_min_value< chi_red_value < Chi_max_value :
                density[i, j] += 1  # 直接累加到 density
    # 处理密度数据，避免 log 归一化问题
    density[density == 0] = np.nan
    vmin, vmax = np.nanpercentile(density, [1, 100])
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    # 绘制图像
    fig, ax = plt.subplots(figsize=(6, 5), dpi=400)
    # 绘制密度图
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(density.T, cmap="viridis", origin="lower", 
                aspect="auto", extent=extent, norm=norm)

    # 标注最佳 (alpha, beta)
    ax.scatter(best_alpha, best_beta, color="red", marker="x", s=10)
    ax.text(best_alpha + 3, best_beta, f"({best_alpha:.3f}, {best_beta:.3f})", 
            color="red", fontsize=7, ha="left", va="bottom")
    # 显示最小卡方值
    ax.text(0.02, 0.98, f"$\\chi^2_{{\\nu}}$ = {best_chi_red:.2f}$^{{+{Chi_upper_error:.2f}}}_{{-{Chi_lower_error:.2f}}}$", 
        color="black", fontsize=8, fontweight="bold", ha="left", va="top", 
        transform=ax.transAxes, bbox=dict(facecolor='grey', alpha=0.35, edgecolor='none')) 
    # 设置坐标轴标签
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    ax.grid(False)
    # 保存图像
    fig.savefig("2D_1_plot_combined.png", dpi=300, bbox_inches='tight', transparent=True)

    print("图像已保存为 2D_1_plot_combined.png")       
    mask = (chi_red > best_chi_red + 1)
    alpha_filtered = alpha[~mask]
    beta_filtered = beta[~mask]
    chi_red_filtered = chi_red[~mask]
    print("有效数据点：", len(alpha_filtered))
    # 创建图像
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    # 设置背景透明

    # 计算二维直方图（密度估计）
    bins = 200
    density, xedges, yedges = np.histogram2d(alpha_filtered, beta_filtered, bins=bins, density=True)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    # 计算符合条件的点并累积到 density
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            # 计算卡方值（这里需要 rvm_model 和相关变量）
            psi_fit_chi = rvm_model(Angle_filtered, alpha=x, beta=y, 
                                    phase_offset=phase_offset, psi_offset=psi_offset)
            squared = np.sum(((PA_filtered - psi_fit_chi) / PA_err_filtered) ** 2)
            chi_red_value = squared / (N - 4)
            if Chi_min_value< chi_red_value < Chi_max_value :
                density[i, j] += 1  # 直接累加到 density

    # 处理数据，避免 log 归一化时遇到 0
    density[density == 0] = np.nan  # 避免 log(0) 变成负无穷
    # 定义 vmin 和 vmax 用于 log 归一化
    vmin = np.nanpercentile(density, 100)  # 1% 分位数，忽略极端小值
    vmax = np.nanpercentile(density, 100)  # 99% 分位数，忽略极端大值
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    # 显示原始的密度图（去掉平滑）
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(density.T, cmap="gray", origin="lower", aspect="auto", extent=extent, norm=norm)
    # 找到最佳的 alpha 和 beta（假设 best_alpha 和 best_beta 需要计算）
    # 标出最佳的 alpha 和 beta
    ax.grid(False)
    ax.scatter(best_alpha, best_beta, color="red", marker="x", s=10)
    ax.text(best_alpha + 3, best_beta, f"({best_alpha:.3f}, {best_beta:.3f})", 
            color="red", fontsize=7, ha="left", va="bottom")

    # 显示卡方值最小值
    ax.text(0.02, 0.98, 
        f"$\\text{{best: }}\\ \\chi_{{red}}$ = {best_chi_red:.2f}\n"
        f"$\\text{{min: }}\\ \\chi_{{red}}$ = {Chi_min_value:.2f}\n"
        f"$\\text{{max: }}\\ \\chi_{{red}}$ = {Chi_max_value:.2f}", 
        color="black", fontsize=8, fontweight="bold", ha="left", va="top", 
        transform=ax.transAxes, bbox=dict(facecolor='grey', alpha=0.35, edgecolor='none'))
    # 设置坐标轴标签
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    # 保存图像
    fig.savefig("参数估计图.png", dpi=300, bbox_inches='tight', transparent=True)
    print("===========================================================================================") 



    def plot_data(data, x_min, x_max, phi, model,psr, x_max_slope=None, y_max_slope=None):
        # 设置默认的刻度线方向为向内
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # 创建图形和网格布局
        fig = plt.figure(figsize=(8, 8), dpi=800)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.2,0.6,1], hspace=0)   # Set hspace to 0 to remove space between subplots
        # 创建子图
        ax1 = fig.add_subplot(gs[0])  # Top subplot for Polarization Angle
        ax2 = fig.add_subplot(gs[1])  # Bottom subplot for Flux
        ax3 = fig.add_subplot(gs[2])  # Bottom subplot for Flux
         # Bottom subplot for Flux
        # 强制设置刻度方向为向内
        ax1.tick_params(axis='x', direction='in')
        ax1.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='x', direction='in')
        ax2.tick_params(axis='y', direction='in')
        ax3.tick_params(axis='x', direction='in')
        ax3.tick_params(axis='y', direction='in')

        ax1.plot(phi, model + 90, linewidth=1, linestyle='--', color='green', alpha=0.8)
        ax1.plot(phi, model - 90, linewidth=1, linestyle='--', color='green', alpha=0.8)
        ax1.plot(phi, model + 180, linewidth=1, linestyle='--', color='green', alpha=0.8)
        ax1.plot(phi, model - 180, linewidth=1, linestyle='--', color='green', alpha=0.8)


        def plot_pa_with_error(ax, angle, pa, pa_error, label, color, pa_replace=None):
            if pa_replace is not None:
                replace_condition, replace_value = pa_replace
                pa = np.where(pa == replace_condition, replace_value, pa)
            pa = np.where(pa == 0, np.nan, pa)
            ax.errorbar(angle, pa, xerr=0.1, yerr=pa_error, fmt='o',color=color,markersize=4,capsize=2,elinewidth=1, label=label)

        # 绘制偏振角数据和拟合结果
        plot_pa_with_error(ax1, data['Angle'], data['PA'], data['PA_error'], 'Original Data', color='grey')
        plot_pa_with_error(ax1, data['Angle'], data['PA_filtered'], data['PA_error_filtered'], 'Fitting Data', color='black')
        overlap_mask = ~np.isnan(data['PA']) & ~np.isnan(data['PA_filtered']) & (data['PA'] == data['PA_filtered'])
        angle_overlap = data['Angle'][overlap_mask]
        pa_overlap = data['PA'][overlap_mask]
        if len(pa_overlap) != 0:
            pa_overlap[pa_overlap == 0] = np.nan
            pa_error_overlap = data['PA_error'][overlap_mask]
            ax1.errorbar(angle_overlap, pa_overlap, xerr=0.1, yerr=pa_error_overlap, fmt='o', color='red',label = "Overlapping data", markersize=4, capsize=2, elinewidth=1)
       
       
        ax1.plot(phi, model, linewidth=2, linestyle='-', color='green', label=f"$\\chi^2_{{\\nu}}$ = {best_chi_red:.2f}", alpha=1)
        ax1.set_title(f"PSR {psr}", fontsize=16, pad=12, fontweight='bold')
        ax1.set_ylabel('PA (deg)', fontsize=12)
        ax1.set_ylim(min(model)-50, max(model)+50) 
        ax1.set_xlim(x_min, x_max)
        ax1.axvline(best_phase, color='m', linestyle='--', linewidth=0.8)
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(False)
        ax1.legend(loc='upper right', fontsize=6, framealpha=0.8)
       # ax1.legend(loc='best', fontsize=8, framealpha=1)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.tick_params(axis='both', which='minor', labelsize=10)
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))     
        mask = (data['Angle'] >= x_min) & (data['Angle'] <= x_max)
        data_selected = data[mask]
        model = rvm_model(data_selected['Angle'],alpha=best_alpha,beta=best_beta,phase_offset=best_phase,psi_offset=best_psi)
        residuals = model - data_selected['PA_filtered']
        ax2.errorbar(data_selected['Angle'],residuals,yerr=data_selected['PA_error_filtered'],fmt='o',color='#4B0082',markersize=4,capsize=2,elinewidth=1,)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(residuals.min() - 10, residuals.max() + 10)
        ax2.set_ylabel('Res(deg)', fontsize=12,labelpad=12)
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.tick_params(axis='both', which='minor', labelsize=10)
        ax2.xaxis.set_visible(False)


        # 筛选有效（非 NaN）PA_filtered 对应的 Angle 值
        valid_mask = ~np.isnan(data['PA_filtered'])
        wask = (data['Angle'] >= x_min) & (data['Angle'] <= x_max)
        valid_angles = data['Angle'][valid_mask&wask]

        # 最小值、最大值及其绝对差值（非负）
        min_angle = valid_angles.min()
        max_angle = valid_angles.max()
        
        PW_range1 = max_angle - best_phase  # 保证非负
        print(PW_range1)
        PW_range2 = best_phase - min_angle # 保证非负
        print(PW_range2)
        PW_range = max(PW_range1, PW_range2)
        print(PW_range)
  



        print("最小 Angle:", min_angle)
        print("最大 Angle:", max_angle)
        print("最大和最小 Angle 之间的距离（非负）:", PW_range)



        angle_mask = (data['Angle'] >= 50) & (data['Angle'] <= 110)
        std_I = np.std(data["I_normalized"][angle_mask]) 
        # 构建一个 mask：总强度大于 2σ
        intensity_mask = data['I_normalized'] > 3 * std_I
        linear_pol_mask = data['Linear_Polarization'] >= 2* std_I
        non_negative_I = data['I_normalized'] >= 0

        # 构建最终 mask
        final_L_mask = intensity_mask & linear_pol_mask & non_negative_I 

        # 绘制 L/I

        V_pol_mask = abs(data['V_normalized']) >= 2* std_I
        final = intensity_mask & V_pol_mask  # 如果你希望也限定角度范围
   
        print("std_I =", std_I)
        print("2σ =", 2 * std_I)

        ax3.plot(data['Angle'], data['I_normalized'], color='black', linewidth=2, label='I')
        ax3.plot(data['Angle'], data['Linear_Polarization'], color='blue', linewidth=2, label='L')
        ax3.plot(data['Angle'], data['V_normalized'], color='red', linewidth=2, label='V')

        Imax = data['I_normalized'].max()
        threshold = Imax * 0.1
        # 先选出角度在 [-25, -15] 范围内的行
        anglemask = (data['Angle'] >= x_min) & (data['Angle'] <= x_max)
        # 再在这个子集里筛出 I_normalized >= threshold 的行
        mask = anglemask & (data['I_normalized'] >= threshold)& (data['I_normalized'] >= 3*std_I)
        subset = data[mask]
   
        # 输出这些点在 Angle 上的最大和最小值（或你需要的其他列）
        angle_min = subset['Angle'].min()
        angle_max = subset['Angle'].max()
        print(angle_min,angle_max)

        Imin = data.loc[data['Angle'] == angle_min, 'I_normalized'].values[0]
        Imax = data.loc[data['Angle'] == angle_max, 'I_normalized'].values[0]
        ax3.errorbar(x=angle_min,y=Imin,yerr=0.1,fmt='o', color='m',markersize=0.01,elinewidth=2,capsize=0, linestyle='--', linewidth=0.8)
        ax3.errorbar(x=angle_max,y=Imax,yerr=0.1,fmt='o', color='m',markersize=0.01,elinewidth=2,capsize=0, linestyle='--', linewidth=0.8)
        w10 = angle_max - angle_min
        print("w10宽度",w10)
        midpoint = (angle_min + angle_max) / 2
        print("w10zhong点",midpoint)  # 输出 5.0
        ax3.axvline(midpoint, color='m', linestyle='--', linewidth=0.8)
        w10_center = midpoint


        q = psrqpy.QueryATNF(params=["NAME","PSRJ","PSRB", "P0", "P1", "TYPE", "BINARY", "ASSOC", "P1_I"])  
        t = q.table

        psr_data = t[t['NAME'] == psr]
        if len(psr_data) == 0:
            psr_data = t[t['PSRJ'] == psr]
            if len(psr_data) == 0:
                psr_data = t[t['PSRB'] == psr]

        P= psr_data["P0"].data[0]*1000
        print("脉冲星周期：",P)
        tb  = P/(len(data["Angle"])-1)
        print("bin数",len(data["Angle"])-1)
        w10_error = tb * math.sqrt(1 + (std_I / 0.1) ** 2)
        print("w10误差",w10_error)

        # 假设你想画的矩形中心点为 (x, y)，宽度为 w，高度为 h
        x = x_min+2     # 例如角度
        y = 0.4       # 例如线性偏振值
        w = 360/(len(data["Angle"])-1) # 矩形宽度（x方向）
        h = 3*std_I      # 矩形高度（y方向）
        # Rectangle 需要左下角的坐标，因此需要从中心坐标计算
        lower_left_x = x - w/2
        lower_left_y = y - h/2
        rect = patches.Rectangle((lower_left_x, lower_left_y), w, h,linewidth=0.8, edgecolor='black', facecolor='none', linestyle='-')
        ax3.add_patch(rect)



        ax3.set_ylabel('Intensity', fontsize=12,labelpad=12)
        ax3.set_xlabel('Longitude(deg)', fontsize=10)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(-0.18, 1.1)
        ax3.grid(False)
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.tick_params(axis='both', which='minor', labelsize=10)
        ax3.legend(loc='upper right', fontsize=8, framealpha=1)
 
        mask = data["I_normalized"] >= 3 * std_I
        Nd = mask.sum()
        print("I大于3sigma的点:",Nd)
        I_sum, L_sum, V_sum, absV_sum = data["I_normalized"][mask].sum(), data["Linear_Polarization"][mask].sum(), data["V_normalized"][mask].sum(), np.abs(data["V_normalized"][mask]).sum()
        L_I, V_I, absV_I = L_sum / I_sum, V_sum / I_sum, absV_sum / I_sum
        σI, σL, σV = std_I, np.std(data["Linear_Polarization"][angle_mask]), np.std(data["V_normalized"][angle_mask])
        σ_LI = np.sqrt(Nd * ((σL/I_sum)**2 + (L_sum*σI/I_sum**2)**2))
        σ_VI = np.sqrt(Nd * ((σV/I_sum)**2 + (V_sum*σI/I_sum**2)**2))
        σ_absVI = np.sqrt(Nd * ((σV/I_sum)**2 + (absV_sum*σI/I_sum**2)**2))

        print(f"L/I = {L_I:.4f} ± {σ_LI:.4f}")
        print(f"V/I = {V_I:.4f} ± {σ_VI:.4f}")
        print(f"|V|/I = {absV_I:.4f} ± {σ_absVI:.4f}")
        
        plt.savefig(psr + "_RVMFIT.png", dpi=700, facecolor='white', edgecolor='black')
        plt.savefig(psr + "_RVMFIT.pdf")
        return w10, w10_error, w10_center,L_I,σ_LI,V_I,σ_VI,absV_I,σ_absVI,PW_range


    phi_fit = np.linspace(all_min, all_max, 1000) 
    model = rvm_model(phi_fit, alpha=best_alpha, beta=best_beta, phase_offset=best_phase, psi_offset=best_psi)
    w10, w10_error, w10_center,L_I,σ_LI,V_I,σ_VI,absV_I,σ_absVI,PW_range = plot_data(data=data, x_min=all_min, x_max=all_max, phi=phi_fit, model=model, psr=psr)



    best_alpha = best_alpha
    alpha_min = min(posterior_sample["alpha"])
    alpha_max = max(posterior_sample["alpha"])
    best_alpha_lower = abs(best_alpha - alpha_min)
    best_alpha_upper = abs(best_alpha - alpha_max)
    
    best_beta =  best_beta
    beta_min = min(posterior_sample["beta"])
    beta_max = max(posterior_sample["beta"]) 
    best_beta_lower =  abs(best_beta - beta_min)
    best_beta_upper =  abs(best_beta - beta_max)

   
    best_phase_offset = best_phase
    phase_upper = min(posterior_psi_phi["phase_offset"])
    phase_lower = max(posterior_psi_phi["phase_offset"])
    phase_offset_upper = abs(phase_upper - best_phase)
    phase_offset_lower = abs(phase_lower - best_phase)

    best_psi_offset = best_psi
    psi_lower = min(posterior_psi_phi["psi_offset"])
    psi_upper = max(posterior_psi_phi["psi_offset"])
    psi_offset_upper = abs(psi_upper - best_psi)
    psi_offset_lower = abs(psi_lower - best_psi)



    def compute_rho_with_uncertainties(alpha_deg, alpha_err_up, alpha_err_down,
                                        beta_deg, beta_err_up, beta_err_down,
                                        W10_deg, d_W10_deg):
        """
        计算束开角 rho 及其上下误差。

        参数：
            alpha_deg       - alpha（度）
            alpha_err_up    - alpha 上误差（度）
            alpha_err_down  - alpha 下误差（度）
            beta_deg        - beta（度）
            beta_err_up     - beta 上误差（度）
            beta_err_down   - beta 下误差（度）
            W10_deg         - W10 脉冲宽度（度）
            d_W10_deg       - W10 的对称误差（度）

        返回：
            rho, drho_up, drho_down - 束开角（度）及其上下误差
        """

        def calc_rho(alpha_d, beta_d, W10_d):
            # 角度 -> 弧度
            alpha = alpha_d * math.pi / 180
            beta = beta_d * math.pi / 180
            W10 = W10_d * math.pi / 180

            sin_alpha = math.sin(alpha)
            sin_zeta = math.sin(alpha + beta)
            sin_W10_4 = math.sin(W10 / 4)
            sin_beta_2 = math.sin(beta / 2)

            term1 = (sin_W10_4 ** 2) * sin_alpha * sin_zeta
            term2 = sin_beta_2 ** 2
            sqrt_term = math.sqrt(term1 + term2)
            rho_rad = 2 * math.asin(min(1.0, sqrt_term))  # 安全限制
            return rho_rad * 180 / math.pi  # 弧度 -> 角度

        # 中心值
        rho_center = calc_rho(alpha_deg, beta_deg, W10_deg)

        # 每个方向扰动后的值
        rho_alpha_up = calc_rho(alpha_deg + alpha_err_up, beta_deg, W10_deg)
        rho_alpha_down = calc_rho(alpha_deg - alpha_err_down, beta_deg, W10_deg)
        rho_beta_up = calc_rho(alpha_deg, beta_deg + beta_err_up, W10_deg)
        rho_beta_down = calc_rho(alpha_deg, beta_deg - beta_err_down, W10_deg)
        rho_W10_up = calc_rho(alpha_deg, beta_deg, W10_deg + d_W10_deg)
        rho_W10_down = calc_rho(alpha_deg, beta_deg, W10_deg - d_W10_deg)

        # 各方向误差
        drho_alpha_up = abs(rho_alpha_up - rho_center)
        drho_alpha_down = abs(rho_center - rho_alpha_down)
        drho_beta_up = abs(rho_beta_up - rho_center)
        drho_beta_down = abs(rho_center - rho_beta_down)
        drho_W10 = abs(rho_W10_up - rho_W10_down) / 2

        # 合成总误差（平方和）
        drho_up = math.sqrt(drho_alpha_up**2 + drho_beta_up**2 + drho_W10**2)
        drho_down = math.sqrt(drho_alpha_down**2 + drho_beta_down**2 + drho_W10**2)

        return rho_center, drho_up, drho_down

    rho, drho_up, drho_down = compute_rho_with_uncertainties(
        alpha_deg= best_alpha, 
        alpha_err_up= best_alpha_upper, 
        alpha_err_down= best_alpha_lower,
        beta_deg=best_beta, 
        beta_err_up= best_beta_upper, 
        beta_err_down=best_beta_lower,
        W10_deg=w10, 
        d_W10_deg=w10_error
    )

    print(f"束开角 rho = {rho:.2f} +{drho_up:.2f}/-{drho_down:.2f} 度")



    # 初始化变量
    information = []  # 用于存储 information.csv 数据
    information_dict = {"PSR": psr}  # 用字典存储每个 psr 的参数估计值和误差
    q = psrqpy.QueryATNF(params=["NAME","PSRJ","PSRB", "P0", "P1", "TYPE", "BINARY", "ASSOC", "P1_I"])  
    t = q.table
    # 查找目标脉冲星的数据
    pulsarname = 'NAME'
    psr_data = t[t['NAME'] == psr]

    if len(psr_data) == 0:
        pulsarname = "PSRJ"
        psr_data = t[t['PSRJ'] == psr]
        if len(psr_data) == 0:
            pulsarname = "PSRB"
            psr_data = t[t['PSRB'] == psr]
            
    period0 = psr_data["P0"].data[0]
    period1 = psr_data["P1"].data[0]
    information_dict["P0"] = period0
    information_dict["P-dot"] = period1
    information_dict["best_alpha"] = best_alpha
    information_dict["alpha_Upper_Error"] = best_alpha_upper
    information_dict["alpha_Lower_Error"] = best_alpha_lower
    information_dict[f"best_beta"] = best_beta
    information_dict[f"beta_upper_Error"] = best_beta_upper
    information_dict[f"beta_lower_Error"] = best_beta_lower
    information_dict[f"best_phase_offset"] = best_phase
    information_dict[f"phase_offset_upper_error"] = phase_offset_upper
    information_dict[f"phase_offset_lower_error"] = phase_offset_lower
    information_dict[f"best_psi_offset"] = best_psi
    information_dict[f"psi_offset_upper_error"] = psi_offset_upper
    information_dict[f"psi_offset_lower_error"] = psi_offset_lower
    information_dict[f"best_chi_red"] = best_chi_red
    information_dict["Chi_upper_error"] = Chi_upper_error
    information_dict["Chi_lower_error"] =Chi_lower_error
    information_dict["best_K"] = best_K
    information_dict["K_upper_error"] = K_upper_error
    information_dict["K_lower_error"] = K_lower_error
    information_dict["w10"] = w10
    information_dict["w10_error"] = w10_error
    information_dict["w10_center"] = w10_center
    information_dict["L_I"] = L_I
    information_dict["σ_LI"] = σ_LI
    information_dict["V_I"] = V_I
    information_dict["σ_VI"] = σ_VI
    information_dict["absV_I"] = absV_I
    information_dict["σ_absVI"] = σ_absVI

    information_dict["rho"] = rho
    information_dict["drho_up"] = drho_up
    information_dict["drho_down"] = drho_down
    information_dict["PW"] = PW_range
 


    information.append(information_dict)
    print(information)
    
    # 动态生成表头，包含所有参数和 chi_squared_red
    fieldnames = [
        "PSR", "P0", "P-dot",
        "best_alpha", "alpha_Upper_Error", "alpha_Lower_Error",
        "best_beta", "beta_upper_Error", "beta_lower_Error", 
        "best_phase_offset", "phase_offset_upper_error", "phase_offset_lower_error",
        "best_psi_offset", "psi_offset_upper_error", "psi_offset_lower_error",
        "best_chi_red", "Chi_upper_error","Chi_lower_error",
        "best_K", "K_upper_error", "K_lower_error",
        "w10","w10_error","w10_center",
        "L_I","σ_LI","V_I","σ_VI","absV_I","σ_absVI",
        "rho","drho_up","drho_down","PW"
        ]
    information_filename = "../information.csv"
    # 读取现有的 information.csv 文件内容
    if os.path.exists(information_filename):
        with open(information_filename, mode='r', newline='') as info_file:
            reader = csv.DictReader(info_file)
            existing_data = [row for row in reader]
    else:
        existing_data = []
    # 查找是否已经存在对应的 psr
    psr_exists = False
    for row in existing_data:
        if row["PSR"] == information_dict["PSR"]:
            # 如果存在，更新现有行
            row.update(information_dict)
            psr_exists = True
            break
    # 如果 psr 不存在，添加新的一行
    if not psr_exists:
        existing_data.append(information_dict)
    # 将更新后的数据写回 information.csv
    with open(information_filename, mode='w', newline='') as info_file:
        writer = csv.DictWriter(info_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)
    print("===========================================================================================")    
    print(f"Information saved to {information_filename}")






