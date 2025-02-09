import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def ej3_plots(title, data):
    # Extracción del primer dígito
    data = data.apply(str).str.extract(r'([1-9])')[0]
    data = data.apply(int)

    # Nornalización de datos empiricos
    N = data.size
    x, y = np.histogram(data, bins=np.arange(1, 11))
    empirical_prob = x/N
    digits = np.arange(1, 10)

    # Datos teóricos (ley de Benford)
    theorical_prob = np.log10(1+1/digits)

    # ---- Figura: ---- #
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
    # Masa de probabilidad comparada
    ax[0,0].set_title(f'Masa de Probabilidad - {title}')
    ax[0,0].set_xlabel('Primera Cifra')
    ax[0,0].set_ylabel('Probabilidad')
    ax[0,0].step(digits, empirical_prob,'--', label='Muestra', color='b')
    ax[0,0].step(digits, theorical_prob,'ro-', label='Teórico')
    ax[0,0].legend()
    
    # Probabilidad acumulada comparada
    ax[0,1].set_title(f'Probabilidad Acumultiva - {title}')
    ax[0,1].set_xlabel('Primera Cifra')
    ax[0,1].set_ylabel('Probabilidad Acumulada')
    ax[0,1].step(digits, empirical_prob.cumsum(),'--', label='Muestra', color='b')
    ax[0,1].step(digits, theorical_prob.cumsum(),'ro-', label='Teórico')
    ax[0,1].legend()
    
    # PP plot
    ax[1,0].set_title(f'PP Plot - {title}')
    ax[1,0].set_xlabel('Probabilidad Acumulada Teórica')
    ax[1,0].set_ylabel('Probabilidad Acumulada Experimental')
    ax[1,0].scatter(x=theorical_prob.cumsum(), y=x.cumsum()/N)
    ax[1,0].plot([0,1],[0,1],'r-')
    
    # qq plot
    ax[1,1].set_title(f'QQ plot - {title}')
    # \\ Cálculo de cuantiles
    theorical_quantiles = np.percentile(theorical_prob, np.linspace(0, 100, 9))
    empirical_quantiles = np.percentile(empirical_prob, np.linspace(0, 100, 9))
    ax[1,1].scatter(x=theorical_quantiles,y=empirical_quantiles)
    ax[1,1].set_xlabel('Cuantiles Teóricos')
    ax[1,1].set_ylabel('Cuantiles Experimentales')
    min, max = theorical_quantiles.min(), theorical_quantiles.max()
    ax[1,1].plot([min,max],[min,max],'r-')

    plt.show()

def ej3_ks(title, data):
    # Extracción del primer dígito
    data = data.apply(str).str.extract(r'([1-9])')[0]
    data = np.array(data.apply(int))

    # Distribuciones
    digits = np.arange(1, 10)
    theorical_prob = np.log10(1+1/digits)   # P Masa
    benford_cdf = np.cumsum(theorical_prob) # P Acumulada

    # Kolmogorov-Smirnov
    print(title)
    d_ks, p_ks = st.ks_1samp(x=data,cdf=(lambda x: benford_cdf[x-1]))
    print(f"K-S: {d_ks:.4f}")
    print(f"P: {p_ks:.10f}")

    # Chi Cuadrado
    observed_counts = np.array([(data == d).sum() for d in digits])
    expected_counts = theorical_prob*len(data)
    chi2, p_chi2 = st.chisquare(f_obs=observed_counts, f_exp=expected_counts)
    print(f"Chi Cuadrado: {chi2:.4f}")
    print(f"P: {p_chi2:.4f}")