import numpy as np

def kl_divergence_normal(mu_p, sigmap, mu_q, sigmaq):
	return np.log(sigmaq/sigmap) - 1/2 + (sigmap**2 + (mu_p-mu_q)**2)/(2*sigmaq**2)
