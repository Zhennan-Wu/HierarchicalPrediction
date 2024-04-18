import torch
import pyro


# Notation aligned with the paper "Hierarchical Dirichlet Processes"

J = 8 # number of restaurants
K = 3 # number of dishes
I = 100 # number of observations
T = 10 # number of tables
H = 20 # dimension of the topic
phi_domain = None # domain of the parameter phi

vec_m = torch.zeros(J, K) # m is the number fo dishes in each restaurant
vec_z = torch.zeros(J, I) # z is the dish of the observation
vec_x = torch.zeros(J, I) # x is the observed data
vec_t = torch.zeros(J, I) # t is the table of the observed data
vec_k = torch.zeros(J, T) # k is the dish of the table
vec_n = torch.zeros(J, T, K) # n is the number of customers in the table t with dish k
vec_phi = torch.zeros(K, H) # phi is the parameter of the dish
vec_psi = torch.zeros(J, T) # psi is the dish of the table

gamma = None
alpha0 = None

def calc_x_jt(vec_x, vec_t, j, t):
    '''
    Calculate x_jt in the paper
    '''
    indices = torch.nonzero(torch.eq(vec_t, t)).squeeze()
    rlt = vec_x[j, indices]
    return rlt

def density(x, theta):
    '''
    '''
    pass

def h(phi):
    '''
    '''
    pass

def calc_prod_fh(vec_x, indices, phi, j, i):
    '''
    '''
    rlt = 1
    for k in indices:
        if k != (j, i):
            rlt *= density(vec_x[k], phi)*h(phi)
    return rlt

def calc_prior_fh(x):
    '''
    '''
    rlt = 1
    for phi in phi_domain:
        rlt *= density(x, phi)*h(phi)
    return rlt

def calc_pos_f(vec_x, vec_z, j, i, k, x):
    '''
    '''
    nominator = 0
    denominator = 0
    for phi in phi_domain:
        indices = torch.nonzero(torch.eq(vec_z, k)).squeeze()
        prod = calc_prod_fh(vec_x, indices, phi, j, i)
        nominator += density(x, phi)*prod
        denominator += prod
    return nominator/denominator

def calc_pos_x(x, j, i, t_new):
    '''
    '''
    rlt = 0
    m = torch.sum(vec_m)
    for k in range(K):
        mk = torch.sum(vec_m[:, k])
        rlt += mk/(m+gamma)*calc_pos_f(vec_x, vec_z, j, i, k, x) + gamma/(m+gamma)*calc_prior_fh(x)
    return rlt

def calc_pos_t(t, j, i):
    '''
    '''
    if (t in vec_t):
        return torch.sum(vec_n[j, t])*calc_pos_f(vec_x, vec_z, j, i, k, vec_x[j, i])
    else:
        return alpha0*calc_pos_x(vec_x[j, i], j, i, t)

def calc_pos_k(k, j, i, t):
    '''
    '''
    if (k in vec_k):
        return torch.sum(vec_m[:, k])*calc_pos_f(vec_x, vec_z, j, i, k, vec_x[j, i])
    
    else:
        return gamma * calc_prior_fh(vec_x[j, i])


 


