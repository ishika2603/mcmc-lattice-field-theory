''' Implementation of a Lattice scalar phi^4 field theory. 
    Following implementation from - https://hef.ru.nl/~tbudd/mct/lectures/lattice_field_theory.html#equation-twopointcorrelator
'''
import numpy as np
import matplotlib.pylab as plt

rng = np.random.default_rng()

# def potential_v(x,lamb):
#     '''Compute the potential function V(x).'''
#     return lamb*(x*x-1)*(x*x-1)+x*x

# xrange = np.linspace(-1.5,1.5,41)
# lambdas = [0.5,1,2,4]
# for l in lambdas:
#     plt.plot(xrange,potential_v(xrange,l))
# plt.ylim(0,5)
# plt.legend([r"$\lambda = {}$".format(l) for l in lambdas])
# plt.xlabel(r"$\varphi$")
# plt.ylabel("V")
# plt.show()

def potential_v(x,lamb):
    '''Compute the potential function V(x).'''
    return lamb*(x*x-1)*(x*x-1)+x*x

def neighbor_sum(phi,s):
    '''Compute the sum of the state phi on all 8 neighbors of the site s.'''
    w = len(phi)
    return (phi[(s[0]+1)%w,s[1],s[2],s[3]] + phi[(s[0]-1)%w,s[1],s[2],s[3]] +
            phi[s[0],(s[1]+1)%w,s[2],s[3]] + phi[s[0],(s[1]-1)%w,s[2],s[3]] +
            phi[s[0],s[1],(s[2]+1)%w,s[3]] + phi[s[0],s[1],(s[2]-1)%w,s[3]] +
            phi[s[0],s[1],s[2],(s[3]+1)%w] + phi[s[0],s[1],s[2],(s[3]-1)%w] )

def scalar_action_diff(phi,site,newphi,lamb,kappa):
    '''Compute the change in the action when phi[site] is changed to newphi.'''
    return (2 * kappa * neighbor_sum(phi,site) * (phi[site] - newphi) +
            potential_v(newphi,lamb) - potential_v(phi[site],lamb) )

def scalar_MH_step(phi,lamb,kappa,delta):
    '''Perform Metropolis-Hastings update on state phi with range delta.'''
    site = tuple(rng.integers(0,len(phi),4))
    newphi = phi[site] + rng.uniform(-delta,delta)
    deltaS = scalar_action_diff(phi,site,newphi,lamb,kappa)
    if deltaS < 0 or rng.uniform() < np.exp(-deltaS):
        phi[site] = newphi
        return True
    return False

def run_scalar_MH(phi,lamb,kappa,delta,n):
    '''Perform n Metropolis-Hastings updates on state phi and return number of accepted transtions.'''
    total_accept = 0
    for _ in range(n):
        total_accept += scalar_MH_step(phi,lamb,kappa,delta)
    return total_accept

def batch_estimate(data,observable,k):
    '''Devide data into k batches and apply the function observable to each.
    Returns the mean and standard error.'''
    batches = np.reshape(data,(k,-1))
    values = np.apply_along_axis(observable, 1, batches)
    return np.mean(values), np.std(values)/np.sqrt(k-1)

lamb = 2.0
kappas = np.linspace(0.08,0.18,11)
width = 3
num_sites = width**4
delta = 1.5  # chosen to have ~ 50% acceptance
equil_sweeps = 1000
measure_sweeps = 2
measurements = 4000

mean_magn = []
for kappa in kappas:
    phi_state = np.zeros((width,width,width,width))
    run_scalar_MH(phi_state,lamb,kappa,delta,equil_sweeps * num_sites)
    magnetizations = np.empty(measurements)
    for i in range(measurements):
        run_scalar_MH(phi_state,lamb,kappa,delta,measure_sweeps * num_sites)
        magnetizations[i] = np.mean(phi_state)
    mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
    mean_magn.append([mean,err])
    
plt.errorbar(kappas,[m[0] for m in mean_magn],yerr=[m[1] for m in mean_magn],fmt='-o')
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$|m|$")
plt.title(r"Absolute field average on $3^4$ lattice, $\lambda = 1.5$")
plt.show()

