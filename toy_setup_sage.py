# -*- coding: utf-8 -*-

# %% Imports

import inspect
import sys                                                                
sys.path.insert(1, '/home/anton/')                                        
import Zeta 

from Zeta.convex import PositiveOrthant
from Zeta.toric import ToricDatum

class ToricDatumAlgebra:
    def __init__(self, T):
        self._toric_datum = T.simplify()
        self.rank = 0

    def toric_datum(self, objects=None, name=None):
        return self._toric_datum

    def find_good_basis(self, objects=None, name=None):
        return None

    def change_basis(self, A, check=True):
        return self



from Zeta.reps import RepresentationProcessor
from Zeta.ask import AskProcessor

from Zeta.convex import RationalSet
#from Zeta.toric import ToricDatum

from Zeta import logger, smurf, surf, torus, toric, abstract, cycrat, triangulate, reps, subobjects, ask, cico, addmany

from sage.all import *

import bisect
import numpy as np
import sympy as sp
import IPython
import random
import copy
import math
import torch
from torch import nn


# %% """# Helpful functions on polynomials"""

# functions I use to perform 'tricks' on lists of pairs of polynomials

def divide_if_possible(poly1, poly2, poly): # if possible, divide both polynomials by the same polynomial
    quotient1, remainder1 = poly1.quo_rem(poly)
    quotient2, remainder2 = poly2.quo_rem(poly)
    return (quotient1, quotient2) if (remainder1 == 0) & (remainder2 == 0) else (poly1, poly2)


def trick_divide_by(pairs_of_polynomials, poly): # apply the function above to all pairs in the list
    return [divide_if_possible(p1,p2, poly) for p1, p2 in pairs_of_polynomials]

def trick_change_var(pairs_of_polynomials, change_of_vars): #change of variable
    return [(p1.subs(change_of_vars), p2.subs(change_of_vars)) for p1, p2 in pairs_of_polynomials]
    

#def trick_blowup_cc(pairs_of_polynomials, var1, var2, variables, domain): #blowup between var1 and var2
#    cc1 = trick_divide_by(trick_change_var(pairs_of_polynomials, {var1:var1*var2},variables, domain),sp.Poly(var2, *variables, domain=domain))
#    cc2 = trick_divide_by(trick_change_var(pairs_of_polynomials, {var2:var2*var1},variables, domain),sp.Poly(var1, *variables, domain=domain))
#    return cc1, cc2

def trick_blowup_cc(pairs_of_polynomials, var1, var2): #blowup between var1 and var2
    return trick_divide_by(trick_change_var(pairs_of_polynomials, {var1:var1*var2}),var2), trick_divide_by(trick_change_var(pairs_of_polynomials, {var2:var2*var1}),var1)





def ith_solution(i, k, d):
    # ith solution of x_1 + ... x_k =d
    # i index starting from 0, k -- number of variables, d -- the sum
    n = d + k - 1  # Total positions
    result = []
    stars_left = d
    bars_left = k - 1
    pos = 0  # current position

    while bars_left > 0:
        count = 0
        # Try placing 0 to stars_left stars before next bar
        for x in range(stars_left + 1):
            ways = math.comb(stars_left - x + bars_left - 1, bars_left - 1)
            if count + ways > i:
                result.append(x)
                i -= count
                stars_left -= x
                bars_left -= 1
                break
            count += ways

    result.append(stars_left)  # Last variable gets remaining stars
    return result[::-1]


def get_monomial_from_index(i, n, d, variables):
    """
    Gets the i-th monomial (lexicographically ordered, degree 1 to d, coeff 1)
    on n variables from its index.

    Args:
        i (int): The index of the monomial (0-based).
        n (int): The number of variables.
        d (int): The maximum degree of the monomials.
        variables (list): A list of SymPy symbols (variables).

    Returns:
        sympy.core.mul.Mul: The corresponding SymPy monomial.
    """
    if i < 0:
        raise ValueError("Index must be non-negative.")

    # Calculate the number of monomials for each degree from 1 to d
    num_monomials_up_to_degree = [0] * (d + 1)
    for deg in range(1, d + 1):
        num_monomials_up_to_degree[deg] = math.comb(n + deg, deg) - 1  # Exclude degree 0 monomial

    # Find the degree of the monomial at index i
    current_index = i
    monomial_degree = 0
    for deg in range(1, d + 1):
        count_at_this_degree = num_monomials_up_to_degree[deg] - num_monomials_up_to_degree[deg - 1]
        if current_index < count_at_this_degree:
            monomial_degree = deg
            break
        current_index -= count_at_this_degree
    else:
        raise ValueError("Index out of range for given n and d.")

    # Find the starting index of monomials of this degree
    starting_index_of_degree = num_monomials_up_to_degree[monomial_degree - 1]

    # The index within the monomials of this degree
    index_within_degree = i - starting_index_of_degree

    # Now, convert the index within the degree to the monomial
    # This part requires a method to map an index to a lexicographically ordered combination
    # of variables for a fixed degree. SymPy doesn't have a direct function for this,
    # so you'd need to implement a combinatorial number system approach.

    # A simplified approach for demonstration (you might need a more robust implementation
    # for large n and d):
    exponents = [0] * n
    temp_index = index_within_degree
    remaining_degree = monomial_degree

    exponents = ith_solution(index_within_degree, n, monomial_degree)
    #monomial = sp.S.One
    #for j in range(n):
    #    monomial *= variables[j]**exponents[j]
    monomial = prod(var**e for var, e in zip(variables,exponents))

    return monomial

# %% Examples

r = 6
n = r*(r+1)//2

R = PolynomialRing(QQ, 'x', n)
R.inject_variables()

gens = list(R.gens())

# Define individual symbols for direct use in expressions
#for sym in gens:
#    globals()[sym.name] = sym
ith_solution(9, n, 4)

get_monomial_from_index(9, n, 4, gens)

# %% 

def trick_blowup_td(Td, var1, var2):
    ri = Td.ring
    indices = [i for i in range(Td.ambient_dim)]
    ind_dic = dict(zip(ri.gens(),indices))
    ini = Td.initials
    tcc = trick_blowup_cc(Td.cc, var1, var2)
    integ1 = copy.deepcopy(Td.integrand)
    integ1[0][ind_dic[var2]] = integ1[0][ind_dic[var2]] + integ1[0][ind_dic[var1]]
    integ1[1][ind_dic[var2]] = integ1[1][ind_dic[var2]] + integ1[1][ind_dic[var1]]

    integ2 = copy.deepcopy(Td.integrand)
    integ2[0][ind_dic[var1]] = integ2[0][ind_dic[var1]] + integ2[0][ind_dic[var2]]
    integ2[1][ind_dic[var1]] = integ2[1][ind_dic[var1]] + integ2[1][ind_dic[var2]]

    polyh1 = Td.polyhedron

    ineq_2 = [1 if j == (ind_dic[var2]+1) else 0 for j in range(Td.ambient_dim+1)]
    ineq_2[0]= -1
    polyh2 = Td.polyhedron & Polyhedron(ieqs = [ineq_2]) 
    
    td_out = []
    td_out.append(ToricDatum(ri, integ1, tcc[0], None, polyh1))
    td_out.append(ToricDatum(ri, integ2, tcc[1], None, polyh2))
    
    return td_out
    #return [ToricDatum(ri, integ1, tcc[0], None, polyh1), ToricDatum(ri, integ2, tcc[1], None, polyh2)] 

def trick_divide_td(Td, poly):
    return ToricDatum(Td.ring, Td.integrand, trick_divide_by(Td.cc,poly), None, Td.polyhedron)

def trick_changevar_td(Td, change_of_vars):
    conds=trick_change_var(Td.cc, change_of_vars)
    return ToricDatum(Td.ring, Td.integrand, conds, None, Td.polyhedron)



# %% free variables and auto change of variables

def free_variables_td(Td):
    not_free_var = set()
    all_var= list(Td.ring.gens())
    for monomial, _ in Td.cc:
            not_free_var.update(list(monomial.variables()))
    
    for i in range(0, len(all_var)):
      if (Td.integrand[0][i] != 0) or (Td.integrand[1][i] != -1):
        not_free_var.add(all_var[i])
    free_var = (set(all_var) - not_free_var)
    return free_var    

def trick_autochenge_td(TD):
    free_var = free_variables_td(TD)  
    for x in free_var:
        #print(x)
        for g, f in TD.cc:
            terms_as_polynomials = [c * m for c, m in zip(f.coefficients(), f.monomials())]
            for t in terms_as_polynomials:
                if t.degree(x) == 1:
                    #print(x,t)
                    t_over_x, _ = t.quo_rem(x)
                    # Ensure t_over_x is not zero to avoid division by zero
                    if t_over_x != 0:
                        #quotient, remainder = f.quo_rem(t_over_x)
                        f_min_t_terms = [term for term in terms_as_polynomials if term != t]
                        quotient_terms = []
                        for tr in f_min_t_terms:
                            quo, rem = tr.quo_rem(t_over_x)
                            if rem == 0:
                                quotient_terms.append(quo)
                        quotient = sum(quotient_terms, TD.ring(0))
                        if quotient.degree() > 0 and quotient.degree(x) ==0:
                          # Construct the change of variable
                          change = {x: x - quotient}
                          #print(change)
                          new_cc = trick_change_var(TD.cc, change)
                          new_TD = ToricDatum(TD.ring, TD.integrand, new_cc, None, TD.polyhedron)
                          # Check if the weight is reduced
                          #print(new_TD.weight(),TD.weight())
                          if new_TD.weight() < TD.weight():
                              return new_TD, True, change  # Return the improved state

    return TD, False, None # Return the original state if no beneficial change is found


# %% cone conditions from polyhedron function

def cc_from_pol(pol, var):
    M = matrix([ineq.A() for ineq in pol.inequalities()])
    pairs = []
    for row in M.rows():
        X = prod(var[i]**abs(val) for i, val in enumerate(row) if val < 0)
        Y = prod(var[i]**abs(val) for i, val in enumerate(row) if val > 0)
        if X != 1 and Y != 1:  # Only keep non-trivial inequalities
            pairs.append((X, Y))
    return pairs

def pol_to_cc_td(TD):   # function to push polyhedron inequolities to cone conditions
    var = TD.ring.gens()
    pol = TD.polyhedron
    new_cc = TD.cc + cc_from_pol(pol,var)
    for init in TD.initials:
        if init != None:
            raise ValueError("Initials are not None")
            
    return ToricDatum(TD.ring, TD.integrand, new_cc, None, PositiveOrthant(len(var)))

#cc_from_polyh == cc_from_pol(T.polyhedron, R.gens())            
#T_new = pol_to_cc_td(T)
#print(T_new)            


# %%  Random polynomials

#Below are the functions to generate random monomials and random polynomials 
#on given set of variables. Both eaither over $\mathbb{Z}$ or $\mathbb{F}_{32003}.$ If `coefficient_ring` is `None`, then monomials and polynomials are generated over $\mathbb{Z}$ with all coefficients set to 1. Monomials have given degree, polynomials age defined to be of degree at most the one that is given as a parameter.


def random_mon(varnames, deg, coefficient_ring=GF(32003), coefficient_range=None):
    """
    Generates a random monomial of a given degree using SageMath.

    Args:
        varnames (list or str): A list of variable names or a string like 'x y z'.
        deg (int): Desired total degree of the monomial.
        coefficient_ring: SageMath ring for coefficients (default: GF(32003)).
        coefficient_range (tuple, optional): (min, max) range for integer coefficients if using ZZ.

    Returns:
        A SageMath monomial expression.
    """
    if deg < 0:
        raise ValueError("Degree must be non-negative.")
        
    if coefficient_ring is None:
        #print("yep")
        R = PolynomialRing(ZZ, varnames)
    else:
        R = PolynomialRing(coefficient_ring if coefficient_range is None else ZZ, varnames)
    #print("yep2")
    #R = PolynomialRing(coefficient_ring if coefficient_range is None else ZZ, varnames)
    vars = R.gens()
    n = len(vars)

    if deg == 0:
        return R(1)  # Monomial of degree 0 is 1

    # Generate a random exponent vector of total degree `deg`
    # Pick a random vector from all integer vectors of degree `deg` and `n` variables
    exponents = random.choice(IntegerVectors(deg, n))

    # Build the monomial from variables and exponents
    monomial_part = prod(v**e for v, e in zip(vars, exponents))

    # Assign a random coefficient
    if coefficient_range is not None:
        min_c, max_c = coefficient_range
        coeff = ZZ.random_element(min_c, max_c + 1)
    else:
        if coefficient_ring is None:
            coeff = ZZ.random_element(-1,  1)
            while coeff == 0:
                coeff = ZZ.random_element(-1,  1)  # Avoid 0 coefficient
        else:
            coeff = coefficient_ring.random_element()
            while coeff == 0:
                coeff = coefficient_ring.random_element()  # Avoid 0 coefficient

    return coeff * monomial_part

# %%

def random_poly(variables, max_degree, num_terms, coefficient_ring=GF(32003), coefficient_range=None):
    """
    Generates a random polynomial consisting of a sum of random monomials.

    Args:
        variables (list): A list of SymPy symbols (variables).
        max_degree (int): The maximum degree for the monomials in the polynomial.
        num_terms (int): The number of random monomials to include in the sum.
        coefficient_ring (sympy.polys.domains.Domain, optional): The coefficient ring.
                                                                 Defaults to sp.FF(32003).
        coefficient_range (tuple, optional): A tuple (min, max) for the range
                                             of random integer coefficients if
                                             coefficient_ring is None.

    Returns:
        sympy.core.expr.Expr: A SymPy expression representing the random polynomial.
    """
    #print("nt", num_terms)
    if max_degree < 0:
        raise ValueError("Maximum degree must be non-negative.")
    if num_terms < 0:
        raise ValueError("Number of terms must be non-negative.")

    polynomial_terms = []
    for _ in range(num_terms):
        # Randomly choose a degree between 1 and max_degree
        term_degree = random.randint(1, max_degree)

        # Generate a random monomial of the chosen degree
        if coefficient_ring:
            random_term = random_mon(variables, term_degree, coefficient_ring=coefficient_ring)
        else:
            random_term = random_mon(variables, term_degree, coefficient_ring=None, coefficient_range=coefficient_range)

        polynomial_terms.append(random_term)

    # Sum the generated terms
    return sum(polynomial_terms)

# %%"""## Examples of the use of above functions"""

#Examples
#r = 6
#n = r*(r+1)//2

#R = PolynomialRing(QQ, 'x', n)
#R.inject_variables()

#gens = list(R.gens())
#print(gens)
#print(random_mon(gens, 4, None))
#print(random_poly(gens, 5, 7 , None))

# %% """# ToyToric class and random_td 

#Rewriting the class ToyToric that uses sympy polynomials 
#into list of functions that will use ToricDatum class from Zeta

# %% get list of variables for TOricDatum

def toric_variables(TD):
    return list(TD.ring.gens())

# %% Example
"""
L = Zeta.lookup('Fil4') 
t = L.toric_datum('subalgebras')

toric_variables(t)
t.lhs
t.integrand

R = t.ring
R.inject_variables()
"""
# %%

def not_free_variables(TD):
    
    variables = toric_variables(TD)
    if not variables:
        return []

    not_free_var = []

    #for monomial in TD.lhs:
        # Convert PolyElement to an expression before getting free symbols
        #not_free_var.update(monomial.as_expr().free_symbols)

    for i in range(0, len(variables)):
      if (TD.integrand[0][i] != 0) or (TD.integrand[1][i] != -1):
        not_free_var.append(variables[i])

    return not_free_var

# %%

def free_variables(TD):
    """
    Returns a list of variables that do not appear in the first entry
    (the monomial) of each tuple in the cone_conditions.

    Returns:
        list: A list of SymPy symbols.
    """
    variables = toric_variables(TD)
    if not variables:
        return []

    all_variables_set = set(variables)
    not_free_var = set(not_free_variables(TD))

    # Find the difference between all variables and variables in monomials
    free_var = list(all_variables_set - not_free_var)



    return free_var

# %%

def get_data_for_NN(TD, max_num_cond ,max_num_ter, no_coef=True):
    '''
    max_num_cond -- maximal number of conditions in toric datum
    max_num_ter  -- maximal number of terms in a polynomial in a condition
    no_coef      -- If True, omits coefficients from the output

    If no_coef = True, then returns a numpy array of shape (max_num_cond, n*(max_num_ter+1))
    where each condition is represented by a sequence of length n*(max_num_ter+1):
    first n entries are povers of corresponding variables in the left hand side
    the rest is powers of corresponding variables in terms of the right hand side
    concatinated in some order

    If no_coef = False, then returns a numpy array of shape (max_num_cond, n+(n+1)*(max_num_ter)), where coefficients are included
    in the entry before the power series of the term.

    '''
    variables = toric_variables(TD)
    if not variables:
        raise ValueError("Toric datum is empty/ has no variables attached")
    
    n = len(variables)     #number of variables
    m = len(TD.cc)   #number of cone conditions
    s = max([poly.number_of_terms() for poly in TD.rhs]) # max number of terms in a condition
    if m > max_num_cond:
      raise ValueError("Number of conditions is larger than max_num_cond.")


    if s > max_num_ter:
      raise ValueError("Number of terms in a polynomial is larger than max_num_ter.")


    if no_coef:
        nn_data = np.zeros((max_num_cond, n*(max_num_ter+1)), dtype=np.float32)
        for i in range(m):
          poly0 = TD.lhs[i]
          power_seq0 = poly0.exponents()[0]
          poly1 = TD.rhs[i]
          p1exp = poly1.exponents()
          power_seq1 = []
          
          for j in range(len(p1exp)):
              power_seq1.extend(p1exp[j])
              
        

          power_seq = []
          power_seq.extend(power_seq0)
          power_seq.extend(power_seq1)

          for j in range(len(power_seq)):
            nn_data[i,j] = power_seq[j]

        return nn_data
    else:
        nn_data = np.zeros((max_num_cond, n+ (n+1)*(max_num_ter)), dtype=np.float32) # n entries for left-hand side, n+1 for each term of right-hand side (powers of vars + coef)
        for i in range(m):
          poly0 = TD.lhs[i]
          power_seq0 = poly0.exponents()[0]


          poly1 = TD.rhs[i]
          p1exp = poly1.exponents()
          p1coef= poly1.coefficients()
          power_seq1 = []
          
          for j in range(len(p1exp)):
              power_seq1.append(p1coef[j])  #recording the coeficient
              power_seq1.extend(p1exp[j])   #recording the powers
          
          
          

          

          power_seq = []
          power_seq.extend(power_seq0)
          power_seq.extend(power_seq1)

          for j in range(len(power_seq)):
            nn_data[i,j] = power_seq[j]

        return nn_data
    
# %%   Data for NN but in torch format


def get_data_torch(TD, max_num_cond, max_num_ter, no_coef=True):
    '''
    Returns get_data_for_NN in torch format, reshaped in 1 dimension.
    '''
    return (torch.from_numpy(get_data_for_NN(TD, max_num_cond ,max_num_ter, no_coef))).reshape(-1)

# %% weight of ToricDatum already exists. So I transformed this into alternative one of such is needed
# right now works same way as TD.weight()

def td_weight_alt(TD):
    if not TD.cc:
        return 1  # Return 0 if there are no cone conditions

    #max_terms = 0
    #max_degree = 0
    num_terms = 0

    for monomial, poly in TD.cc:
        # Calculate number of terms in the polynomial
        #num_terms = len(poly.as_expr().as_ordered_terms())
        #if num_terms > max_terms:
        #    max_terms = num_terms
        num_terms = num_terms + poly.number_of_terms()-1

        # Find the degree of the polynomial
        #poly_degree = sp.total_degree(poly.as_expr())
        #if poly_degree > max_degree:
        #    max_degree = poly_degree
    #print(max_terms, max_degree)
    #return max_terms * max_degree
    return num_terms


# %% Reduce function deletes terms of rhs that are divisible by lhs

def reduce_td(TD):
    n_cc = copy.deepcopy(TD.cc)
    for index, (g, f) in enumerate(TD.cc):
        #if sp.rem(f,g) == 0:
        #    self.cone_conditions.remove((g, f))
        #else:
        divisible_t = []
        mon_f = f.monomials()
        coe_f = f.coefficients()
        for i in range(len(mon_f)):
            t = mon_f[i]
            _ , re = t.quo_rem(g) 
            #print(re)
            if re == 0:
                divisible_t.append(coe_f[i]*t)
        sum_t = sum(divisible_t)
        #print(sum_t)
        n_cc[index] = (g, (f - sum_t))
    return ToricDatum(TD.ring, TD.integrand, n_cc, TD.initials, TD.polyhedron)
                
# %% Generate ToricDatum with "random" cone conditions

   

def random_td(r,m, s, d, coefficient_ring=GF(32003),coefficient_range=None, integ = None):
    """Return random cone conditions with default integrand.
    r -- free rank of the algebra
    n -- number of variables
    m -- number of conditions
    s -- max number of terms in a condition
    d -- max degree of a term in a condition

    """
    if m < 0:
        raise ValueError("Number of tuples (m) must be non-negative.")
    if r < 0:
        raise ValueError("Rank of module (r) must be non-negative.")

    n = r*(r+1)//2

    if integ:
        integrand = integ
    else:
        #generate default integrand
        positions = [0]  # The first one is at index 0
        current_position = 0
        current_k = r
        while current_position + current_k < n:
          current_position += current_k
          positions.append(current_position)
          current_k -= 1
          if current_k <= 0: # Stop if the step size becomes zero or negative
            break

        integrand = [np.zeros(n, dtype=int),  np.full(n,-1, dtype=int)]
        integrand[0][positions] = 1
        current_k = 0
        for i in positions:
          integrand[1][i] = current_k
          current_k += 1
    
    if coefficient_ring is None:
        ring = PolynomialRing(ZZ, 'x', n)
    else:
        ring = PolynomialRing(coefficient_ring if coefficient_range is None else ZZ, 'x', n)
    ring.inject_variables()
    #define the list of generators of the polynomial ring
    gens = ring.gens()

    cc_list = []
    for _ in range(m):
      # Generate a random monomial
      monomial_degree = random.randint(1, d)
      if coefficient_ring:
          random_monomial_term = random_mon(gens, monomial_degree, coefficient_ring=coefficient_ring)
      else:
          random_monomial_term = random_mon(gens, monomial_degree, coefficient_ring=None, coefficient_range=coefficient_range)

      # Generate a random polynomial
      num_polynomial_terms = random.randint(1, s)
      #print(num_polynomial_terms )
      if coefficient_ring:
          random_polynomial_term = random_poly(gens, d, num_polynomial_terms, coefficient_ring=coefficient_ring)
      else:
          random_polynomial_term = random_poly(gens, d, num_polynomial_terms, coefficient_ring=None, coefficient_range=coefficient_range)

      # Create a tuple of the monomial and polynomial
      pair = (random_monomial_term, random_polynomial_term)

      # Append the tuple to the list
      cc_list.append(pair)

    return ToricDatum(ring, integrand, cc_list, None, PositiveOrthant(n)) 



# %% Examples
"""
r = 5
n = r*(r+1)//2
R = PolynomialRing(QQ, 'x', n)
R.inject_variables()
gens = R.gens()


ccfil4 =[ (x12*x14, x1*x9*x12 + x0*x10*x12 - x0*x9*x13),
(x9*x12, x0*x6*x9 - x0*x5*x10),
(x9*x12*x14, x2*x5*x9*x12 - x1*x6*x9*x12 - x0*x7*x9*x12 + x0*x5*x11*x12 + x0*x6*x9*x13 - x0*x5*x10*x13),
(x9, x0*x5),
(x12, x0*x9),
(x14, x0*x12),
(x14, x5*x9)]


# Examples
integr = random_td(5,6, 16, 6, None).integrand
toyt = ToricDatum(R, integr, ccfil4, None, PositiveOrthant(n)) 

print(toyt)
print(not_free_variables(toyt))
print(free_variables(toyt))
print(td_weight_alt(toyt))



print(get_data_torch(toyt,7,10, False)[0:20])
get_data_torch(toyt,7,10, False)[1]
"""


# %% Blowup  and change of variables on ToyToric
# this not will be replaced by corresponding functions for ToricDatum with _td suffix
# so, commente  out
'''

def trick_blowup_tt(TT, var1, var2):
    """
    Performs a blow-up operation on a ToyToric object between two specified variables.

    This operation generates two new ToyToric objects representing the two charts
    after the blow-up. The cone conditions and the integrand are transformed
    according to the blow-up rules for each chart.

    Args:
        TT (ToyToric): The input ToyToric object on which to perform the blow-up.
        var1 (sympy.core.symbol.Symbol): The first variable involved in the blow-up.
        var2 (sympy.core.symbol.Symbol): The second variable involved in the blow-up.

    Returns:
        list: A list containing two new ToyToric objects resulting from the blow-up.
              The first element corresponds to the chart where var1 is replaced by
              var1 * var2, and the second element corresponds to the chart where
              var2 is replaced by var1 * var2.
    """

    if TT.ring:
      domain = TT.ring
    else:
      domain = sp.ZZ

    indices = [i for i in range(len(TT.variables))]
    ind_dic = dict(zip(TT.variables,indices))
    #ini = Td.initials
    tcc0, tcc1 = trick_blowup_cc(TT.cone_conditions, var1, var2, TT.variables, domain)
    integ1 = copy.deepcopy(TT.integrand)
    integ1[0][ind_dic[var2]] = integ1[0][ind_dic[var2]] + integ1[0][ind_dic[var1]]
    integ1[1][ind_dic[var2]] = integ1[1][ind_dic[var2]] + integ1[1][ind_dic[var1]]

    integ2 = copy.deepcopy(TT.integrand)
    integ2[0][ind_dic[var1]] = integ2[0][ind_dic[var1]] + integ2[0][ind_dic[var2]]
    integ2[1][ind_dic[var1]] = integ2[1][ind_dic[var1]] + integ2[1][ind_dic[var2]]

    
    return [ToyToric( integ1, tcc0, TT.ring, TT.variables), ToyToric( integ2, tcc1, TT.ring, TT.variables)]


def trick_changevar_tt(TT, change_of_vars):
    if TT.ring:
        dom = TT.ring
    else:
        dom = sp.ZZ
    conds=trick_change_var(TT.cone_conditions, change_of_vars, TT.variables, dom)
    return ToyToric(TT.integrand, conds, TT.ring, TT.variables)

#Function that returns TT where all "good" changes of variables are done: reduce weight
# run through free variables x and non-monomial conditions (g,f) where x appear with degree 1 in some term t of f
# find part r of f that is divisible by t/x  and do x -> x - rx/t

def trick_changevar_reduce(TT):
    """
    Applies a beneficial change of variables to a ToyToric object if found.

    Iterates through free variables and cone conditions to find a change of
    variables of the form x -> x - r, where x is a free variable, (g, f) is a
    cone condition, t is a term in f containing x to the power of 1, and r is
    the part of f divisible by t/x. If such a change reduces the weight of the
    ToyToric object, the modified object is returned. Otherwise, the original
    object is returned.

    Args:
        TT (ToyToric): The input ToyToric object.

    Returns:
        ToyToric: The modified ToyToric object if a beneficial change of
                  variables is found, otherwise the original object.
    """
    free_var = TT.free_variables()  # Call the method to get free variables
    #print(free_var)
    for x in free_var:
        for g, f in TT.cone_conditions:
            for t in f.as_expr().as_ordered_terms():
                t_powers = t.as_powers_dict()
                x_degree = t_powers.get(x, 0)
                if x_degree == 1:
                    t_over_x = t / x
                    #print(x, t_over_x)
                    # Ensure t_over_x is not zero to avoid division by zero
                    if t_over_x != 0:
                        # Use sp.div for polynomial division
                        #t_poly = sp.Poly(t, *TT.variables, domain=TT.ring if TT.ring else sp.ZZ)
                        t_over_x_poly = sp.Poly(t_over_x, *TT.variables, domain=TT.ring if TT.ring else sp.ZZ)
                        f_min_t_terms = [term for term in f.as_expr().as_ordered_terms() if term != t]
                        #print(f_min_t_terms)
                        quotient_terms = []
                        for tr in f_min_t_terms:
                            tr_poly = sp.Poly(tr, *TT.variables, domain=TT.ring if TT.ring else sp.ZZ)
                            quo, rem = sp.div(tr_poly, t_over_x_poly)
                            if rem == 0:
                                quotient_terms.append(quo)
                        quotient = sum(quotient_terms)
                        #print(quotient)
                        # Check if the quotient is not a constant
                        if sp.total_degree(quotient) > 0:
                            #print(quotient)
                            #q_powers = quotient.as_expr().as_powers_dict()
                            #x_degree_quotient = q_powers.get(x, 0)
                            if x not in quotient.free_symbols:
                              # Construct the change of variable
                              #print(quotient)
                              #print(quotient.free_symbols)
                              change = {x: x - quotient.as_expr()}
                              # Apply the change of variable
                              new_TT = trick_changevar_tt(TT, change)
                              # Check if the weight is reduced
                              if new_TT.weight() < TT.weight():
                                  return new_TT, True, change  # Return the improved state

    return TT, False, None # Return the original state if no beneficial change is found
'''
# %% Examples

#print(toyt)
#print(toyt.free_variables())


#btoyt = trick_blowup_tt(toyt, gens[0], gens[12])
#print(btoyt)
#print(btoyt[0])
#print(btoyt[1])


#print(toyt.cone_conditions)
#print(btoyt[0].cone_conditions)
#print(btoyt[0])
#print(btoyt[1])

#chtoyt = trick_changevar_tt(toyt, {gens[0]:gens[0]+gens[1]})
#print(chtoyt)



#nTT, is_new, ch = trick_changevar_reduce(btoyt[0])
#print(is_new)
#print(ch)
#print(nTT)

#nTT.reduce()
#print(nTT)

#print(nTT.cone_conditions[0][0])
#print(nTT.cone_conditions[0][1])
#g = nTT.cone_conditions[0][0]
#f = nTT.cone_conditions[0][1]


#nTT.reduce()
#print(nTT)

# %% State of the game and moves
"""
State of the game is a list of ToricDatum instances. On a turn we can do:
1. `blowup()` a blow-up on one of the instances (this addas and extra instance to the state)
2. `change_var()`  a  change of basis in one of the instances

These are implemented as methods.

Also `get_data_for_NN` returns `get_data_for_NN` for each toric in the list concatinated and `get_data_torch` returns it flattened into onedimnsional torch tensor.

## Method code_to_move1

While blowups and chages of variables on a ToyToric are coded very generally (we **can** perform any change of variables), we will only use some special changes in our game. In particular, the move should not introduce ToyTorics into the state that have cone conditions with too many terms or of a too large degree.

We also need to incode moves to np arrays/torch arrays and a way to get them back from such an array.

For changes of variables, I want to implement two cases   
  1. when coefficients are in $\mathbb{Z}$, $x_i \mapsto x_i' \pm P$ where $P$ is a monomial not containing $x_i$ as a factor. This is useful if we know that all coefficients in cone conditions are small integers.     
  2. when coefficients are in $\mathbb{F}_{32003},$ $x_i \mapsto x_i' \pm \alpha P$ where $P$ is a monomial not containing $x_i$ as a factor and $\alpha \in \mathbb{F}_{32003}^*$. This allows much more freedom (more moves) but more costly.

In both cases $x_i$ must be a free variable.


Let $n$ be ne munber of variables and let $d$ be the maximal degree of $P$ above we permit. In first case we have $$1 + \binom{n}{2} + 2n \left( \sum_{i=1}^d \binom{i+n-1}{n-1} \right)$$ moves. Here the first move is to do nothing, then $\binom{n}{2}$ possible blowups and $2n \left( \sum_{i=1}^d \binom{i+n-1}{n-1} \right)$ possible changes of variables: $n$ variables to change, 2 for the sign and $\left( \sum_{i=1}^d \binom{i+n-1}{n-1} \right)$ possible monomials.

Hence in the second case it becomes
$$1 + \binom{n}{2} + (32002)n \left( \sum_{i=1}^d \binom{i+n-1}{n-1} \right).$$

Note that

$$\sum_{i=1}^d \binom{i+n-1}{n-1} = \binom{n+d}{d} -1.$$

Each of these moves can be performed on one of N parts of the state. So another multiplication by N. In total there are
$$1 + N \left(\binom{n}{2} + C  n \left( \binom{n+d}{d} -1 \right)\right)$$
moves where $C$ is the number of possible coefficients.

`code_to_move` takes an integer between 0 and the number abover and performs the move indexed by this number. See the code for detales.

 This is A LOT of moves that give us a very large torch tensor, so in case this does not work, there is another way to code the moves.

## code_to_move2

Second option of incoded move consists of:

1. First three entries: 3 "buckets" for "do nothing", "do a blowup", "do a change of variables". There will be a seperate model for this classification problem.
2. [0,1,0] is followed by $\binom{n}{2}$ "buckets" to choose a pair for a blowup.
3. [0,0,1] is followed by $n$ "buckets" (chose the x_i for change of var) and $2d$ "buckets" ($d$ for + and $d$ for -) where $i$-th bucket means that $\deg P =i$, and then $d$ coordinates that are normalized in the way that all entries are non-negative integers summing to $i$.

Firs model decised type of a move
Second decides a blowup
Third decides  a change of variables  
"""

class ToyState:
    def __init__(self, toy_toric, max_length=15): #here toy_toric is an instance of ToricDatum
        """
        Initializes a ToyState object.

        Args:
            torics (list): A list of ToyToric objects.
            length (int): length of the list .torics
        """
        self.torics = [toy_toric]
        self.length = len(self.torics)
        #self.ring = ring
        self.max_length = max_length
        self.variables =  toric_variables(toy_toric)

    def __str__(self):
        """
        Provides a user-friendly string representation of the ToyState object.
        """
        #return (f"ToyState:\n"
        #        f"  Length: {self.length}\n"
        #        f"  Torics: {self.torics}\n")

         # Create a list of string representations for each ToyToric object
        toric_strings = []
        for toric in self.torics:
            # Get the string representation and clean it up before formatting in the f-string
            ts_cleaned = str(toric).replace('ToricDatum Object:\n', '').replace('  ', '      ')
            toric_strings.append(f"    - {ts_cleaned}")

        # Join the strings with a newline
        torics_representation = "\n".join(toric_strings)

        return (f"ToyState:\n"
                f"  Length: {self.length}\n"
                f"  Torics:\n{torics_representation}\n")


    def __repr__(self):
        """
        Provides an unambiguous string representation of the ToyState object.
        """
        return f"ToyState(length={self.length}, torics={self.torics}"

    def add_toric(self, toy_toric):
        self.torics.append(toy_toric)
        self.length += 1

    def get_data_for_NN(self, max_num_cond ,max_num_ter, no_coef=True):
        n = self.torics[0].ambient_dim # number of variables
        if no_coef:
            nn_data = np.zeros((self.max_length, max_num_cond, n*(max_num_ter+1)), dtype=np.float32)
            for i in range(self.length):
              nn_data[i] = get_data_for_NN(self.torics[i], max_num_cond, max_num_ter)
            return nn_data
        else:
            nn_data = np.zeros((self.max_length, max_num_cond, n+(n+1)*(max_num_ter)), dtype=np.float32)
            for i in range(self.length):
              nn_data[i] = get_data_for_NN(self.torics[i],max_num_cond, max_num_ter, no_coef)
            return nn_data

    def get_data_torch(self, max_num_cond ,max_num_ter, no_coef=True):
        return ((torch.from_numpy(self.get_data_for_NN(max_num_cond ,max_num_ter, no_coef))).reshape(-1)).unsqueeze(0)

    def blowup(self, index, var1, var2):
        self.torics = self.torics[:index] + trick_blowup_td(self.torics[index], var1, var2) + self.torics[index+1:]
        self.length += 1

    def change_var(self, index, change_of_vars):
        self.torics = self.torics[:index] + [trick_changevar_td(self.torics[index], change_of_vars)] + self.torics[index+1:]

    def weight(self):
        #return max([toric.weight() for toric in self.torics])
        return sum([toric.weight() for toric in self.torics])/self.length


    def auto_change_var(self):
        for i in range(self.length):
          can_do_more = True
          while can_do_more:
            nTT, can_do_more, _ = trick_autochenge_td(self.torics[i])
            if can_do_more:
              self.torics[i] = nTT
          self.torics[i] = reduce_td(self.torics[i])


  
    
    
    # now is the part where state reacts on numpay array with encoded move
    def code_to_move1(self, code, N, deg, max_terms, max_deg, coef_pm=True):
        # expects code to be 1-dimensional numpay array (with all but 1 zeroes, ideally. Still will work if its a probability destribution)
        # here N is the maximal number of ToyTorics in the state, deg is maximal degree of change of variables
        # max_deg is the maximum degree allowed in cone conditions
        # max_terms is maximal number of terms in a polynomial in a cone condition
        # coef_pm =True then only +1 and -1 are used as coefficients in the change of variables
        # returns True if move is valid, False otherwise
        gens = self.variables
        n = len(gens)
        K = math.comb(n,2)
        max_index = np.argmax(code)
        if coef_pm:
          num_coef=2
        else:
          num_coef= self.torics[0].ring.base_ring().order()-1  #excluding zero as a coeficient

        if max_index > 1+N*(K+n*num_coef*(math.comb(n+deg,deg)-1)):
          raise ValueError("Input is longer than expected.")
        elif max_index == 1:
          return True
        elif 1 <= max_index  <= N*K:
          if self.length == self.max_length:
            return False #do nothing if reached the limit of ammount of ToyTorics in a state
          cc_ind = (max_index-1) //K

          if cc_ind >= self.length:  # so if code tels us to do something with an empty toric data, we do nothing
            return False
          else:  # code tells us to do something with non-empty toric data
            i= (max_index-1)%K
            a = 0
            # Find the smallest a such that i < comb(n - a, 2)
            while i >= n - a - 1:
                i -= n - a - 1
                a += 1
            b = a + 1 + i

            #Check if degree of polynomials ofter blowup are higher than deg
            deg_exeed = 0
            term_exeed = 0
            for monomial, poly in trick_blowup_td(self.torics[cc_ind], gens[a], gens[b])[0].cc:
              if monomial.total_degree() > max_deg or poly.total_degree() > max_deg:
                deg_exeed = 1
                break
              if (poly.number_of_terms()) > max_terms:
                term_exeed = 1
                break
            for monomial, poly in trick_blowup_td(self.torics[cc_ind], gens[a], gens[b])[1].cc:
              if monomial.total_degree() > max_deg or poly.total_degree() > max_deg:
                deg_exeed = 1
                break
              if (poly.number_of_terms()) > max_terms:
                term_exeed = 1
                break
            if deg_exeed == 0 and term_exeed == 0:
              self.blowup(cc_ind, self.variables[a], self.variables[b])
              self.auto_change_var()
              #return (self.torics[cc_ind].variables[a], self.torics[cc_ind].variables[b])
              return True
            else:
              return False
          
            #DONE UNTILL HERE] free_variables_td


        elif max_index  > N*K:
          ind = max_index-N*K-1
          cc_ind = ind // (n*num_coef*(math.comb(n+deg,deg)-1)) #index of the cone condition in the list
          #print("cc_ind:", cc_ind)

          if cc_ind >= self.length:  # so if code tels us to do something with an empty toric data, we do nothing
            return False
          else:  # code tells us to do something with non-empty toric data
            i= ind % (n*num_coef*(math.comb(n+deg,deg)-1))
            #print("i:", i)
            x_ind = i // (num_coef*(math.comb(n+deg,deg)-1)) # index of the variable we choose for the change
            if gens[x_ind] in free_variables_td(self.torics[cc_ind]): #chack that the variable we do the change of is free
              #print("Variable", gens[x_ind], "is  free!!!")
              i =  i % (num_coef*(math.comb(n+deg,deg)-1))
              #print("i:", i)
              coef_ind =  i // (math.comb(n+deg,deg)-1) # index of a coeficient of the monomial
              #print("coef_ind:", coef_ind)
              if coef_pm:
                coef= (-1)**coef_ind
              else:
                coef=(coef_ind+1)
              #print("coef:", coef)
              i = i % (math.comb(n+deg,deg)-1)
              #print("i:", i)
              mon = get_monomial_from_index(i, n, deg, gens)
              #print("mon:", mon)
              if mon.degree(gens[x_ind]) > 0: # if monomial contains x_ind variable, then such change is not valid
                #print("x_i in P")
                return False
              change = {gens[x_ind]:gens[x_ind]+ coef*mon}
              #print("cnahge:", change)
              #Check if degree of polynomials ofter blowup are higher than deg
              deg_exeed = 0
              term_exeed = 0
              for monomial, poly in trick_changevar_td(self.torics[cc_ind], change).cc:
                if monomial.total_degree() > max_deg or poly.total_degree() > max_deg:
                  deg_exeed = 1
                  break
                if poly.number_of_terms() > max_terms:
                  term_exeed = 1
                  break
              if deg_exeed == 0 and term_exeed == 0:
                self.change_var(cc_ind, change)
                return True
              else:
                return False # to many terms or degree is too large
            else:
              return False # x_ind is not free
        else:
          return False # just in case, should never happen
      
# %% Move1 uncode

def move1_uncode(gens, code, N, deg, max_terms, max_deg, coef_pm=True):
        # expects code to be 1-dimensional numpay array (with all but 1 zeroes, ideally. Still will work if its a probability destribution)
        # here N is the maximal number of ToyTorics in the state, deg is maximal degree of change of variables
        # max_deg is the maximum degree allowed in cone conditions
        # max_terms is maximal number of terms in a polynomial in a cone condition
        # coef_pm =True then only +1 and -1 are used as coefficients in the change of variables
        #gens = self.variables
        n = len(gens)
        K = math.comb(n,2)
        max_index = np.argmax(code)
        if coef_pm:
          num_coef=2
        else:
          num_coef= self.torics[0].ring.base_ring().order()-1 #excluding zero as a coeficient

        if max_index > 1+N*(K+n*num_coef*(math.comb(n+deg,deg)-1)):
          raise ValueError("Input is longer than expected.")
        elif 1 <= max_index  <= N*K:
          cc_ind = (max_index-1) //K
          print("cc_ind:", cc_ind)


          i= (max_index-1)%K
          a = 0
          # Find the smallest a such that i < comb(n - a, 2)
          while i >= n - a - 1:
              i -= n - a - 1
              a += 1
          b = a + 1 + i
          print("blowup:", gens[a], "-", gens[b])
            #self.blowup(cc_ind, self.torics[cc_ind].variables[a], self.torics[cc_ind].variables[b])
            #return (self.torics[cc_ind].variables[a], self.torics[cc_ind].variables[b])

        elif max_index  > N*K:
          ind = max_index-N*K-1
          cc_ind = ind // (n*num_coef*(math.comb(n+deg,deg)-1)) #index of the cone condition in the list
          print("cc_ind:", cc_ind)


          i= ind % (n*num_coef*(math.comb(n+deg,deg)-1))
          #print("i:", i)
          x_ind = i // (num_coef*(math.comb(n+deg,deg)-1)) # index of the variable we choose for the change
          i =  i % (num_coef*(math.comb(n+deg,deg)-1))
          #print("i:", i)
          coef_ind =  i // (math.comb(n+deg,deg)-1) # index of a coeficient of the monomial
          #print("coef_ind:", coef_ind)
          if coef_pm:
            coef= (-1)**coef_ind
          else:
            coef=(coef_ind+1)
          #print("coef:", coef)
          i = i % (math.comb(n+deg,deg)-1)
          #print("i:", i)
          mon = get_monomial_from_index(i, n, deg, gens)
          #print("mon:", mon)
          change = {gens[x_ind]:gens[x_ind]+ coef*mon}
          print("cnahge:", change)

# %%Example of returning data
"""
N = 1
d = 2
max_degree = 6
r = 5
n = r*(r+1)//2
num_coef=2
max_terms = 16
S = ToyState(toyt)
out= S.get_data_torch(7,10, False)
out.shape
S.auto_change_var()
S.torics[0].cc
S.auto_change_var()
print(S)
print(S.weight())
"""
# %% Usage of moves: examples



"""
#toyt = random_td(5,6, 6, 4, None)
#S.blowup(0, gens[0], gens[1])
#print(S)
#print(S.get_data_for_NN(6,6, False))
N = 1
d = 2
max_degree = 6
r = 5
n = r*(r+1)//2
num_coef=2
max_terms = 16
S = ToyState(toyt)
print(S)


code_move_ln =1+N*(math.comb(n,2)+n*num_coef*(math.comb(n+d,d)-1))



code_move = np.zeros(code_move_ln, dtype=np.int32)
#code_move[1780]=1
code_move[1750]=1
S.code_to_move1(code_move, N, d, max_terms, max_degree, True)

move1_uncode(gens, code_move, N, d, max_terms, max_degree, coef_pm=True)


print(S)
print(N*math.comb(n,2))
print(code_move_ln)
print(num_coef*(math.comb(n+d,d)-1))
S.weight()
len(S.torics[0].cc)

#S.get_data_torch(6,6, False)
"""

# %% Example

"""
r = 6
n = r*(r+1)//2

gens = list(sp.symbols('x0:{}'.format(n)))

# Define individual symbols for direct use in expressions
for sym in gens:
    globals()[sym.name] = sym

cc614symbols = [(x18*x20, x9*x18 - x7*x19),
 (x15*x18, x12*x15 - x11*x16),
 (x11*x15, x7*x11 - x6*x12),
 (x11*x15, x2*x11 - x1*x12),
 (x15*x18*x20, x11*x17*x18 + x12*x15*x19 - x11*x16*x19),
 (x15*x18*x20, x3*x15*x18 - x0*x17*x18 + x0*x16*x19),
 (x11*x15*x18, x8*x11*x15 - x6*x13*x15 - x7*x11*x16 + x6*x12*x16),
 (x11*x15*x18, x3*x11*x15 - x1*x13*x15 - x2*x11*x16 + x1*x12*x16),
 (x11*x15*x18, x2*x11*x15 + x0*x13*x15 - x0*x12*x16),
 (x11*x15*x18*x20,
  x6*x14*x15*x18 + x7*x11*x17*x18 - x6*x12*x17*x18 + x8*x11*x15*x19 - x6*x13*x15*x19 - x7*x11*x16*x19 + x6*x12*x16*x19),
 (x11*x15*x18*x20,
  x1*x14*x15*x18 + x2*x11*x17*x18 - x1*x12*x17*x18 + x3*x11*x15*x19 - x1*x13*x15*x19 - x2*x11*x16*x19 + x1*x12*x16*x19),
 (x11*x15*x18*x20,
  x4*x11*x15*x18 + x0*x14*x15*x18 - x0*x12*x17*x18 - x2*x11*x15*x19 - x0*x13*x15*x19 + x0*x12*x16*x19),
 (x20, x18),
 (x20, x16),
 (x18, x15),
 (x18*x20, x15*x19),
 (x20, x13),
 (x20, x12),
 (x15, x11),
 (x20, x8),
 (x18, x7),
 (x11, x6),
 (x20, x2),
 (x11, x1),
 (x11, x0),
 (x11*x15, x0*x12),
 (x15*x18, x0*x16)]

cc614 = [(sp.Poly(p1, *gens, domain =sp.ZZ), sp.Poly(p2, *gens, domain=sp.ZZ) ) for p1, p2 in cc614symbols]

integr = random_td(6,6, 6, 4, None).integrand # this will create a standard integrand for the initial state
toyt614 = ToyToric(integr, cc614, None, gens)
"""
