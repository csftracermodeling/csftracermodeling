import jax.numpy as jnp
from jax import grad

def del_i(g, argnum: int = 0):
    """
    Partial derivative for a function of signature (d,) ---> ().
    Intended to use when defining PINN loss functions.
    
    """
    def g_splitvar(*args):
        x_ = jnp.array(args)
        return g(x_)

    d_splitvar_di = grad(g_splitvar, argnum)

    def dg_di(x):
        return d_splitvar_di(*x)

    return dg_di