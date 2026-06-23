import concurrent.futures, numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from trade_solver import solve_trade_jit, _BUILD_COUNTS

def data(n, seed):
 rng=np.random.default_rng(seed); B=rng.normal(size=(n,n)); S=B.T@B+0.2*np.eye(n); A=np.linalg.cholesky(S).T
 return (jnp.array(rng.normal(size=n)), jnp.array(0.01+rng.random(n)*0.05), jnp.array(rng.normal(size=n)*0.1), jnp.array(A), jnp.array([float(0.5+rng.random())]))

def call(n, seed):
 w, info=solve_trade_jit(*data(n,seed), n_shards=4)
 return np.asarray(w), np.asarray(info)

for n in (3,9,3):
 w,info=call(n,n)
 print('auto_n_call', n, 'w_shape', w.shape, 'info_shape', info.shape, 'status', info[0])
print('build_counts', dict(_BUILD_COUNTS))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 futs=[ex.submit(call, 3, i) for i in range(16)]
 outs=[f.result() for f in futs]
print('concurrency_calls', len(outs), 'all_finite', all(np.isfinite(o[0]).all() and np.isfinite(o[1]).all() for o in outs))
