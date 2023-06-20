from joblib import Parallel, delayed

def f(x):
    return x*x

if __name__ == '__main__':
    results = Parallel(n_jobs=5)(delayed(f)(i) for i in [1, 2, 3])
    print(results)
