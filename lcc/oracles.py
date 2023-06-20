from lcc import lcc
import numpy as np


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError("Func is not implemented.")

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError("Grad is not implemented.")


class LCCLeastSquaresOracle(BaseSmoothOracle):
    """
    LCC oracle for least-squares regression.
        f(x) = 0.5 ||Xw - y||_2^2
    """

    def __init__(
        self,
        X,
        y,
        prime_number,
        num_workers=40,
        num_stragglers=5,
        security_guarantee=2,
        privacy_guarantee=2,
        precision=16,
        fit_intercept=False,
        verbose=False,
    ):
        self.X = np.hstack((np.ones((len(X), 1)), X)) if fit_intercept else X
        self.y = y
        self.XTy = self.X.T.dot(y)
        self.Xq = lcc.quantize(self.X, precision, prime_number)
        self.yq = lcc.quantize(self.y, precision, prime_number)

        self.num_workers = num_workers  # N
        self.num_stragglers = num_stragglers  # S
        self.security_guarantee = security_guarantee  # A
        self.privacy_guarantee = privacy_guarantee  # T
        self.prime_number = prime_number
        self.precision = precision
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        self._init_num_batches()  # K
        self._precomputation()

    def _init_num_batches(self):
        self.poly_degree = 2  # X.T.dot(X)
        self.num_batches = (
            (self.num_workers - 1 - 2 * self.security_guarantee - self.num_stragglers)
            // self.poly_degree
            + 1
            - self.privacy_guarantee
        )
        m, _ = self.X.shape
        while m % self.num_batches:
            self.num_batches -= 1
        self.n_bound = (
            self.num_batches + self.privacy_guarantee - 1
        ) * self.poly_degree + 1
        if self.verbose:
            print(f"Initialized K={self.num_batches} batches")
            print(f"Number of succedeed to restore correctly: N>={self.n_bound}")

    def _precomputation(self):
        n_betas = self.num_batches + self.privacy_guarantee
        n_alphas = self.num_workers
        self.alphas = (np.arange(2 * n_alphas, step=2)).astype("int64")
        self.betas = (np.arange(1, 2 * n_betas, step=2)).astype("int64")
        self.U = lcc.gen_Lagrange_coeffs(self.alphas, self.betas, self.prime_number)
        Xq_split = np.stack(np.split(self.Xq, self.num_batches))
        self.Z = np.random.randint(
            0, self.prime_number, (self.privacy_guarantee, *Xq_split.shape[1:])
        ).astype("int64")
        X_to_encode = np.concatenate([Xq_split, self.Z])
        self.X_encoded = np.mod(
            np.einsum("kn,kmd->nmd", self.U, X_to_encode), self.prime_number
        )  # N x (m // K) x d

    def _worker_func(self, X, w):
        wq = lcc.quantize(w, self.precision, self.prime_number)
        return np.mod(X.dot(wq), self.prime_number)  # (N - S) x (m // K)

    def func(self, w):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        func_encoded = self._worker_func(self.X_encoded[succeeded_workers], w)
        if self.security_guarantee > 0:
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
                self.prime_number,
            )
            func_decoded = np.mod(
                np.einsum("nk,nm->km", U_dec, func_encoded), self.prime_number
            )
            self.worker_func = lcc.dequantize(
                func_decoded, self.precision, self.prime_number
            )
        return 0.5 * ((self.worker_func.reshape(-1) - self.y) ** 2).mean()

    def _debug_func(self, w):
        return 0.5 * ((self.X.dot(w) - self.y) ** 2).mean()

    def _worker_grad(self, X, w):
        wq = lcc.quantize(w, self.precision, self.prime_number)
        Xw = np.mod(np.einsum("nmd,d->nm", X, wq), self.prime_number)  # (N - S) x d
        return np.mod(np.einsum("nmd,nm->nd", X, Xw), self.prime_number)  # (N - S) x d

    def grad(self, w):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        grad_encoded = self._worker_grad(self.X_encoded[succeeded_workers], w)
        if self.security_guarantee > 0:
            malicious_workers = np.random.choice(
                list(set(range(self.num_workers)) - set(succeeded_workers)),
                self.security_guarantee,
                replace=False,
            )
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
                self.prime_number,
            )
            grad_decoded = np.mod(
                np.einsum("nk,nd->kd", U_dec, grad_encoded), self.prime_number
            )
            worker_grad = lcc.dequantize(
                grad_decoded, self.precision, self.prime_number
            )
            self.worker_grad = worker_grad.sum(0)
        return self.worker_grad - self.XTy

    def _debug_grad(self, w):
        return self.X.T.dot(self.X.dot(w) - self.y)

    def _worker_matmul(self, X):
        return np.mod(np.einsum("nml,nmd->nld", X, X), self.prime_number)

    def matmul(self):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        mm_encoded = self._worker_matmul(self.X_encoded[succeeded_workers])
        if self.security_guarantee > 0:
            malicious_workers = np.random.choice(
                list(set(range(self.num_workers)) - set(succeeded_workers)),
                self.security_guarantee,
                replace=False,
            )
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
                self.prime_number,
            )
            mm_decoded = np.mod(
                np.einsum("nk,nld->ld", U_dec, mm_encoded), self.prime_number
            )
            mm_decoded = lcc.dequantize(mm_decoded, self.precision, self.prime_number)
        return mm_decoded

    def debug_matmul(self):
        return self.X.T.dot(self.X)


class ALCCLeastSquaresOracle(BaseSmoothOracle):
    """
    ALCC oracle for least-squares regression.
        f(x) = 0.5 ||Xw - y||_2^2
    """

    def __init__(
        self,
        X,
        y,
        num_workers=40,
        num_stragglers=5,
        security_guarantee=2,
        privacy_guarantee=2,
        beta=1.1,
        sigma=10**3,
        fit_intercept=False,
        verbose=False,
    ):
        self.X = np.hstack((np.ones((len(X), 1)), X)) if fit_intercept else X
        self.y = y
        self.XTy = self.X.T.dot(y)

        self.num_workers = num_workers  # N
        self.num_stragglers = num_stragglers  # S
        self.security_guarantee = security_guarantee  # A
        self.privacy_guarantee = privacy_guarantee  # T
        self.fit_intercept = fit_intercept

        self.beta = beta
        self.sigma = sigma
        self.verbose = verbose
        self._init_num_batches()  # K designed by N, (S, A, T)
        self._precomputation(self.beta, self.sigma)

    def _init_num_batches(self):
        self.poly_degree = 2  # X.T.dot(X)
        self.num_batches = (
            (self.num_workers - 1 - 2 * self.security_guarantee - self.num_stragglers)
            // self.poly_degree
            + 1
            - self.privacy_guarantee
        )
        m, _ = self.X.shape
        while m % self.num_batches:
            self.num_batches -= 1
        self.n_bound = (
            self.num_batches + self.privacy_guarantee - 1
        ) * self.poly_degree + 1
        if self.verbose:
            print(f"Initialized K={self.num_batches} batches")
            print(f"Number of succedeed to restore correctly: N>={self.n_bound}")

    def _precomputation(self, beta=1.1, sigma="adaptive", theta=6):
        n_betas = self.num_batches + self.privacy_guarantee
        n_alphas = self.num_workers
        n_betas = self.num_batches + self.privacy_guarantee
        self.alphas = np.exp(2 * np.pi * 1j * np.arange(n_alphas) / n_alphas)
        self.betas = beta * np.exp(2 * np.pi * 1j * np.arange(n_betas) / n_betas)
        self.U = lcc.gen_Lagrange_coeffs(self.alphas, self.betas)
        X_split = np.stack(np.split(self.X, self.num_batches))
        if self.privacy_guarantee:
            if sigma == "adaptive":
                sigma = self.X.std()
            std = np.sqrt(sigma**2 / self.privacy_guarantee / 2)
            self.Z = (
                (std * np.random.randn(self.privacy_guarantee, *X_split.shape[1:], 2))
                .clip(-theta * std, theta * std)
                .view(np.complex128)
                .squeeze(-1)
            )
            # T x (m // K) x d
            X_to_encode = np.concatenate([X_split, self.Z])
        else:
            X_to_encode = X_split
        self.X_encoded = np.einsum(
            "kn,kmd->nmd", self.U, X_to_encode
        )  # N x (m // K) x d

    def _worker_func(self, X, w):
        return X.dot(w)  # (N - S) x (m // K)

    def func(self, w):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        func_encoded = self._worker_func(self.X_encoded[succeeded_workers], w)
        if self.security_guarantee > 0:
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
            )
            func_decoded = np.einsum("nk,nm->km", U_dec, func_encoded)
            self.worker_func = func_decoded.real
        return np.array(0.5 * ((self.worker_func.reshape(-1) - self.y) ** 2).sum())

    def _debug_func(self, w):
        return np.array(0.5 * ((self.X.dot(w) - self.y) ** 2).sum())

    def _worker_grad(self, X, w):
        Xw = np.einsum("nmd,d->nm", X, w)  # (N - S) x d
        return np.einsum("nmd,nm->nd", X, Xw)  # (N - S) x d

    def grad(self, w):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        grad_encoded = self._worker_grad(self.X_encoded[succeeded_workers], w)
        if self.security_guarantee > 0:
            malicious_workers = np.random.choice(
                list(set(range(self.num_workers)) - set(succeeded_workers)),
                self.security_guarantee,
                replace=False,
            )
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
            )
            grad_decoded = np.einsum("nk,nd->kd", U_dec, grad_encoded)
            self.worker_grad = grad_decoded.sum(0).real
        return self.worker_grad - self.XTy

    def _debug_grad(self, w):
        return self.X.T.dot(self.X.dot(w) - self.y)

    def _worker_matmul(self, X):
        return np.mod(np.einsum("nml,nmd->nld", X, X), self.prime_number)

    def matmul(self):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        mm_encoded = self._worker_matmul(self.X_encoded[succeeded_workers])
        if self.security_guarantee > 0:
            malicious_workers = np.random.choice(
                list(set(range(self.num_workers)) - set(succeeded_workers)),
                self.security_guarantee,
                replace=False,
            )
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
                self.prime_number,
            )
            mm_decoded = np.mod(
                np.einsum("nk,nld->ld", U_dec, mm_encoded), self.prime_number
            )
            mm_decoded = lcc.dequantize(mm_decoded, self.precision, self.prime_number)
        return mm_decoded

    def debug_matmul(self):
        return self.X.T.dot(self.X)


class ALCCLeastSquaresOracle(BaseSmoothOracle):
    """
    ALCC oracle for least-squares regression.
        f(x) = 0.5 ||Xw - y||_2^2
    """

    def __init__(
        self,
        X,
        y,
        num_workers=40,
        num_stragglers=5,
        security_guarantee=2,
        privacy_guarantee=2,
        beta=1.1,
        sigma=10**3,
        fit_intercept=False,
    ):
        self.X = np.hstack((np.ones((len(X), 1)), X)) if fit_intercept else X
        self.y = y
        self.XTy = self.X.T.dot(y)

        self.num_workers = num_workers  # N
        self.num_stragglers = num_stragglers  # S
        self.security_guarantee = security_guarantee  # A
        self.privacy_guarantee = privacy_guarantee  # T
        self.fit_intercept = fit_intercept

        self.beta = beta
        self.sigma = sigma
        self._init_num_batches()  # K designed by N, (S, A, T)
        self._precomputation(self.beta, self.sigma)

    def _init_num_batches(self):
        self.poly_degree = 2  # X.T.dot(X)
        self.num_batches = (
            (self.num_workers - 1 - 2 * self.security_guarantee - self.num_stragglers)
            // self.poly_degree
            + 1
            - self.privacy_guarantee
        )
        m, _ = self.X.shape
        while m % self.num_batches:
            self.num_batches -= 1
        self.n_bound = (
            self.num_batches + self.privacy_guarantee - 1
        ) * self.poly_degree + 1
        print(f"Initialized K={self.num_batches} batches")
        print(f"Numbers of succeded workers to restore correctly: N>={self.n_bound}")

    def _precomputation(self, beta=1.1, sigma="adaptive", theta=6):
        n_betas = self.num_batches + self.privacy_guarantee
        n_alphas = self.num_workers
        n_betas = self.num_batches + self.privacy_guarantee
        self.alphas = np.exp(2 * np.pi * 1j * np.arange(n_alphas) / n_alphas)
        self.betas = beta * np.exp(2 * np.pi * 1j * np.arange(n_betas) / n_betas)
        self.U = lcc.gen_Lagrange_coeffs(self.alphas, self.betas)
        X_split = np.stack(np.split(self.X, self.num_batches))
        if self.privacy_guarantee:
            if sigma == "adaptive":
                sigma = self.X.std()
            std = np.sqrt(sigma**2 / self.privacy_guarantee / 2)
            self.Z = (
                (std * np.random.randn(self.privacy_guarantee, *X_split.shape[1:], 2))
                .clip(-theta * std, theta * std)
                .view(np.complex128)
                .squeeze(-1)
            )
            # T x (m // K) x d
            X_to_encode = np.concatenate([X_split, self.Z])
        else:
            X_to_encode = X_split
        self.X_encoded = np.einsum(
            "kn,kmd->nmd", self.U, X_to_encode
        )  # N x (m // K) x d

    def _worker_func(self, X, w):
        return X.dot(w)  # (N - S) x (m // K)

    def func(self, w):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        func_encoded = self._worker_func(self.X_encoded[succeeded_workers], w)
        if self.security_guarantee > 0:
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
            )
            func_decoded = np.einsum("nk,nm->km", U_dec, func_encoded)
            self.worker_func = func_decoded.real
        return np.array(0.5 * ((self.worker_func.reshape(-1) - self.y) ** 2).sum())

    def _debug_func(self, w):
        return np.array(0.5 * ((self.X.dot(w) - self.y) ** 2).sum())

    def _worker_grad(self, X, w):
        Xw = np.einsum("nmd,d->nm", X, w)  # (N - S) x d
        return np.einsum("nmd,nm->nd", X, Xw)  # (N - S) x d

    def grad(self, w):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        grad_encoded = self._worker_grad(self.X_encoded[succeeded_workers], w)
        if self.security_guarantee > 0:
            malicious_workers = np.random.choice(
                list(set(range(self.num_workers)) - set(succeeded_workers)),
                self.security_guarantee,
                replace=False,
            )
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
            )
            grad_decoded = np.einsum("nk,nd->kd", U_dec, grad_encoded)
            self.worker_grad = grad_decoded.sum(0).real
        return self.worker_grad - self.XTy

    def _debug_grad(self, w):
        return self.X.T.dot(self.X.dot(w) - self.y)

    def _worker_matmul(self, X):
        return np.einsum("nml,nmd->nld", X, X)

    def matmul(self):
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        mm_encoded = self._worker_matmul(self.X_encoded[succeeded_workers])
        if self.security_guarantee > 0:
            malicious_workers = np.random.choice(
                list(set(range(self.num_workers)) - set(succeeded_workers)),
                self.security_guarantee,
                replace=False,
            )
            raise NotImplementedError("Adversarial decoding not yet supported!")
        else:
            U_dec = lcc.gen_Lagrange_coeffs(
                self.betas[: self.num_batches],
                self.alphas[succeeded_workers],
            )
            mm_decoded = np.einsum("nk,nld->ld", U_dec, mm_encoded)
        return mm_decoded

    def debug_matmul(self):
        return self.X.T.dot(self.X)
