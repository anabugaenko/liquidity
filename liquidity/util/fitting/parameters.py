class BaseParam(object):
    def get_values(self):
        pass


class BetaLambdaParam(BaseParam):
    beta: float
    lambda_: float

    def __init__(self, beta, lambda_):
        self.beta = beta
        self.lambda_ = lambda_

    def get_values(self):
        return [self.beta, self.lambda_]


class PeriodicFunctionParam(BaseParam):
    a0: float
    A: float
    omega: float

    def __init__(self, a0, A, omega):
        self.a0 = a0
        self.A = A
        self.omega = omega

    def get_values(self):
        return [self.a0, self.A, self.omega]


class PowerlawParam(BaseParam):
    alpha: float
    C: float

    def __init__(self, alpha, C):
        self.alpha = alpha
        self.C = C

    def get_values(self):
        return [self.alpha, self.C]


class PowerlawSVFAlphaBetaLambdaParam(BaseParam):
    alpha: float
    beta: float
    lambda_: float

    def __init__(self, alpha, beta, lambda_):
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

    def get_values(self):
        return [self.alpha, self.beta, self.lambda_]


class PowerlawSVFPeriodicParam(BaseParam):
    alpha: float
    a0: float
    A: float
    omega: float

    def __init__(self, alpha, a0, A, omega):
        self.alpha = alpha
        self.a0 = a0
        self.A = A
        self.omega = omega

    def get_values(self):
        return [self.alpha, self.a0, self.A, self.omega]
