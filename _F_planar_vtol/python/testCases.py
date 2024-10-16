errorTolerance = 0.001
dynamics_test_cases = [
# initial state (z, h, theta, zdot, hdot, th_dot),  input (fl, fr),     true state of plant (z, h, theta, zdot, hdot, th_dot)
    ([[0.], [0.], [0.], [0.], [0.], [0.]],          [[0.], [0.]],       [[0.], [-0.0004905], [0.], [0.], [-0.0981], [0.]]),
    ([[1.], [0.], [0.], [0.], [1.], [0.]],          [[1.], [1.]],       [[1.], [0.00957617], [0.], [0.], [0.91523333], [0.]]),
    ([[0.], [1.], [0.], [0.], [0.], [1.]],          [[-100.], [100.]],  [[0.], [0.9995095], [0.01609756], [0.], [-0.0981], [2.2195122]]),
    ([[0.], [0.], [1.], [0.], [-1.], [-1.]],        [[2.], [5.]],       [[-1.95897167e-04], [-1.03638059e-02], [9.90914634e-01], [-3.91366349e-02], [-1.07270189e+00], [-8.17073171e-01]]),
    ([[1.], [0.], [0.], [1.], [0.], [-10.]],        [[-2.], [-5.]],     [[1.00998886e+00], [-7.23637144e-04], [-1.00914634e-01], [9.96988504e-01], [-1.44687859e-01], [-1.01829268e+01]]),
    ([[-1.], [-10.], [0.], [1.], [50.], [5.]],      [[0.1], [0.5]],     [[-0.99000367], [-9.5004705], [0.05012195], [0.99923344], [49.90589833], [5.02439024]]),
    ([[0.], [5.], [-5.], [-5.], [5.], [-50.]],      [[-1.5], [-0.9]],   [[-4.99119708e-02], [5.04947463e+00], [-5.49981707e+00], [-4.98307140e+00], [4.89379230e+00], [-4.99634146e+01]]),
    ([[1.], [2.], [3.], [4.], [5.], [6.]],          [[3.], [-20.]],     [[1.04001512], [2.04974107], [3.05603659], [4.00258975], [4.94826341], [5.20731707]]),
    ([[-1.], [2.], [-3.], [4.], [-5.], [6.]],       [[5.], [5.]],       [[-0.95995972], [1.94918054], [-2.94], [4.0087124], [-5.16377775], [6.]]),
    ([[1.], [1.], [1.], [1.], [1.], [1.]],          [[-2.], [-2.]],     [[1.01010908], [1.00943783], [1.01], [1.02183695], [0.88760437], [1.]]),
]
