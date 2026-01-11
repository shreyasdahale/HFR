import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def scale_spectral_radius(W, target_radius=0.95):
    """
    Scales a matrix W so that its largest eigenvalue magnitude = target_radius.
    """
    eigvals = np.linalg.eigvals(W)
    radius = np.max(np.abs(eigvals))
    if radius == 0:
        return W
    return (W / radius) * target_radius

def augment_state_with_squares(x):
    """
    Given state vector x in R^N, return [ x, x^2, 1 ] in R^(2N+1).
    We'll use this for both training and prediction.
    """
    x_sq = x**2
    return np.concatenate([x, x_sq, [1.0]])  # shape: 2N+1

class MCI3D:
    def __init__(
        self,
        reservoir_size=500,
        cycle_weight=0.9,      # 'l' in the paper
        connect_weight=0.9,    # 'g' in the paper
        input_scale=0.2,
        leaking_rate=1.0,
        ridge_alpha=1e-6,
        combine_factor=0.1,    # 'h' in the paper
        seed=47,
        v1=0.6, v2=0.6         # fixed values for v1, v2
    ):
        """
        reservoir_size: N, size of each cycle reservoir 
        cycle_weight : l, ring adjacency weight in [0,1), ensures cycle synergy
        connect_weight: g, cross-connection weight between the two cycle reservoirs
        input_scale   : scale factor for input->reservoir weights
        leaking_rate  : ESN update alpha 
        ridge_alpha   : readout ridge penalty
        combine_factor: h in [0,1], to form x(t)= h*x1(t)+(1-h)*x2(t) as final combined state
        seed          : random seed
        """
        self.reservoir_size = reservoir_size
        self.cycle_weight   = cycle_weight
        self.connect_weight = connect_weight
        self.input_scale    = input_scale
        self.leaking_rate   = leaking_rate
        self.ridge_alpha    = ridge_alpha
        self.combine_factor = combine_factor
        self.seed           = seed
        self.v1 = v1
        self.v2 = v2
        self._build_mci_esn()

    def _build_mci_esn(self):
        np.random.seed(self.seed)

        N = self.reservoir_size
        W_res = np.zeros((N, N))
        for i in range(N):
            j = (i+1) % N
            W_res[j, i] = self.cycle_weight
        self.W_res = W_res
        W_cn = np.zeros((N, N))
        W_cn[0, N-1] = self.connect_weight
        if N>1:
            W_cn[N-1, 0] = self.connect_weight
        self.W_cn = W_cn
        self.Win1 = None
        self.Win2 = None
        self.x1 = None
        self.x2 = None

        self.W_out = None

    def _init_substates(self):
        N = self.reservoir_size
        self.x1 = np.zeros(N)
        self.x2 = np.zeros(N)

    def reset_state(self):
        if self.x1 is not None:
            self.x1[:] = 0.0
        if self.x2 is not None:
            self.x2[:] = 0.0

    def _update(self, u):
        alpha = self.leaking_rate

        pre1 = self.Win1 @ u + self.W_res @ self.x1 + self.W_cn @ self.x2

        new_x1 = np.cos(pre1)

        pre2 = self.Win2 @ u + self.W_res @ self.x2 + self.W_cn @ self.x1
        new_x2 = np.sin(pre2)

        self.x1 = (1.0 - alpha)*self.x1 + alpha*new_x1
        self.x2 = (1.0 - alpha)*self.x2 + alpha*new_x2

    def _combine_state(self):
        h = self.combine_factor
        return h*self.x1 + (1.0 - h)*self.x2

    def collect_states(self, inputs, discard=100):
        self.reset_state()
        states = []
        for t in range(len(inputs)):
            self._update(inputs[t])
            combined = self._combine_state()
            states.append(combined.copy())
        states = np.array(states)
        return states[discard:], states[:discard]


    def fit_readout(self, train_input, train_target, discard=100):
        T = len(train_input)
        if T<2:
            raise ValueError("Not enough training data")

        d_in = train_input.shape[1]
        if self.Win1 is None or self.Win2 is None:
            np.random.seed(self.seed+100)
            N = self.reservoir_size

            sign_V1 = np.random.choice([-1, 1], size=(N, d_in))
            sign_V2 = np.random.choice([-1, 1], size=(N, d_in))

            v1, v2 = self.v1, self.v2

            V1 = v1 * sign_V1 * self.input_scale
            V2 = v2 * sign_V2 * self.input_scale

            self.Win1 = V1 - V2
            self.Win2 = V1 + V2

        self._init_substates()

        states_use, _ = self.collect_states(train_input, discard=discard)
        target_use = train_target[discard:]

        X_list = []
        for s in states_use:
            X_list.append(augment_state_with_squares(s))
        X_aug = np.array(X_list)

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, target_use)
        self.W_out = reg.coef_

    def predict_autoregressive(self, initial_input, n_steps):
        preds = []
        current_in = np.array(initial_input)

        for _ in range(n_steps):
            self._update(current_in)
            combined = self._combine_state()
            big_x = augment_state_with_squares(combined)
            out = self.W_out @ big_x

            preds.append(out)
            current_in = out

        return np.array(preds)
        
    def predict_open_loop(self, test_input):
        preds = []
        for true_input in test_input:
            self._update(true_input)
            combined = self._combine_state()
            x_aug = augment_state_with_squares(combined)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)
