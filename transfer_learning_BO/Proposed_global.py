import random
import random
import sys
from copy import deepcopy
import pickle
from scipy.optimize import minimize
import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from transfer_learning_BO.gp import train_gp
from transfer_learning_BO.utils import from_unit_cube, latin_hypercube, find_location_source, learn_initial_relatedness, draw_progress, draw_progress_small, EI


class Proposed_global:
    """The Proposed One Box algorithm.

    Parameters
    ----------
    f : function handle
    n_init : Number of initial points (2*dim is recommended), int.
    source_data: training dataset (example in dataset directory)
    num_srrc_hpo_trial: number of training data points (must be less than the total of training data points), int.
    max_evals : Total evaluation budget, int.
    beta: the expansion rate, int.
    length_init: the initial region proportion to the restricted domain, float.
    batch_size : Number of points in each batch, int.
    draw_progress: Draw the result, bool.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")
    initial_file: link to initializations.


    """

    def __init__(
        self,
        f,
        source_data,
        num_src_hpo_trial,
        max_evals = 50,
        beta = -1,
        length_init = 0.2,
        n_init = 3,
        batch_size=1,
        draw_progress = False,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        algo = "bo",
        acq_func = "ei", 
        device="cpu", 
        dtype="float64",
        initial_file = None
    ):

        # Very basic input checks
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.length_init = length_init
        self.initial_file = initial_file

        #Learn the location of training data
        self.algo = algo
        self.acquisition_function = acq_func
        self.source_data = source_data
        self.num_src_hpo_trial = num_src_hpo_trial   # Number of sample to train the source surogate function
        self.minimum_source_location, self.maximum_source_location = find_location_source(self.source_data)
        self.lb = self.minimum_source_location.min(axis=0)
        self.ub = self.minimum_source_location.max(axis=0)
        self.dim = len(self.lb)
        self.alpha_list = list()

        # Create initial bounding box
        while True:
            self.initial_lower = np.random.uniform(self.lb, self.ub)
            self.initial_upper = self.initial_lower + self.length_init * (self.ub - self.lb)
            if np.all(self.initial_upper <= self.ub):
                break

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps
        self.draw_progress = draw_progress
        self.alpha = 0
        self.beta = beta   # Expansion rate
        print("The expand rate is: ", self.beta)

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        self.n_cand = min(500 * self.dim, 5000)
        self.n_evals = 0

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Variables for drawing process
        self.center_list = list()
        self.length_list = list()
        self.weight_dimension_arrow_list = list()
        self.weight_arr_list = list()
        self.best_observed_so_far_list = list()
        self.weight_size_box_list = list()
        self.distance2optimum = list()
        self.volume_ratio = list()

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init * (self.ub - self.lb)
        #self.length_list.append(self.length)

    def _optimize_acq_(self, min):   
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        lb = self.center - self.length / 2.0
        ub = self.center + self.length / 2.0

        self.volume_ratio.append(self.length.prod()/self.initial_vol)

        self.distance2optimum.append(np.linalg.norm(self.f.minimum_point - np.clip(self.f.minimum_point, lb[0], ub[0])))

        # We may have to move the GP to a new device
        self.gp = self.gp.to(dtype=self.dtype, device=self.device)

        if self.acquisition_function == "ei":
            ei = EI(model = self.gp,
                 eta = min,
                 par=0.0)
            bnds = np.vstack((lb,ub)).T
            initial_point = np.random.uniform(lb,ub)
            res = minimize(ei, initial_point, bounds = bnds)
            x_cand = res.x
        else:
            raise NotImplementedError(f"{self.acquisition_function} is not yet implemented for optimizing with scipy. Use grid search instead.")

        return x_cand[None,:]
    
    def _create_candidates(self, min): 
        lb = self.center - self.length / 2.0
        ub = self.center + self.length / 2.0

        self.volume_ratio.append(self.length.prod()/self.initial_vol)

        self.distance2optimum.append(np.linalg.norm(self.f.minimum_point - np.clip(self.f.minimum_point, lb[0], ub[0])))
        
        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=self.dtype, device=self.device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert
        X_cand = deepcopy(pert)
        
        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        self.gp = self.gp.to(dtype=dtype, device=device)

         # We use Lanczos for sampling if we have enough data
        X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
        if self.acquisition_function == 'ts':
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                y_cand = self.gp.likelihood(self.gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()
        elif self.acquisition_function == "ei":
            ei = EI(model = self.gp,
                 eta = min,
                 par=0.0)
            y_cand_ei = ei(X_cand_torch)
            y_cand = np.tile(y_cand_ei,(1,self.batch_size))
        elif self.acquisition_function == "ucb":
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                y_cand_mean = self.gp.likelihood(self.gp(X_cand_torch)).mean
                y_cand_var = torch.sqrt(self.gp.likelihood(self.gp(X_cand_torch)).variance)
                y_cand = -(y_cand_mean + 0.1 * y_cand_var).reshape(-1,self.batch_size).cpu().detach().numpy()
        

        # Remove the torch variables
        del X_cand_torch


        return X_cand, y_cand

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        if self.algo == "bo":
            for i in range(self.batch_size):
                # Pick the best point and make sure we never pick it again
                indbest = np.argmin(y_cand[:, i])
                X_next[i, :] = deepcopy(X_cand[indbest, :])
                y_cand[indbest, :] = np.inf
            return X_next
        elif self.algo == "rs":
            m = len(y_cand)
            arr = range(m)
            mysamp = random.sample(arr,self.batch_size)
            X_next = deepcopy(X_cand[mysamp, :])
            return X_next

    
    def misrank_score(self, y1 , y2):
        num_sample = len(y1)
        num_sample = num_sample - 1
        score = 0
        for i in range(num_sample):
            for k in range(i+1, num_sample+1):
                score += bool(~((y1[i] >= y1[k]) ^ (y2[i] >= y2[k])))
        return 2* score / (num_sample * (num_sample + 1))

    def _moving_center(self, thres_var, thres_size):
        num_model = len(self.model_arr)
        for j, model in enumerate(self.model_arr):
            pred_source = self.gp(torch.tensor(self.source_data["X"][j]))
            var_source = pred_source.variance
            certain_point = var_source < thres_var
            if certain_point.sum() < thres_size:
                add = 0
            else:
                y1 = pred_source.mean.detach().numpy()[certain_point]
                y2 = self.source_data["Y"][j].squeeze()[certain_point]
                add = self.misrank_score(y1, y2)

            self.relatedness_arr[j] = add
        if max(self.relatedness_arr) == 0:
            self.alpha = 0
            center = self.best_observed_so_far_list[-1][0]
            self.weight_arr_list.append(np.zeros(num_model))
        else:
            arr = self.relatedness_arr / sum(self.relatedness_arr)
            self.alpha = self.relatedness_arr.mean()
            self.weight_arr_list.append(arr)
            center = self.alpha * self.minimum_source_location.T @ arr + (1 - self.alpha) * self.best_observed_so_far_list[-1][0]
        self.alpha_list.append(self.alpha)
        weight_dimension_arrow = center - self.center
        self.center = deepcopy(center[None, :])
        self.center_list.append(deepcopy(self.center))
        return weight_dimension_arrow[0]
    

    def optimize(self):
        """Run the full optimization process."""
        if len(self.fX) > 0 and self.verbose:
            n_evals, fbest = self.n_evals, self.fX.min()
            print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
            sys.stdout.flush()

        # Initialize parameters
        
        self.length = self.length_init * (self.ub - self.lb)
        self.initial_vol = self.length.prod()
        self.volume_ratio.append(1)
        for i in range(self.n_init):
            self.length_list.append(deepcopy(self.length))
            self.weight_dimension_arrow_list.append(np.zeros(self.dim))

        # Generate and evalute initial design points
        if self.initial_file is None:
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.initial_lower, self.initial_upper)
        else:
            X_init_ = []
            with open(self.initial_file, 'rb') as f:
                datas = pickle.load(f)
            for data in datas:
                X_init_.append(list(data.values()))
            X_init_ = np.array(X_init_)
            _,m = X_init_.shape
            if m>9:
                drop_col = deepcopy(X_init_[:,2:m-8])
                data = np.delete(X_init_, range(2,m-8), 1)
                X_init = np.hstack((data, drop_col))
            else:
                X_init = X_init_

        fX_init = np.array([[self.f(x)] for x in X_init])
        # Set up the initial center
        for i in range(self.n_init):
            self.center = X_init[fX_init[:i+1].argmin().item(), :][None, :]
            self.center_list.append(self.center)
            self.best_observed_so_far_list.append(self.center)

        # Update budget and set as initial data for this TR
        self.n_evals += self.n_init
        self.X = np.vstack((self.X, deepcopy(X_init)))
        self.fX = np.vstack((self.fX, deepcopy(fX_init)))

        # Create the initial relatedness after each restart
        self.relatedness_arr, self.model_arr, self.mean_arr = learn_initial_relatedness(self.source_data, torch.tensor(self.X), torch.tensor(self.fX), self.num_src_hpo_trial)
        for i in range(self.n_init):
            self.weight_arr_list.append(self.relatedness_arr)

        if self.verbose:
            fbest = self.fX.min()
            print(f"Starting from fbest = {fbest:.4}")
            sys.stdout.flush()

        while self.n_evals < self.max_evals:
            # Warp inputs
            X = deepcopy(self.X)
            fX = deepcopy(self.fX).ravel()

            # Standardize function values.
            mu, sigma = np.median(fX), fX.std()
            sigma = 1.0 if sigma < 1e-6 else sigma
            fX = (deepcopy(fX) - mu) / sigma

            # Figure out what device we are running on
            if len(X) < self.min_cuda:
                device, dtype = torch.device("cpu"), torch.float64
            else:
                device, dtype = self.device, self.dtype

            # We use CG + Lanczos for training if we have enough data
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                X_torch = torch.tensor(X).to(device=device, dtype=dtype)
                y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
                self.gp = train_gp(
                    train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=self.n_training_steps, hypers={}
                )

            min_ = torch.min(y_torch).item()
            # Create th next batch
            self.X_next = self._optimize_acq_(min_)     # Optimize with scipy
            #X_cand, y_cand = self._create_candidates(min_)    # Optimize with grid
            #self.X_next = self._select_candidates(X_cand, y_cand)

            # Evaluate batch
            fX_next = np.array([[self.f(x)] for x in self.X_next])


            if self.verbose and fX_next.min() < self.fX.min():
                n_evals, fbest = self.n_evals, fX_next.min()
                print(f"{n_evals}) New best: {fbest:.4}")
                sys.stdout.flush()
            
            # Update budget and append data
            self.n_evals += self.batch_size
            self.X = np.vstack((self.X, deepcopy(self.X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))

            # Update best observed so far list
            self.best_observed_so_far_list.append(self.X[self.fX.argmin().item(), :][None, :])

            # Moving center trust region
            thres_var = self.gp.covar_module.outputscale.detach().item() * 0.5
            weight_dimension_arrow = self._moving_center(thres_var = thres_var, thres_size = 2 * self.dim)
            self.weight_dimension_arrow_list.append(weight_dimension_arrow)

            # Update trust region
            box_initial = self.ub - self.lb
            size0_box_ind = np.where(box_initial==0)[0]
            box_initial[size0_box_ind] = box_initial[box_initial!=0].min()

            self.length += self.length_init * box_initial * (self.n_evals - self.n_init)**(self.beta)
            length = deepcopy(self.length)
            self.length_list.append(length)

        print("Best evaluation: {}, Log Regret: {}".format(self.fX.min(), np.log(self.fX.min())))
        start_iteration = self.n_init-1
        stop_iteration = len(self.center_list)-1

        if self.draw_progress:
            # draw full steps (Only draw when knowing minimum point of objective function)
            draw_progress(start_iteration = start_iteration, stop_iteretaion = stop_iteration, weight_arr_list = self.weight_arr_list, 
                        weight_dimension_arrow_list = self.weight_dimension_arrow_list, centroid_list = self.center_list, 
                        length_list = self.length_list, minimum_source_location = self.minimum_source_location, 
                        best_candidate_points = self.best_observed_so_far_list, history_points=self.X, lower_bound = self.lb, upper_bound = self.ub,
                        best_point = self.f.minimum_point, local_point = None, offline_data = None)

            # draw few steps (Only draw when knowing minimum point of objective function)
            draw_progress_small(start_iteration = start_iteration, stop_iteretaion = stop_iteration, weight_arr_list = self.weight_arr_list, 
                        weight_dimension_arrow_list = self.weight_dimension_arrow_list, centroid_list = self.center_list, 
                        length_list = self.length_list, minimum_source_location = self.minimum_source_location, 
                        best_candidate_points = self.best_observed_so_far_list, history_points=self.X, lower_bound = self.lb, upper_bound = self.ub,
                        best_point = self.f.minimum_point, local_point = None, offline_data = None)
