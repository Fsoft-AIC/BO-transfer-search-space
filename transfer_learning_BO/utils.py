import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gpytorch
from scipy.stats import norm
import torch


def scale_from_unit_square_to_domain(X, domain):
    # X contains elements in unit square, stretch and translate them to lie domain
    return X * domain.ptp(axis=1) + domain[:, 0]

class EI:

    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the incumbent.
    """

    def __init__(self,
                 model,
                 eta,
                 par: float=0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        self.long_name = 'Expected Improvement'
        self.model = model
        self.par = par
        self.eta = eta

    def __call__(self, X, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[None,:]
        X = torch.tensor(X)

        m = self.model.likelihood(self.model(X)).mean.cpu().detach().numpy()
        s = torch.sqrt(self.model.likelihood(self.model(X)).variance).cpu().detach().numpy()
        #m, v = self.model.predict_marginalized_over_instances(X)
        #s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")
        
        return -f.reshape(-1,1)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    #assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X

def find_location_source(data):
    num_dataset = len(data["X"])
    minimum_source_location = []
    maximum_source_location = []
    for i in range(num_dataset):
        X = data["X"][i]
        Y = data["Y"][i][:,0]
        minimum = X[np.array(Y).argmin()]
        maximum = X[np.array(Y).argmax()]
        minimum_source_location.append(minimum)
        maximum_source_location.append(maximum)
    return np.array(minimum_source_location), np.array(maximum_source_location)

def learn_initial_relatedness(data, X_test, y_test, num_src_hpo_trial = 500):
    num_dataset = len(data["X"])
    total_trial = len(data["X"][0])
    related_arr = []
    model_arr = []
    mean_arr = []
    for i in range(num_dataset):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(torch.tensor(data["X"][i][:min(num_src_hpo_trial, total_trial)]), torch.tensor(data["Y"][i][:min(num_src_hpo_trial, total_trial),0]), likelihood)
        model.eval()
        likelihood.eval()
        f_pred = model(X_test)
        f_mean = f_pred.mean
        S = 0
        n = len(y_test)
        for j in range(n):
            for k in range(j+1,n):
                S += (~((f_mean[j]>f_mean[k]) ^ (y_test[j]>y_test[k])))
        S = 2*S/(n*(n-1))
        related_arr.append(S.item())
        model_arr.append(model)
        mean_arr.append(f_mean)
    related_arr = np.array(related_arr)
    return related_arr, model_arr, mean_arr

def draw_progress(start_iteration, stop_iteretaion, weight_arr_list, weight_dimension_arrow_list, centroid_list, length_list,
                   minimum_source_location, best_candidate_points, history_points, lower_bound, upper_bound, best_point, local_point, offline_data):
    first_index = 0
    second_index = 2
    first_dimension = minimum_source_location[:,first_index]
    second_dimension = minimum_source_location[:,second_index]
    center_target = (upper_bound + lower_bound)/2
    print("Length weight_arr_list: ", len(weight_arr_list))
    print("Length weight_dimension_arrow_list: ", len(weight_dimension_arrow_list))
    print("Length of centroid_list: ", len(centroid_list))
    print("Length of length_list: ", len(length_list))
    print("Length best_candidate_points: ", len(best_candidate_points))
    print("Length history_points: ", len(history_points))
    fig = plt.figure(figsize=(60,50))
    for i in range(start_iteration, stop_iteretaion):
        index_max_related = np.argmax(weight_arr_list[i+1])
        dx = weight_dimension_arrow_list[i + 1][first_index]
        dy = weight_dimension_arrow_list[i + 1][second_index]
        ax = fig.add_subplot(int((stop_iteretaion+ 1)/4),4,i+1 - start_iteration)
        plt.scatter(x = centroid_list[i][0][first_index], y = centroid_list[i][0][second_index], color = 'black')
        plt.scatter(first_dimension, second_dimension)
        plt.scatter(first_dimension[index_max_related], second_dimension[index_max_related], color = "red")
        plt.scatter(best_candidate_points[i][0][first_index], best_candidate_points[i][0][second_index], color = "orange")
        plt.scatter(history_points[i+1][first_index], history_points[i+1][second_index], color = "tab:green")
        # for o in range(len(weight_arr_list[i])):
        #     plt.text(first_dimension[o], second_dimension[o], str((round(first_dimension[o],3), round(second_dimension[o],3))))
        if best_point is not None:
            plt.scatter(x = best_point[first_index], y = best_point[second_index], marker= 'x', color = 'red')
            #plt.text(best_point[first_index], best_point[second_index], str((round(best_point[first_index],3), round(best_point[second_index],3))))
        if local_point is not None:
            plt.scatter(x = local_point[first_index], y = local_point[second_index], marker= 'x', color = 'gray')
        
        #plt.title("Iteration " + str(i))
        plt.arrow(centroid_list[i][0][first_index],centroid_list[i][0][second_index],dx,dy,width = 0.003, length_includes_head=True, color = 'red')
        left = centroid_list[i][0][first_index] - length_list[i][first_index]/2
        bottom = centroid_list[i][0][second_index] - length_list[i][second_index]/2
        width = length_list[i][first_index]
        height = length_list[i][second_index]
        rect=mpatches.Rectangle((left,bottom),width,height, 
                                fill=False,
                                color="purple",
                            linewidth=1.5)
                            #facecolor="red")
        plt.gca().add_patch(rect)
        plt.text(1,1, "Iteration {}".format(i), ha='right', va='bottom', transform=ax.transAxes, bbox=dict(boxstyle="square",
                                                                    ec=(1., 0.5, 0.5),
                                                                    fc=(1., 0.8, 0.8),
                                                                    ))
    plt.show()

def draw_progress_small(start_iteration, stop_iteretaion, weight_arr_list, weight_dimension_arrow_list, centroid_list, length_list,
                   minimum_source_location, best_candidate_points, history_points, lower_bound, upper_bound, best_point, local_point, offline_data):
    first_index = 0
    second_index = 2
    first_dimension = minimum_source_location[:,first_index]
    second_dimension = minimum_source_location[:,second_index]
    # lower_target = offline_data["X"].min(axis=0)
    # upper_target = offline_data["X"].max(axis=0)
    # center_target = (lower_target + upper_target)/2
    center_target = (upper_bound + lower_bound)/2
    print("Length weight_arr_list: ", len(weight_arr_list))
    print("Length weight_dimension_arrow_list: ", len(weight_dimension_arrow_list))
    print("Length of centroid_list: ", len(centroid_list))
    print("Length of length_list: ", len(length_list))
    print("Length best_candidate_points: ", len(best_candidate_points))
    print("Length history_points: ", len(history_points))
    fig = plt.figure(figsize=(12,7))
    ind_pic = 1
    for i in range(start_iteration, stop_iteretaion,8):
        index_max_related = np.argmax(weight_arr_list[i+1])
        dx = weight_dimension_arrow_list[i + 1][first_index]
        dy = weight_dimension_arrow_list[i + 1][second_index]
        ax = fig.add_subplot(2,3,ind_pic)
        ind_pic += 1
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)
        #plt.scatter(offline_data["X"][:,0], offline_data["X"][:,1], color = 'magenta', alpha=0.2 )
        plt.scatter(x = centroid_list[i][0][first_index], y = centroid_list[i][0][second_index], color = 'black')
        plt.scatter(first_dimension, second_dimension)
        plt.scatter(first_dimension[index_max_related], second_dimension[index_max_related], color = "red")
        plt.scatter(best_candidate_points[i][0][first_index], best_candidate_points[i][0][second_index], color = "orange")
        plt.scatter(history_points[i+1][first_index], history_points[i+1][second_index], color = "tab:green")
        # for o in range(len(weight_arr_list[i])):
        #     plt.text(first_dimension[o], second_dimension[o], str((round(first_dimension[o],3), round(second_dimension[o],3))))
        if best_point is not None:
            plt.scatter(x = best_point[first_index], y = best_point[second_index], marker= 'x', color = 'red')
            #plt.text(best_point[first_index], best_point[second_index], str((round(best_point[first_index],3), round(best_point[second_index],3))))
        if local_point is not None:
            plt.scatter(x = local_point[first_index], y = local_point[second_index], marker= 'x', color = 'gray')
        
        #plt.title("Iteration " + str(i))
        plt.arrow(centroid_list[i][0][first_index],centroid_list[i][0][second_index],dx,dy,width = 0.003, length_includes_head=True, color = 'red')
        left = centroid_list[i][0][first_index] - length_list[i][first_index]/2
        bottom = centroid_list[i][0][second_index] - length_list[i][second_index]/2
        width = length_list[i][first_index]
        height = length_list[i][second_index]
        rect=mpatches.Rectangle((left,bottom),width,height, 
                                fill=False,
                                color="purple",
                            linewidth=1.5)
                            #facecolor="red")
        plt.gca().add_patch(rect)
        plt.text(1,1, "Iteration {}".format(i), fontsize = 12, ha='right', va='bottom', transform=ax.transAxes, bbox=dict(boxstyle="square",
                                                                    ec=(1., 0.5, 0.5),
                                                                    fc=(1., 0.8, 0.8),
                                                                    ))
    plt.show()

