import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from sklearn.datasets import make_moons, make_circles, make_blobs
import matplotlib.pyplot as plt
import numpy as np
#from prettytable import PrettyTable
import copy
#import scipy

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 200
NOISE = 0.05
INPUT_DIM = 2
INTRINSIC_DIM = 2 # intrinsic dimension of input data
HIDDEN_DIM = 32 # width of the neural network
N_LAYERS = 4 # depth of the neural network
LR = 1e-3 # learning rate for the optimizer
BATCH_SIZE = 32  # was 32
N_EPOCHS = 300 # number of times each data points is used for training


class CouplingLayer(nn.Module):
    """
    An implementation of a RealNVP-style coupling layer.
    The input is split in half. One half is used to predict the scale and shift
    parameters for the other half.
    """
    def __init__(self, input_dim, hidden_dim, mask):
        super().__init__()
        self.mask = mask


        self.s_t_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2) 
        )

    def forward(self, x):
        """ The forward pass for training (data -> latent space). """
        # Create a masked version of the input to feed into the network
        x_masked = x * self.mask
        
        # Compute scale and shift parameters from the masked input
        s_t = self.s_t_net(x_masked)
        s, t = s_t.chunk(2, dim=-1)
        
        
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Use tanh to stabilize training and prevent explosion
        s = torch.tanh(s)
 
        # Apply the transformation: y = x * exp(s) + t
        y = x * torch.exp(s) + t
        
        # The log-determinant of the Jacobian is the sum of the log-scaling factors.
        log_det_J = s.sum(dim=-1)

        return y, log_det_J

    def inverse(self, y):
        """ The inverse pass for sampling (latent (base) space -> data). """
        y_masked = y * self.mask
        
        s_t = self.s_t_net(y_masked)
        s, t = s_t.chunk(2, dim=-1)
        
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        s = torch.tanh(s)
        x = (y - t) * torch.exp(-s)
        
        return x
    
    # Placeholder for the implementation of the computations of the Jacobian (suggestion is to use AutoGrad)
    def compute_jacobian(self, x):
        x = x.clone().requires_grad_(True)  # Ensure we retain graph for gradient calculation
        #jacobian = torch.zeros(x.size(0), x.size(1), x.size(1))  # dummy to replace later
        
        # Compute Jacobian per sample
        jacobian = []
        for i in range(x.size(0)):
            x_i = x[i].detach().clone().requires_grad_(True)  # individual input
            f_i = lambda x: self.forward(x)[0]
            J_i = torch.autograd.functional.jacobian(f_i, x_i) # shape: (output_dim, input_dim)
            jacobian.append(J_i)

        # Stack into tensor of shape (batchsize, output_dim, input_dim)
        jacobian = torch.stack(jacobian)

        #print("Jacobian shape:", jacobian.shape)  # torch.Size([batchsize, 2, 2])
        return jacobian

class NF(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Alternating mask for the coupling layer
        for i in range(n_layers):
            mask = torch.zeros(input_dim)
            mask[i % 2::2] = 1 # Alternate wich half is masked
            self.layers.append(CouplingLayer(input_dim, hidden_dim, mask.to(DEVICE)))
            
    def forward(self, x):
        """ Maps data to the base distribution. """
        total_log_det_J = 0
        for layer in self.layers:
            x, log_det_J = layer(x)
            total_log_det_J += log_det_J
        return x, total_log_det_J

    def inverse(self, z):
        """ Maps the base distribution to data. """
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def compute_jacobians(self, x):
        """Computes a list of Jacobian matrices for each layer."""
        jacobians = []
        for layer in self.layers:
            jacobians.append(layer.compute_jacobian(x))
            x, _ = layer(x) 
        return jacobians


base_distribution = MultivariateNormal(
    loc=torch.zeros(INPUT_DIM).to(DEVICE),
    covariance_matrix=torch.eye(INPUT_DIM).to(DEVICE)
)


# Various types of datasets
X, _ = make_moons(n_samples=N_SAMPLES, noise=NOISE)
#X, _ = make_circles(n_samples=N_SAMPLES, noise=NOISE)
#X, _ = make_blobs(n_samples = N_SAMPLES, n_features = INTRINSIC_DIM, centers=3, cluster_std=1.2)


if INPUT_DIM > 2:  # embedding two-dimensional data in higher dimensions
    v1 = np.random.uniform(low=-1, high=1, size=INPUT_DIM)
    v2 = np.random.uniform(low=-1, high=1, size=INPUT_DIM)
    while np.all(v1 == 0):
        v1 = np.random.uniform(low=-1, high=1, size=INPUT_DIM)
    while np.all(v2 == 0):
        v1 = np.random.uniform(low=-1, high=1, size=INPUT_DIM)
    # Creating two random vectors in R^n and ensuring that they are not zero vectors
    
    u1 = v1 / np.linalg.norm(v1)
    # normalizing v1
    
    ortho = v2 - np.dot(v2,u1) * u1
    # Gram-Schmidt orthogonalization for the second vector
    
    u2 = ortho / np.linalg.norm(ortho)
    # normalizing ortho
    
    # Now u1 and u2 are two orthonormal vectors in R^n and we create
    # a matrix with u1 and u2 as the rows
    embed_matrix = np.vstack((u1,u2))
    
    X_2d = copy.copy(X)
    # storing the original 2-dimensional data points
    
    X = np.matmul(X,embed_matrix)
    # creates the new data points 
    


data = torch.from_numpy(X).float().to(DEVICE)

model = NF(INPUT_DIM, HIDDEN_DIM, N_LAYERS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("The dimension of the input is ",INPUT_DIM)
print("Starting training...")
model.train()

# just an example how it can be structured
losses = []
condition_numbers_mean = []
condition_numbers_max = []
condition_bound_mean = []

corr_coeff_first_layer = []
corr_coeff_second_layer = []
corr_coeff_sec_to_last_layer = []
corr_coeff_last_layer = []
 
# for a triangular matrix we use the lower bound 
# abs(max diagonal element) / abs(min diagonal element)

for epoch in range(N_EPOCHS+1):
    permutation = torch.randperm(data.size(0))
    for i in range(0, data.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x = data[indices]

        optimizer.zero_grad()
        jac = model.compute_jacobians(batch_x)
        
        # dummy to compute and log condition numbers
        cn_over_batch = []
        cn_bound_over_batch = []
        
        for i in range(len(jac)):
        # iterating over the layers
            for j in range(len(batch_x)):
            # iterating over the batch in the layer
                matrix = jac[i][j].detach().cpu()
                U, S, V = torch.linalg.svd(matrix)
                cn = S.max() / S.min()
                #cn = np.max(jac[i][j].detach().cpu().numpy())  # dummy
                cn_over_batch.append(cn)
                
                abs_diag = torch.abs(torch.diagonal(matrix))
                cn_bound = torch.max(abs_diag) / torch.min(abs_diag)
                cn_bound_over_batch.append(cn_bound)

        z, log_det_J = model(batch_x)
        log_prob_z =  -0.5 * torch.sum(z**2, dim=1) - 0.5 * INPUT_DIM * np.log(2 * np.pi)
        log_prob_x = log_prob_z + log_det_J
        
        
        if losses != []:
            previous_loss = loss_item
        else:
            previous_loss_torch = -torch.mean(log_prob_x)
            previous_loss = previous_loss_torch.item()
            # for the first batch; this makes the previous_loss equal the same as the loss;
            # therefore their ratio will be 1
            
        loss = -torch.mean(log_prob_x)
        loss_item = loss.item()
        

        losses.append(loss_item)
        
        arr_cn_over_batch = np.reshape(np.array(cn_over_batch),(len(jac),len(batch_x)))
        arr_log_prob_cn = np.insert(arr_cn_over_batch,0,log_prob_x.tolist(),axis=0)
        corrcoef_matrix = np.corrcoef(arr_log_prob_cn)
        corrcoef_batch_list = corrcoef_matrix[0]

        
        corr_coeff_first_layer.append(corrcoef_batch_list[1])
        corr_coeff_second_layer.append(corrcoef_batch_list[2])
        corr_coeff_sec_to_last_layer.append(corrcoef_batch_list[-2])
        corr_coeff_last_layer.append(corrcoef_batch_list[-1])

        condition_numbers_mean.append(np.nanmean(cn_over_batch))  
        condition_numbers_max.append(np.nanmax(cn_over_batch))
        condition_bound_mean.append(np.nanmean(cn_bound_over_batch))
        
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{N_EPOCHS}, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    z_samples = base_distribution.sample((N_SAMPLES,))
    x_generated = model.inverse(z_samples).cpu().numpy()
    z_transformed, _ = model(data)
    z_transformed = z_transformed.cpu().numpy()
    
if INPUT_DIM > 2:
    X = X_2d  # 2-dimensional data points generated at the beginning
    
    transp_embed_matrix = np.transpose(embed_matrix) 
    # transposed embedding matrix
    
    x_generated = np.matmul(x_generated,transp_embed_matrix)
    # mapping the subspace in R^n back to R^2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('NF for dataset X', fontsize=16)

axes[0].set_title("Target Distribution")
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6, c='blue')
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[0].set_aspect('equal', adjustable='box')

axes[1].set_title("Samples from Base $\Rightarrow$ Target")
axes[1].scatter(x_generated[:, 0], x_generated[:, 1], alpha=0.6, c='green')
x_min, x_max = x_generated[:, 0].min() - 0.5, x_generated[:, 0].max() + 0.5
y_min, y_max = x_generated[:, 1].min() - 0.5, x_generated[:, 1].max() + 0.5
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].set_aspect('equal', adjustable='box')

axes[2].set_title("Data $\Rightarrow$  Base")
axes[2].scatter(z_transformed[:, 0], z_transformed[:, 1], alpha=0.6, c='red')
x_min, x_max = z_transformed[:, 0].min() - 0.5, z_transformed[:, 0].max() + 0.5
y_min, y_max = z_transformed[:, 1].min() - 0.5, z_transformed[:, 1].max() + 0.5
axes[2].set_xlim(x_min, x_max)
axes[2].set_ylim(y_min, y_max)
axes[2].set_aspect('equal', adjustable='box')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title('Loss over Batches')
plt.xlabel('Batch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(condition_numbers_mean)
plt.title('Mean Condition Number over Batches')
plt.xlabel('Batch')
plt.ylabel('Condition Number')

plt.subplot(2, 2, 3)
plt.plot(condition_numbers_max)
plt.title('Maximum Condition Number over Batches')
plt.xlabel('Batch')
plt.ylabel('Condition Number')

plt.subplot(2, 2, 4)
plt.plot(condition_bound_mean)
plt.title('Mean Condition Number Bound over Batches')
plt.xlabel('Batch')
plt.ylabel('Condition Number Bound')

plt.tight_layout()


plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(corr_coeff_first_layer)
mean_y = np.mean(corr_coeff_first_layer)
plt.axhline(mean_y, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_y:.2f}')
plt.title('Correlation log_prob_x and cond. number of first layer')
plt.xlabel('Batch')
plt.ylabel('Correlation coefficient')

plt.subplot(2, 2, 2)
plt.plot(corr_coeff_second_layer)
mean_y = np.mean(corr_coeff_second_layer)
plt.axhline(mean_y, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_y:.2f}')
plt.title('Correlation log_prob_x and cond. number of second layer')
plt.xlabel('Batch')
plt.ylabel('Correlation coefficient')

plt.subplot(2, 2, 3)
plt.plot(corr_coeff_sec_to_last_layer)
mean_y = np.mean(corr_coeff_sec_to_last_layer)
plt.axhline(mean_y, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_y:.2f}')
plt.title('Correlation log_prob_x and cond. number of second to last layer')
plt.xlabel('Batch')
plt.ylabel('Correlation coefficient')

plt.subplot(2, 2, 4)
plt.plot(corr_coeff_last_layer)
mean_y = np.mean(corr_coeff_last_layer)
plt.axhline(mean_y, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_y:.2f}')
plt.title('Correlation log_prob_x and cond. number of last layer')
plt.xlabel('Batch')
plt.ylabel('Correlation coefficient')

plt.tight_layout()
plt.show()