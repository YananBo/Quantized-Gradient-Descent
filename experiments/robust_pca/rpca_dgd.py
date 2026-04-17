from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.linalg import svd
import time
import csv
import copy
import pandas as pd

class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
class Load_Sequence: 
    def __init__(self, num_agents, batch_size=200):
        self.videos = [
            "./videos/Bootstrap",
            "./videos/Camouflage",
            "./videos/ForegroundAperture",
            "./videos/LightSwitch",
            "./videos/MovedObject",
            "./videos/TimeOfDay",
            "./videos/WavingTrees"
        ]

        self.transform = transforms.Compose([
            transforms.Resize((56, 56)),  # 调整图片大小
            transforms.ToTensor()         # 将图片转换为Tensor
        ])

        self.batch_size = batch_size
        self.num_agents = num_agents

    def load_data(self):
        
        dataset = SimpleDataset(root_dir=self.videos[0], transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size * self.num_agents, shuffle=True)

        for images in dataloader:
            # print(images.shape)  # torch.Size([batch_size, 3, 56, 56])
            images_flattened = images.view(self.batch_size * self.num_agents, -1)
            # print(images_flattened.shape)  
            break
        
        agent_data = torch.chunk(images_flattened, num_agents, dim=0)
        print(agent_data[0].shape)
        return agent_data

    
def sparse_operator(tensor, alpha):
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    num_rows, num_cols = tensor.shape
    num_elements_to_keep_row = max(int(num_cols * alpha), 1)
    num_elements_to_keep_col = max(int(num_rows * alpha), 1)

   
    mask = torch.zeros_like(tensor, dtype=torch.bool)

  
    topk_row_values, topk_row_indices = torch.topk(tensor.abs(), k=num_elements_to_keep_row, dim=1)
    row_thresholds = topk_row_values[:, -1].unsqueeze(1).expand_as(tensor)
    mask |= tensor.abs() >= row_thresholds

    
    topk_col_values, topk_col_indices = torch.topk(tensor.abs(), k=num_elements_to_keep_col, dim=0)
    col_thresholds = topk_col_values[-1, :].unsqueeze(0).expand_as(tensor)
    mask &= tensor.abs() >= col_thresholds

    sparse_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))

    return sparse_tensor


def initialize_UV_S(D_tensor, r, device):

    # D_tensor = torch.tensor(D).clone().detach() # Convert NumPy array D to PyTorch tensor
    S_0 = sparse_operator(D_tensor, alpha=0.2)

    U_0, Sigma_0, V_0 = svd(D_tensor)
    
    U_0 = U_0[:, :r].to(device)
    Sigma_0 = torch.diag(Sigma_0[:r]).to(device)
    V_0 = V_0[:r, :].to(device)

    U = U_0 @ torch.sqrt(Sigma_0[:r, :r])
    V = torch.sqrt(Sigma_0[:r, :r]) @ V_0

    L = U_0 @ Sigma_0 @ V_0
    S = sparse_operator(D_tensor - L, alpha=0.2)
    
    return U_0, V_0, S

def initialize_agents(agent_matrix, r, num_agents, device):
    
    U_temp, V_temp, S_temp = initialize_UV_S(agent_matrix[0], r, device)

    U_x_dim, U_y_dim = U_temp.shape
    V_x_dim, V_y_dim = V_temp.shape
    S_x_dim, S_y_dim = S_temp.shape

    
    agents_U = torch.empty(num_agents, U_x_dim, U_y_dim, device=device)
    agents_V = torch.empty(num_agents, V_x_dim, V_y_dim, device=device)
    agents_S = torch.empty(num_agents, S_x_dim, S_y_dim, device=device)

    for i in range(num_agents):   

        U_temp, V_temp, S_temp = initialize_UV_S(agent_matrix[i], r, device)
        agents_U[i] = U_temp
        agents_V[i] = V_temp
        agents_S[i] = S_temp

    return agents_U, agents_V, agents_S


def log_to_csv(filename, iteration, loss_value, rec_error):

    
    """
    Log iteration, loss value, and reconstruction error to a CSV file.

    Parameters:
    - filename: Name of the CSV file
    - iteration: Current iteration
    - loss_value: Loss value
    - rec_error: Reconstruction error
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, lineterminator = '\n')
        writer.writerow([iteration])
        writer.writerow(loss_value)
        
        
        
        writer.writerow([rec_error])
        writer.writerow([])
        
def robust_pca_gradient_descent(agent_matrix_tensor, filename, r, agents_U, agents_V, agents_S,
                                num_agents, w,  mu, alpha, device,
                                learning_rate, max_iter):
    
    
    agents_U.to(device)
    agents_V.to(device)
    agents_S.to(device)
    w.to(device)
    err = []
    loss_value = []
    new_agents_U = agents_U.detach().clone()
    new_agents_V = agents_V.detach().clone()
    new_agents_S = agents_S.detach().clone()
    
    mu_inv = 1 / mu
    
    # print(agents_U.shape)
    
    # Consensus 
    for j in range(num_agents):
        
        new_agents_U[j] = torch.sum(w[j].unsqueeze(-1).unsqueeze(-1) * agents_U, dim=0)
        new_agents_V[j] = torch.sum(w[j].unsqueeze(-1).unsqueeze(-1) * agents_V, dim=0)
        
    for i in range(num_agents):

        loss = torch.norm(agent_matrix_tensor[i] - torch.matmul(agents_U[i], agents_V[i]) - agents_S[i])**2 + mu_inv * (torch.norm(agents_U[i])**2 + torch.norm(agents_V[i])**2)
        
        gradients_U = []
        gradients_V = []
        
        E = agent_matrix_tensor[i] - torch.matmul(agents_U[i], agents_V[i]) - agents_S[i]

        gradient_U = -2 * torch.matmul(E, agents_V[i].T) + 2 * mu_inv * agents_U[i]
        gradient_V = -2 * torch.matmul(agents_U[i].T, E) + 2 * mu_inv * agents_V[i]
        
        agents_U[i] = new_agents_U[i] - learning_rate * gradient_U
        agents_V[i] = new_agents_V[i] - learning_rate * gradient_V    

        # Recompute S after updating U, V
        agents_S[i] = sparse_operator(agent_matrix_tensor[i] - torch.matmul(agents_U[i], agents_V[i]), alpha=0.2)

        # Compute error
        # norm_sum = sum(torch.norm(tensor) for tensor in agent_matrix_tensor)
        norm_sum = torch.norm(agent_matrix_tensor[i])
        err_temp = torch.norm(agent_matrix_tensor[i] - torch.matmul(agents_U[i], agents_V[i]) - agents_S[i])/ norm_sum   
        loss_value.append(loss.item()) 
        err.append(err_temp.item())    
    
   
    # Print iteration information
    err_mean = sum(err) / len(err)
    # print(err_mean)

    return agents_U, agents_V, agents_S, loss_value, err_mean


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_iterations = 5000
num_agents = 5
w = np.array([[0.6, 0,   0,   0.4, 0],
              [0.2, 0.8, 0,   0,   0],
              [0.2, 0.1, 0.4, 0,   0.3],
              [0,   0,   0,   0.6, 0.4],
              [0,   0.1, 0.6, 0,   0.3]])

w=torch.tensor(w).float().to(device)
# print(w)
ranks = 30 #rank
mu = 100 # multiplier
learning_rate = 0.5e-4
tol = 1e-5
iter_print = 1000
iter_log = 1

loader = Load_Sequence(num_agents, batch_size = 200)
agent_data_chunks = loader.load_data()
agent_data = tuple(chunk.to(device) for chunk in agent_data_chunks)



cwd = os.getcwd() # Get the current working directory
loca=time.strftime('%Y-%m-%d-%H-%M-%S')

results_path = os.path.join(cwd, "results") # Construct the path for the "results" directory
filename = os.path.join(results_path, f"A_ERRORBAR:DGD_{str(loca)}_I{n_iterations}_lr:{learning_rate}_ranks:{ranks}_mu_{mu}.csv") 
print(filename)   
    
if not os.path.isdir(results_path): # Check if the "results" directory does not exist
    os.mkdir(results_path)   # Create the "results" directory if it doesn't exist
    

    
    
agents_U, agents_V, agents_S = initialize_agents(agent_data, ranks, num_agents, device)

for iteration in range(n_iterations):

    agents_U, agents_V, agents_S, loss_value, err_mean = robust_pca_gradient_descent(agent_data, filename, r=ranks, agents_U=agents_U,
                                                                                   agents_V=agents_V, agents_S=agents_S,
                                                                                   num_agents=num_agents, w=w,
                                                                                   mu=mu, alpha=0.2, device=device,
                                                                                   learning_rate=learning_rate,
                                                                                   max_iter=n_iterations)
                                                                                
    
    # if (iteration % iter_print) == 0 or iteration == 1 or iteration >= n_iterations or err_mean <= tol:
    #     print('iteration: {0}, Rec. error: {1}, loss: {2}'.format(iteration, err_mean, loss_value)) 
    if iteration % iter_log == 0:
        log_to_csv(filename, iteration, loss_value, err_mean)
    if err_mean < tol:
        break 
