import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

import ackley_problem, xsinx_problem


from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plot(network):
    X = numpy.arange(-5, 5, .2)
    Y = numpy.arange(-5, 5, .2)
    X, Y = numpy.meshgrid(X, Y)
    Z=numpy.zeros_like(X)
    li_arr=[]
    for x_index in range(len(X)):
        for y_index in range(len(X)):
            arr=torch.FloatTensor(numpy.array([X[x_index,y_index],Y[x_index,y_index]]).reshape(1,2))
            dummy=torch.FloatTensor(numpy.zeros((1,10)))
            #print(arr.shape)
            #print(dummy.shape)
            #assert False
            Z[x_index,y_index]=network(dummy,arr)[0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    '''
    a_li=network.anchor_points.predict([dummy])
    xs=[a[0,0] for a in a_li]
    ys=[a[0,1] for a in a_li]
    print(xs)
    a_li=[a.flatten() for a in a_li]
    a_li=numpy.array(a_li)
    QRef=network.network.predict([numpy.zeros((len(a_li),1)),a_li])
    QRef=[q for q in QRef]
    ax.scatter(xs, ys, QRef, marker='^',color='black',alpha=1)
    '''
    # Plot the surface.
    ax.view_init(elev=32., azim=46)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,alpha=0.5,
                           linewidth=0, antialiased=True)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$a_1$")
    ax.set_ylabel(r"$a_0$")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$\widehat r(a;\theta)$",rotation=0,labelpad=10)
    plt.show()


def rbf_function(centroid_locations, action, beta, N):
    centroid_locations_squeezed = [l.unsqueeze(1) for l in centroid_locations]
    centroid_locations_cat = torch.cat(centroid_locations_squeezed, dim=1)
    action_unsqueezed = action.unsqueeze(1)
    action_cat = torch.cat([action_unsqueezed for _ in range(N)], dim=1)
    diff = centroid_locations_cat - action_cat
    diff_norm = torch.norm(diff,p=2,dim=2)
    diff_norm_smoothed_negated = diff_norm * beta * -1
    output = F.softmax(diff_norm_smoothed_negated, dim=1)
    return output 

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #params
        self.N = 30
        self.beta = 3
        self.state_size = 10
        self.action_size = 2
        self.max_a = 5
        self.lr = 0.01
        #params

        self.value_side1 = nn.Linear(self.state_size, 100)
        self.value_side1_parameters = self.value_side1.parameters()
        self.value_side2 = nn.Linear(100, self.N)
        self.value_side2_parameters = self.value_side2.parameters()

        self.location_side1 = nn.Linear(self.state_size, 100)
        self.location_side2 = []
        for _ in range(self.N):
            self.location_side2.append(nn.Linear(100, self.action_size))
        self.criterion = nn.MSELoss()
        #print(list(self.parameters()))
        #assert False
        #assert False
        #print([x.shape for x in list(self.parameters())])
        #assert False
        #print(list(self.parameters()))
        #self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        
        params_dic=[]
        params_dic.append({'params': self.value_side1_parameters, 'lr': 1e-2})
        params_dic.append({'params': self.value_side2_parameters, 'lr': 1e-2})
        params_dic.append({'params': self.location_side1.parameters(), 'lr': 5e-3}) 
        for i in range(self.N):
            params_dic.append({'params': self.location_side2[i].parameters(), 'lr': 5e-3}) 
        self.optimizer = optim.Adam(params_dic)
        
    def forward(self, s, a):
        centroid_values = self.get_centroid_values(s)
        centroid_locations = self.get_all_centroids(s)
        centroid_weights = rbf_function(centroid_locations, a, self.beta, self.N)
        output = torch.mul(centroid_weights,centroid_values)
        output = output.sum(1,keepdim=True)
        return output

    def update(self,s,a,y):
        self.optimizer.zero_grad()
        y_hat = net(s,a)
        self.loss = self.criterion(y_hat,y)
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.loss

    def get_all_centroids(self, s):
        temp = F.relu(self.location_side1(s))
        centroid_locations = []
        for i in range(self.N):
            centroid_locations.append( self.max_a*torch.tanh(self.location_side2[i](temp)) )
        return centroid_locations
        
    def get_best_centroid(self, s, maxOrmin='max'):
        all_centroids = self.get_all_centroids(s)
        all_centroids_matrix = torch.cat(all_centroids, dim=0)
        weights = rbf_function(all_centroids, all_centroids_matrix, self.beta, self.N)
        values = self.get_centroid_values(s)
        values = torch.transpose(values, 0, 1)
        temp = torch.mm(weights,values)
        if maxOrmin=='max':
            values, indices = temp.max(0)
        elif maxOrmin=='min':
            values, indices = temp.min(0)
        Q_star = values.data.numpy()[0]
        index_star = indices.data.numpy()[0]
        a_star = list(all_centroids[index_star].data.numpy()[0])
        return Q_star, a_star

    def get_centroid_values(self, s):
        temp = F.relu(self.value_side1(s))
        centroid_values = self.value_side2(temp)
        return centroid_values

numpy.random.seed(0)
torch.manual_seed(7)
net = Net()

sampler_function=ackley_problem.ackley_function_get_batch

max_iter = 2000
batch_size = 256
for counter in range(max_iter):
    x_batch,y_batch = sampler_function(batch_size)
    s,a,y = torch.FloatTensor(numpy.zeros((batch_size,10))),torch.FloatTensor(x_batch),torch.FloatTensor(y_batch)
    loss = net.update(s,a,y)
    if counter%50 == 0:
        print("loss: ",loss)

s = torch.FloatTensor(numpy.zeros((1,10)))
Q_star, a_star = net.get_best_centroid(s)
print(Q_star,a_star)
Q_star, a_star = net.get_best_centroid(s,maxOrmin = 'min')
print(Q_star,a_star)
assert False


