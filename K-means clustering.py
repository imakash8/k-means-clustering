import numpy as np

class matrix(object):
    '''This represent 2-D array'''
                
    def __init__(self,file_name='wine_dataset.csv'):
        self.array_2d = self.load_from_csv(file_name)
        self.array_2d = self.standardise()
        self.n,self.m = self.array_2d.shape[0], self.array_2d.shape[1]
        self.centroids = np.array([])
        self.S = np.zeros((self.n))
        self.weights = get_initial_weights(self.m)

    def load_from_csv(self,file_name):
        data = open(file_name)
        return np.loadtxt(data, delimiter=",")

    def standardise(self):
        Max,Min,Avg = np.max(self.array_2d, axis=0), np.min(self.array_2d, axis=0), np.mean(self.array_2d, axis=0)
        return (self.array_2d-Avg)/(Max-Min)

    #calculating distance from centroid from rows of data  
    def get_distance(self,centroids,weights,beta):
        d = (weights**beta)*(self.row - centroids)**2
        return np.sum(d,axis=1,keepdims = True)

    def get_count_frequency(self):
        k,frequency = np.unique(self.S, return_counts=True)
        return {int(k[i])+1: frequency[i] for i in range(len(k))}

#weight intialised in class
def get_initial_weights(m):
    weight = np.random.rand(m).reshape(1,m)
    return weight/weight.sum()

#finding centroid from the cluster
def get_centroids(m,S,K):
    centroids = np.vstack([np.nanmean(m.array_2d[S==i,:],axis=0) for i in range(K)])
    return centroids

# extra function defined  for for better visualisation of code
# get_S is use to calculate vector S on input of centroid
def get_S(m,centroids,beta):
    for i in range(m.n):
        m.row = m.array_2d[i]
        distance = m.get_distance(centroids,m.weights,beta)
        m.S[i] = np.argmin(distance,axis=0)
    return m.S

def get_groups(m,k,beta):
    assert k >= 2 and k<m.n-1                      #check k is in allowed range
    m.beta= beta                                    #attribute beta for calculating get_new_weights
    #Randomly selecting K rows from the data
    centroids =  m.array_2d[np.random.choice(m.n, k,replace=False),:]
    P=get_S(m,centroids,beta)
    #used range(100) instead while True so that on heavy data it should not stuck on in fine loop
    for i in range(100):  
        check=np.copy(P)
        centroids=get_centroids(m,P,k)              #updating centroid
        temp=get_S(m,centroids,beta)                #calculating matrix S
        m.weights = get_new_weights(m,P,temp)       #weight update
        if np.array_equal(check,temp):break         #checking that no further change in cluster
        check = temp
    return m
           
def get_new_weights(m,cenroids,S):
    J = np.zeros(m.m)                               #intialising vector J with m no of columns
    for K in range(cenroids.shape[0]):              #calculating delta J 
        J += (np.where(S == K, 1, 0).reshape(m.n,1)*((m.array_2d-cenroids[K])**2)).sum(0)
    return 1/((sum(J.reshape(-1,1)/J))**(1/(m.beta-1)))

def run_test():
    m = matrix('wine_dataset.csv')
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10)
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))
run_test()
