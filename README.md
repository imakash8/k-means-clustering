# k-means-clustering
Build a K-means clustering algorithm form scratch using python Numpy library. 

Identifying which wines are similar from wind_dataset.csv

Technical details:
You must contain the code for the functions and methods below. If you wish you can write more functions and methods, but those described below must be present.
1) Class: matrix You will code a class called matrix, which will have an attribute called array_2d. This attribute is supposed to be a NumPy array containing numbers in two dimensions. The class matrix must have the following methods:
(in these, the parameters are in addition to self)
1.1) load_from_csv
This method should have one parameter, a file name (including, if necessary, its path and extension). This method should read this CSV file and load its data to the array_2d of matrix. Each row in this file should be a row in array_2d. Notice that in CSV files a comma separates columns (CSV = comma separated values).
You should also write code so that
m = matrix(‘validfilename.csv’)
Creates a matrix m with the data in the file above in array_2d.
1.2) standardise
This method should have no parameters. It should standardise the array_2d in the matrix calling this method. For details on how to standardise a matrix, read the appendix.
1.3) get_distance
This method should have three parameters, two matrices (let us call them other_matrix and weights) and a number (let us call it beta). If the matrix calling this method and the matrix weights have only one row, this function should return a matrix containing the weighted Euclidean distance between the row in the matrix calling this method and each of the rows in other_matrix. For details about how to calculate this distance, read the appendix.
To be clear: if other_matrix has n rows, the matrix returned in this method will have n rows and 1 column.
1.4) get_count_frequency
This method should have no parametes, and it should work if the array_2d of the matrix calling this method has only one column. This method should return a dictionary mapping each element of the array_2d to the number of times this element appears in array_2d.
2) Functions
The code should also have the functions (i.e. not methods, so not part of the class matrix) below. No code should be outside any function or method in this assignment.
get_initial_weights
This function should have one parameter, an integer m. This function should return a matrix with 1 row and m columns containing random values, each between zero and one. The sum of these m values should be equal to one.
 3
get_centroids
This function should have three parameters: (i) a matrix containing the data, (iii) the matrix S, (iii) the value of K. This function should implement the Step 9 of the algorithm described in the appendix. It should return a matrix containing K rows and the same number of columns as the matrix containing the data.
get_groups
This function should have three parameters: a matrix containing the data, and the number of groups to be created (K), and a number beta (for the distance calculation). This function follows the algorithm described in the appendix. It should return a matrix S (defined in the appendix). This function should use the other functions you wrote as much as possible. Do not keep repeating code you already wrote.
get_new_weights
This function takes three parameters: a matrix containing the data, a matrix containing the centroids, and a matrix S (see the algorithm in the Appendix). This function should return a new matrix weights with 1 row and as many columns as the matrix containing the data (and the matrix containing the centroids). Follow Step 10 of the algorithm in the Appendix.
run_test
Your code must contain the function below (do not change anything)
def run_test():
   m = matrix(‘Data.csv’)
   for k in range(2,5):
for beta in range(11,25):
S = get_groups(m, k, beta/10) print(str(k)+‘-’+str(beta)+‘=’+str(S.get_count_frequency()))
The aim of this function is just to run a series of tests. By consequence, here (and only here) you can use hard-coded values for the strings containing the filenames of data and values for K.
More details
You will implement a data-driven algorithm that creates groups of entities (here, an entity is a wine, described as a row in our data matrix) that are similar. If two entities are assigned to the same group by the algorithm, it means they are similar. This will create groups of similar wines. Your software just needs the number of groups the user wants to partition the data into, the data itself, and a numeric value for Beta.
The number of partitions (K) is clearly a positive integer. Your software should only allow values in the interval [2, n-1], where n is the number of rows in the data. This way you’ll avoid trivial partitions. You can test values of Beta that are higher than 1.
Your software should follow the algorithm described in the appendix and generate a matrix S indicating to which group (1, 2, ..., K) each entity (wine, a row in the data matrix) has been assigned to. Clearly S will have n elements.
You can find more information online if you search for clustering, or k-means (this is not the algorithm we are implementing, but it is similar).
For the brave:
You are implementing this algorithm: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1407871 If you cannot understand the paper in the link, do NOT panic. You do NOT need to understand the paper above in order to do this assignment. You CAN get an excellent mark in this assignment without even opening this link. If you are not strong in maths (multivariable calculus, in particular) do NOT open the link.
 4

Appendix
Data standardization
Let 𝐷 be a data matrix, so that 𝐷𝑖𝑗 is the value of 𝐷 at row i and column j. You can standardize 𝐷 by following the equation below.
 𝐷−𝐷 𝐷′𝑖𝑗 = 𝑖𝑗 𝑗
 𝑚𝑎𝑥(𝐷 ) − 𝑚𝑖𝑛⁡(𝐷 ) 𝑗𝑗
where 𝐷 is the average of column j, 𝑚𝑎𝑥(𝐷 ) is the highest value in column j, and 𝑚𝑖𝑛⁡(𝐷 ) is the 𝑗𝑗𝑗
lowest value in column j. 𝐷′𝑖𝑗 is the standardized version of 𝐷𝑖𝑗 – the algorithm below should only be applied to 𝐷′𝑖𝑗 (i.e. do not apply the algorithm below to 𝐷𝑖𝑗).
Basic notation for the algorithm
n = number of rows of in the data matrix
m = number of columns in the data matrix
K = number of clusters (notice that k is not the same thing as K) Beta = exponent used in the distance calculation
Clustering algorithm
1. Set a positive value for K, and a positive value for Beta.
2. Initialise a matrix called weights with 1 row and m columns. Each value in this matrix should
be between zero and one, and the sum of all values in weights should be equal to one.
3. Create an empty matrix called centroids.
4. Create a matrix called S with n rows and 1 column, initialise all of its elements to zero.
5. Select K different rows from the data matrix at random.
6. For each of the selected rows
a. Copy its values to the matrix centroids.
(at the end of step 6, centroids should have K rows and m columns)
7. For each row i in the data matrix
a. Calculate the weighted Euclidean distance between data row 𝐷′𝑖 and each row in centroids (use weights and Beta in this calculation, as per equation in the Appendix).
b. Set 𝑆𝑖 to be equal to the index of the row in centroids that is the nearest to the row
𝐷′𝑖. For instance, if the nearest row in centroids is row 3, then assign the number 3 to
row i in S.
8. If the previous step does not change S, stop.
9. For each k = 1, 2, ..., K
a. Update the k row in centroids. Each element j of this row should be equal to the mean
of the column 𝐷′ but only taking into consideration those rows whose value in S is 𝑗
equal to k (i.e. those who have been assigned to cluster k). 10. For each v = 1, 2, ..., m
a. Update the entry v of the matrix weights (see Appendix). 11. Go to Step 7.
Weighted Euclidean distance
There are different weighted Euclidean distances, in this assignment you must follow the below. The distance between a vector a and a vector b, using the weights in a vector w (all three vectors with
size m), with a value for Beta 𝛽 is given by:
𝑚
𝑑=∑𝑤𝛽(𝑎𝑖 −𝑏𝑖)2 𝑖
   𝑖=1
5

Calculating weights
Weights (w in the below) is a vector (a matrix with 1 row and m columns in your implementation). To
calculate the entry j of this vector (i.e. 𝑤 , that is, row 1 column j of the matrix weights) we first need to 𝑗
calculate the dispersion of the column j in the data matrix:
𝐾𝑛
∆=∑∑𝑢 (𝐷′ −𝑐 )2
𝑘=1 𝑖=1
 where,
n is the number of rows in the data matrix
m is the number of columns in the data matrix (used below)
K is the number of clusters (remember k is not the same thing as K) c is the matrix centroids
𝑢𝑖𝑘 is equal to one if 𝑆𝑖 = 𝑘, and zero otherwise.
If the value ∆ = 0 then 𝑤 = 0, otherwise: 𝑗𝑗
𝑗
𝑖𝑘 𝑖𝑗 𝑘𝑗
1 𝑗
𝑤=
1 ∆ 𝛽−1
 ∑𝑚 [𝑗] 𝑡=1 ∆𝑡
