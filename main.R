# Your project path.
setwd('/Users/Salmon/GitHub/NTUST-1041-ML-final-project/')

# Step 0, the guide to build the Shared Library. 'OpenCV' and 'mxnet'.
# Link, http://mxnet.readthedocs.org/en/latest/build.html

# Please move the '#' below to install library, if you first time to execute.
# install.packages('drat', repos='https://cran.rstudio.com')
# drat:::addRepo('dmlc')
# install.packages('mxnet')

require(mxnet)
require(methods)


train <- read.csv('./datasets/train.csv', header=TRUE)
test  <- read.csv('./datasets/test.csv', header=TRUE)
train <- data.matrix(train)
test  <- data.matrix(test)

# 784 pixels of number image.
train.x <- train[,-1]
# Label of number. e.g. 1, 2, 3, ... , 9, 0.
train.y <- train[,1]

train.x <- t(train.x/255)
test    <- t(test/255)

table(train.y)

# function : classify.
source('./src/classification.R', local=TRUE)
classify(train, test)

# function : CNN
source('./src/convolutional_neural_networks.R', local=TRUE)
CNN(train, test)
