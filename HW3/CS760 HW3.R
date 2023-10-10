# load in libraries
library(DescTools)
library(ggplot2)
library(ROCR)

# kNN with n-fold cross validation
euclidean_d <- function(x1, x2) {
  sqrt(sum((x1 - x2)^2))
}


knn_fxn1 <- function(training, test, k) {
  train_X <- training[,1:2]
  dist <- apply(train_X, 1, function(x) euclidean_d(x, test))
  nn <- order(dist)[1:k]
  if (k == 1){
    prediction <- training[nn, 'y']
  }
  else {
    prediction <- Mode(c(training[nn, 'y']))
  }
  return(prediction)
}

knn_fxn2 <- function(training, test, k) {
  train_X <- training
  dist <- apply(train_X, 1, function(x) euclidean_d(x, test))
  nn <- order(dist)[1:k]
  if (k == 1){
    prediction <- training[nn, 'y']
  }
  else {
    prediction <- Mode(c(training[nn, 'y']))
  }
  return(prediction)
}


cv_knn <- function(data, k, n) {
  set.seed(7603)
  fold_size <- nrow(data) / n
  accuracy <- precision <- recall <- numeric(n)
  for (i in 1:n) {
    test_ind <- (1+(i-1) * fold_size):(i*fold_size)
    test_data <- data[test_ind,]
    training_data <- data[-test_ind,]
    predictions <- sapply(1:nrow(test_data), function(x) knn_fxn2(training_data, test_data[x, 1:3000], k))
    actual <- test_data$Prediction
    TP <- sum(predictions == 1 & actual == 1)
    FP <- sum(predictions == 1 & actual == 0)
    TN <- sum(predictions == 0 & actual == 0)
    FN <- sum(predictions == 0 & actual == 1)
    total <- nrow(test_data)
    accuracy[i] <- (TP + TN) / total
    recall[i] <- TP / (TP + FN)
    precision[i] <- TP / (TP + FP)
  }
  res <- rbind(accuracy,recall,precision)
  return(res)
}

# Logistic Regression with n-fold cross validation
sigmoid <- function(z) {
  return(1 / (1 + exp(-z)))
}

gradient_descent <- function(X, y, learning_rate, niter) {
  n <- length(y)
  p <- ncol(X)
  theta <- rep(0, p)
  X <- as.matrix(X)
  
  for (i in 1:niter) {
    z <- X %*% theta
    s <- sigmoid(z)
    gradient <- t(X) %*% (s - y) / n
    theta <- theta - learning_rate * gradient
  }
  return(theta)
}


cv_lr <- function(data, learning_rate, niter, n) {
  set.seed(7603)
  fold_size <- nrow(data) / n
  accuracy <- precision <- recall <- numeric(n)
  for (i in 1:n) {
    test_ind <- (1+(i-1) * fold_size):(i*fold_size)
    test_data <- data[test_ind,]
    training_data <- data[-test_ind,]
    theta <- gradient_descent(training_data[,1:3000],training_data$Prediction,learning_rate,niter)
    predictions <- t(theta) %*% test_data[,1:3000]
    actual <- test_data$Prediction
    TP <- sum(predictions == 1 & actual == 1)
    FP <- sum(predictions == 1 & actual == 0)
    TN <- sum(predictions == 0 & actual == 0)
    FN <- sum(predictions == 0 & actual == 1)
    total <- nrow(test_data)
    accuracy[i] <- (TP + TN) / total
    recall[i] <- TP / (TP + FN)
    precision[i] <- TP / (TP + FP)
  }
  res <- rbind(accuracy,recall,precision)
  return(res)
}


## Q1 
data <- read.table('hw3Data-3/hw3Data/D2z.txt')
colnames(data) <- c('x1','x2','y')
training_data <- data
domain <- seq(-2,2,0.1)
test_data <- data.frame(x1=unlist(lapply(domain,function(x){rep(x,41)})),x2=rep(domain,41))
predictions <- sapply(1:nrow(test_data), function(j) knn_fxn1(training_data, test_data[j,], 1))
test_res <- cbind(test_data,y=as.character(predictions))
p <- ggplot(test_res, aes(x = x1, y = x2)) +
  geom_point(aes(color = y))
p <- p + geom_point(data = training_data, aes(x = x1, y = x2, shape = factor(y)),
                    color = "black", size = 3) +
  scale_shape_manual(values = c(1, 3)) + theme(legend.position = "none")
print(p)

## Q2
data2 <- read.csv('hw3Data-3/hw3Data/emails.csv') 
rownames(data2) <- data2[,1]
data2 <- data2[,-1]
res_df2 <- cv_knn(data2,1,5)

## Q3
res_df3 <- cv_lr(data2,0.1,5000,5)

## Q4
sp1 <- cv_knn(data2,1,5)
sp2 <- cv_knn(data2,3,5)
sp3 <- cv_knn(data2,5,5)
sp4 <- cv_knn(data2,7,5)
sp5 <- cv_knn(data2,10,5)
# plot out average accuracy from results above

## Q5
train_data <- data2[1:4000,1:3000]
test <- data2[4001:5000,1:3000]
y <- data2$Prediction[1:4000]
y_test <- data2$Prediction[4001:5000]