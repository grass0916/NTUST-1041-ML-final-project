CNN <- function (train, test) {
  # input
  data <- mx.symbol.Variable('data')
  # first conv
  conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
  tanh1 <- mx.symbol.Activation(data=conv1, act_type="relu")
  pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # second conv
  conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
  tanh2 <- mx.symbol.Activation(data=conv2, act_type="relu")
  pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # first fullc
  flatten <- mx.symbol.Flatten(data=pool2)
  fc1     <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  tanh3   <- mx.symbol.Activation(data=fc1, act_type="relu")
  # second fullc
  # num_hidden = 10, means that labels are number 0 ~ 9.
  fc2     <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
  # loss
  lenet <- mx.symbol.SoftmaxOutput(data=fc2)
  
  train.array <- train.x
  dim(train.array) <- c(28, 28, 1, ncol(train.x))
  test.array <- test
  dim(test.array) <- c(28, 28, 1, ncol(test))
  
  device.cpu <- mx.cpu()
  
  # Common computing.
  mx.set.seed(0)
  tic <- proc.time()
  model <- mx.model.FeedForward.create(lenet,
                                       X=train.array,
                                       y=train.y,
                                       ctx=device.cpu,
                                       num.round=30,
                                       array.batch.size=100,
                                       learning.rate=0.05,
                                       momentum=0.9,
                                       wd=0.00001,
                                       eval.metric=mx.metric.accuracy,
                                       epoch.end.callback=mx.callback.log.train.metric(100))
  print(proc.time() - tic)
  
  preds <- predict(model, test.array)
  pred.label <- max.col(t(preds)) - 1
  submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
  dir.create('./outputs/', showWarnings=FALSE)
  write.csv(submission, file='./outputs/submission(CNN).csv', row.names=FALSE, quote=FALSE)
}