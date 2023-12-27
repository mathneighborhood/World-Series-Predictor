install.packages("tidyverse")
install.packages('rvest')
install.packages('ggcorrplot')
install.packages('ROCR')
library(pROC)
library(ROCR)
library(plotly)
library(ggcorrplot)
library(caret)
library(dplyr)
library(MASS)
library(GGally)
library(tidyverse)
library(class)
library(readr)
library(rvest)
library(ggplot2)
library(readxl)
 
full.team.data <- data.frame(read_xls("/Users/jon_southam/Desktop/Hopkins/Data Mining/Baseball_Data/fullteamdata.aug.xls"))
View(full.team.data)

set.seed(2718)

### Separate full data into 2023 season and the other seasons 
### to use 2023 as a fun test of our model.
team.data.23 <- full.team.data[c(1499:1528),]
team.data <- full.team.data[-c(1499:1528),]

### Remove 1981, 1994, and 2020 seasons due to covid and strike
team.data <- team.data[team.data$year != 2020, ]
team.data <- team.data[team.data$year != 1981, ]
team.data <- team.data[team.data$year != 1994, ]

##################################
## Smaller sample size
ws.teams <- team.data[team.data$WS == 1,]
not.ws.teams <- team.data[team.data$WS == 0,]
not.ws.teams.sample <- not.ws.teams %>% slice_sample(n = 200)

x <- sample(not.ws.teams,200)

small.data <- rbind(ws.teams,not.ws.teams.sample)
small.index <- sample(1:nrow(small.data), nrow(small.data) / 2)
small.training1 <- small.data[small.index, ]
small.testing1 <- small.data[-small.index, ]

smodel <- glm(WS~SB + OPS + tSho ,data = small.training1[,-c(1)],family = binomial)
summary(smodel)

pred.WS.glm.small <- predict(smodel,small.testing1,type = 'response')

rock <- roc(small.testing1$WS,pred.WS.glm.small,
            plot = TRUE,
            legacy.axes = TRUE,
            xlab = "False Positive Rate",
            ylab = 'True Positive Rate',
            main = "Logistic Model ROC Curve")
auc(rock)
# Convert predicted probabilities to binary values (0 or 1)
predictions.log <- ifelse(pred.WS.glm.small >= 0.5, 1, 0)

# Create a confusion matrix
confusion.matrix.log <- table(Actual = small.testing1$WS, Predicted = predictions.log)

# Print the confusion matrix
print(confusion.matrix.log)

#Predict the 2023 season
pred.WS.glm.23 <- predict(smodel,team.data.23,type = 'response')
predictions.log.small <- ifelse(pred.WS.glm.23 >= 0.5, 1, 0)

#################################


### Training and testing sets

index <- sample(1:nrow(team.data), nrow(team.data) / 2)
training1 <- team.data[index, ]
testing1 <- team.data[-index, ]



# Fit the model using the filtered data

modelWS <- glm(WS~.,data=training1[,-c(1)],family=binomial)

modelWSL <- lm(WS~.,data=team.data[,-c(1,8,45)])

summary(modelWS)


## Variables for testing world series predictor
var.WS <- c("SB","OPS", "SO.Batting","tSho","cSho")

### Correlation plot and distributions
ggpairs(training1[var.WS])
corr <- round(cor(training1[var.WS]),2)
ggcorrplot(corr,
           type = "lower",
           lab = TRUE, 
           lab_size = 5,
           title="correlation", 
           ggtheme=theme_bw)

## Fun visual analysis with pairs of data
ggplot(training1, aes(x = SB, y = OPS)) +
  geom_point(aes(color = factor(WS))) +
  scale_color_manual(values = c("0" = "magenta", "1" = "navy"))




### LDA model for WS
model.WS.lda <- lda(WS~ SB + OPS + SO.Batting + tSho ,data = training1)
#model.WS.lda <- lda(WS~ OPS ,data = training1)
model.WS.lda
pred.WS.lda <- predict(model.WS.lda,testing1,type = "response")

lda.roc <- roc(testing1$WS,pred.WS.lda$posterior[,2],
    plot = TRUE,
    legacy.axes = TRUE,
    xlab = "False Positive Rate",
    ylab = 'True Positive Rate',
    main = "LDA Model ROC Curve")
auc(lda.roc)

mean(pred.WS.lda$class == testing1$WS)




# Convert predicted probabilities to binary values (0 or 1)
#predictions.lda <- ifelse(as.numeric(pred.WS.lda$class) >= 0, 1, 0)

# Create a confusion matrix
confusion.matrix.lda <- table(Actual = testing1$WS, Predicted = pred.WS.lda$class)

print(confusion.matrix.lda)

# Prediction for 2023 season

pred.WS.lda.23 <- predict(model.WS.lda,team.data.23)
confusion.matrix.lda.23 <- table(Actual = team.data.23$WS, Predicted = pred.WS.lda.23$class)

print(confusion.matrix.lda.23)


### Logistic Model

model.WS.glm <- glm(WS~SB + OPS + SO.Batting + tSho,data = training1,family=binomial)
#model.WS.glm <- glm(WS~OPS ,data = training1,family=binomial)


summary(model.WS.glm)
pred.WS.glm <- predict(model.WS.glm,testing1,type = 'response')

rock <- roc(testing1$WS,pred.WS.glm,
            plot = TRUE,
            legacy.axes = TRUE,
            xlab = "False Positive Rate",
            ylab = 'True Positive Rate',
            main = "Logistic Model ROC Curve")
auc(rock)
# Convert predicted probabilities to binary values (0 or 1)
predictions.log <- ifelse(pred.WS.glm >= 0.5, 1, 0)

# Create a confusion matrix
confusion.matrix.log <- table(Actual = testing1$WS, Predicted = predictions.log)

# Print the confusion matrix
print(confusion.matrix.log)

#Predict the 2023 season
pred.WS.glm.23 <- predict(model.WS.glm,team.data.23,type = 'response')

#View(pred.WS.glm.23)
#View(team.data.23)

predictions.log.23 <- ifelse(pred.WS.glm.23 >= 0.5, 1, 0)
confusion.matrix.log.23 <- table(Actual = team.data.23$WS, Predicted = predictions.log.23)
print(confusion.matrix.log.23)


### KNN
cols.to.use <- c("SB" ,"OPS", "SO.Batting", "tSho","WS")
train.subset <- training1[,cols.to.use]
test.subset <- testing1[,cols.to.use]

# Fitting KNN model 
set.seed(2718)
# k = 1
classifier.knn <- knn(train = train.subset, 
                      test = test.subset, 
                      cl = train.subset$WS, 
                      k = 1) 
cm <- table(test.subset$WS, classifier.knn) 
cm

knn.roc <- roc(testing1$WS,pred.WS.lda$posterior[,2],
               plot = TRUE,
               legacy.axes = TRUE,
               xlab = "False Positive Rate",
               ylab = 'True Positive Rate',
               main = "LDA Model ROC Curve")
auc(lda.roc)


# k = 2
cols.to.use <- c("SB" ,"OPS", "SO.Batting", "tSho","WS")
train.subset <- training1[,cols.to.use]
test.subset <- testing1[,cols.to.use]


classifier.knn <- knn(train = train.subset, 
                      test = test.subset, 
                      cl = train.subset$WS, 
                      k = 2,prob=TRUE) 

cm <- table(test.subset$WS, classifier.knn) 
cm
#recall = TP / (TP + FN)
recall <- sum(test.subset$WS == 1 &classifier.knn ==1  )/ sum(test.subset$WS) 
print(paste('Recall =', recall)) 
#precision = TP / (TP + FP)
precision <- sum(test.subset$WS == 1 &classifier.knn ==1  )/ sum(classifier.knn==1) 
print(paste('Precision =', precision)) 
#accuracy = (TP + TN) / Total
misClassError <- mean(classifier.knn != test.subset$WS) 
print(paste('Accuracy =', 1-misClassError)) 

#Predict the 2023 season
cols.to.use <- c("SB" ,"OPS", "SO.Batting", "tSho","WS")
train.subset <- training1[,cols.to.use]
test.subset.23 <- team.data.23[,cols.to.use]


predict.knn.23 <- knn(train = train.subset, 
                      test = test.subset.23,
                      cl = train.subset$WS, 
                      k = 2,
                      prob = TRUE) 

cm <- table(team.data.23$WS, predict.knn.23) 
cm


#k = 3
classifier.knn <- knn(train = train.subset, 
                      test = test.subset, 
                      cl = train.subset$WS, 
                      k = 3) 

cm <- table(test.subset$WS, classifier.knn) 
cm

#recall = TP / (TP + FN)
recall <- sum(test.subset$WS == 1 &classifier.knn ==1  )/ sum(test.subset$WS) 
print(paste('Recall =', recall)) 
#precision = TP / (TP + FP)
precision <- sum(test.subset$WS == 1 &classifier.knn ==1  )/ sum(classifier.knn==1) 
print(paste('Precision =', precision)) 
#accuracy = (TP + TN) / Total
misClassError <- mean(classifier.knn != test.subset$WS) 
print(paste('Accuracy =', 1-misClassError)) 



# k = 4
classifier.knn <- knn(train = train.subset, 
                      test = test.subset, 
                      cl = test.subset$WS, 
                      k = 4) 


cm <- table(test.subset$WS, classifier.knn) 
cm

#recall = TP / (TP + FN)
recall <- sum(test.subset$WS == 1 &classifier.knn ==1  )/ sum(test.subset$WS) 
print(paste('Recall =', recall)) 
#precision = TP / (TP + FP)
precision <- sum(test.subset$WS == 1 &classifier.knn ==1  )/ sum(classifier.knn==1) 
print(paste('Precision =', precision)) 
#accuracy = (TP + TN) / Total
misClassError <- mean(classifier.knn != test.subset$WS) 
print(paste('Accuracy =', 1-misClassError)) 


######### LDA model for 3D visualization WS
train <- training1[,c(14, 21, 40, 67)]

View(train)

model.WS.lda.3D <- lda(WS~ SB + OPS  + tSho ,data = train)
#model.WS.lda <- lda(WS~ OPS ,data = training1)
model.WS.lda.3D
pred.WS.lda.3D <- predict(model.WS.lda.3D,testing1)

confusion.matrix.lda.3D <- table(Actual = testing1$WS, Predicted = pred.WS.lda.3D$class)
print(confusion.matrix.lda.3D)

write.table(train, "train.tsv", sep = "\t", row.names = FALSE)


##### 3D model of training data
SB.lda <- 0.008056699
OPS.lda <- 15.174308879
tSho.lda <- 0.205465030

groupmean<-(model.WS.lda.3D$prior%*%model.WS.lda.3D$means)
constant<-(groupmean%*%model.WS.lda.3D$scaling)
as.numeric(constant)

fig <- plot_ly(testing1, x = ~SB, y = ~OPS, z = ~tSho, color = ~WS, colors = c('#BF382A', '#0C4B8E'))
fig <- fig %>% add_markers()
fig <- fig %>% layout(scene = list(xaxis = list(title = 'SB'),
                                   yaxis = list(title = 'OPS'),
                                   zaxis = list(title = 'tSho')))

x.lda <- seq(0, 400, length.out = 100)
y.lda <- seq(0.5, 0.9, length.out = 100)
z.lda <- outer(x.lda, y.lda, function(x, y) ( (-SB.lda * x - OPS.lda * y + as.numeric(constant)) / tSho.lda))

fig <- fig %>% add_surface(x = x.lda, y = y.lda, z = z.lda,colors = "#0C4B8E")  # Adjust the colors as needed

fig





