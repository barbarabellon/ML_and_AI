#Autoevaluación módulo 5

#Hay que precedir el numero de bicicletas que se alquilarán (cnt) a través de varios métodos
#como se tiene que predecir cuantas es un problema de regresión, no de clasificación

#Importar librerías

rm(list = ls()) 
library(caret)
library(corrplot)
library(gmodels)
library(ggplot2)
library(doParallel)
library(ROCR)
library(party)
library(rpart)
library(randomForest)
library(C50)
library(CHAID)
library(gbm)
library(imputeTS)
library(pROC)
library(psych)
library(rpart.plot)
library(plyr)
library(questionr)
library(mice)
library(tidyverse)

#working directory
path<-"/home/barbara/Dropbox/Data Mierdas/Modulos/5-MINERIA DE DATOS I/2-Taera/autoevaluacion"
setwd(path)

#lectura de datos
archivos<-dir(path = path, pattern = "*csv", all.files = T,
    full.names = T)

datos <- read.csv(archivos)


#Exploramos la base de datos
str(datos)


## Cuento el número de valores diferentes para las numéricas
sapply(Filter(is.numeric, datos),function(x) length(unique(x)))

#The variables with less than 10 different values could be converted to factors, but as they are already dummy variable
datos[,c(3:10)] <- lapply(datos[,c(3:10)], factor)
summary(datos)
#vemos que tenemos unos cuantos NAs en las variables que tienen que ver con las condiciones climáticas, 
#se podría utilizar como imputación la media o la mediana para estos casos, pero se corre el riesgo de asignar 
#valores de verano a missing si estos ocurren en invierno. También podrían ignorarse o eliminarse ya que no son muchos 
#datos 
clusterCPU <- makePSOCKcluster(detectCores()-1)
registerDoParallel(clusterCPU)


md.pattern(datos,plot=T,rotate.names = T)

##tempData <- mice(datos, m=5, maxit=50,meth='mean',seed='500')
# imputed <- rfImpute(cnt ~ temp+windspeed+hum+weathersit+season, datos)

## MISSINGS
#Busco si existe algún patrón en los missings, que me pueda ayudar a entenderlos
corrplot(cor(is.na(datos[colnames(datos)[colSums(is.na(datos))>0]])),method = "circle",type = "upper") #No se aprecia ningún patrón

# #Proporción de missings por variable y observación
# datos$prop_missing<-apply(is.na(datos),1,mean) # Por observación
# summary(datos$prop_missing)
# (prop_missingsVars<-apply(is.na(input),2,mean)) # Por variable

#graficos

g1 <- ggplot(datos, aes(season)) +
  geom_bar()

g2 <- ggplot(datos, aes(hr)) +
  geom_bar()

g3 <-  ggplot(datos, aes(weathersit)) +
  geom_bar()

g4 <- ggplot(datos, aes(workingday)) +
  geom_bar()


grid.arrange(g1, g2, g3, g4, nrow = 2)

datos %>%
  gather(-cnt, key = "var", value = "value") %>%
  ggplot(aes(x = value, y = cnt)) +
  geom_point() +
  facet_wrap(~ var, scales = "free") +
  theme_bw()

plot(cnt~,data=datos)

#correlaciones entre variables
correl <- cor(datos[,-c(2:10)])

correl


corrplot(cor(datos[-c(2:10)]))


#se observa que los datos estan mas o menos centrados en la variable temperatura, por lo que usamos la media
# #en las otras dos vamos a usar la moda
# datos[is.na(datos$temp), "temp"] <- round(mean(datos$temp,na.rm=T),0)
# 
# moda <- function(v) {
#   uniqv <- unique(v)
#   uniqv[which.max(tabulate(match(v, uniqv)))]
# }
# datos[is.na(datos$hum), "hum"] <- moda(datos$hum)
# datos[is.na(datos$windspeed), "windspeed"] <- moda(datos$windspeed)

datos<-na_interpolation(datos, option = "linear", na.identifier = NA)

summary(datos)
summary(imputacion)
mean(datos$temp)
mean(imputacion$temp)

md.pattern(datos)

#ahora ya no tenemos ningun missing. 

#en las unicas variables que vamos a poder tener valores atipicos es en estas, ya que las
#demás se mueven en valores fijos

corrplot(cor(datos[-c(2:10)],method = "pearson"),method = "circle",type = "upper")
cor(datos[,c(11:12)],method = "pearson")


#outliers
library(mvoutlier)
aq.plot(datos[,-c(1:10)], alpha=0.1)
res<-chisq.plot(datos[,-c(1:10)])
res <- mvoutlier.CoDa(datos[-c(1:10)])
#pues no analizo outliers

##Seleccion de variables
library(glmnet)

y <- as.double(as.matrix(datos[, 16]))
x<-model.matrix(y~., data=datos[,-2])[,-1]
set.seed(1712)
cv.lasso <- cv.glmnet(x,y,nfolds=5)
plot(cv.lasso)
betas<-coef(cv.lasso, s=cv.lasso$lambda.1se)
row.names(betas)[which(as.matrix(betas)!=0)]
betas[which(as.matrix(betas)!=0)]

##no entiendo nada


pca_data<-datos[,c(11:13)]
pca<-princomp(pca_data,rank=3)
plot(pca)
summary(pca)
#Gŕafica de variables
library(factoextra)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# Results for Variables
res.var <- get_pca_var(pca)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 



var<-get_pca_var(pca)
ind<-get_pca_ind(pca)

print(var$contrib)
print(var$cos2)

corrplot(var$cos2)

#Porcentaje de variabilidad explicada por las tres CP
fviz_cos2(pca,choice="var",axes=1:3)

#porcentaje de la varianza de cada componente debido a cada variable
corrplot(var$contrib, is.corr=T)
print(var$contrib, digit=2)


#Contribución de las variables a cada Componente
fviz_contrib(pca,choice="var",axes=1,top=10)
fviz_contrib(pca,choice="var",axes=2,top=10)
fviz_contrib(pca,choice="var",axes=3,top=10)




pca_data2<-datos[,c(14:16)]
pca2<-princomp(pca_data2,rank=3)
plot(pca2)
summary(pca2)
#Gŕafica de variables
library(factoextra)
fviz_pca_var(pca2,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# Results for Variables
res.var <- get_pca_var(pca2)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 



var<-get_pca_var(pca2)
ind<-get_pca_ind(pca2)

print(var$contrib)
print(var$cos2)

corrplot(var$cos2)

#Porcentaje de variabilidad explicada por las tres CP
fviz_cos2(pca2,choice="var",axes=1:3)

#porcentaje de la varianza de cada componente debido a cada variable
corrplot(var$contrib, is.corr=T)
print(var$contrib, digit=2)


#Contribución de las variables a cada Componente
fviz_contrib(pca,choice="var",axes=1,top=10)
fviz_contrib(pca,choice="var",axes=2,top=10)
fviz_contrib(pca,choice="var",axes=3,top=10)

#Se observa que las variables atemp y temp estan muy correlacionadas
cor(datos[,c(11:12)],method = "pearson")
#casi un 100% de correlacion, por tanto una de ellas podría eliminarse sin necesidad de hacer componentes principales

#factores

cortest.bartlett(datos[,c(11:13)])
datos2<-datos[,c(11:13)]
KMO(datos2)


#segun el test de esfericidad se pueden llevar a cabo factores

correl<-cor(datos2,use="pairwise.complete.obs")
correl
fa.parallel(correl, n.obs = length(datos), fm = "gls")
datos$atemp<-NULL
#nos sugiere que  no hagaos analsis factorial
datos.fa <- fa(correl, n.obs = length(datos), fm = "gls", nfactors = 10, rotate ="varimax")
print(datos.fa)

datos.fa2<-factanal(datos2, factors=3, rotation="varimax",scores = "Bartlett")
Belts.fa
#este no funciona

library(rela)
res <- paf(as.matrix(datos2))
summary(res)
barplot(res$Eigenvalues[,1]) # Primera columna de valores propios.

resv <- varimax(res$Factor.Loadings) # La rotación Varimax es posible más tarde.
print(resv)

#no sacamos factores porque no entiendo una mierda :)

# Generamos el 80% de la base de datos para muestra de entrenamiento
set.seed(7)
datos2<-datos[,c(3:13,16)]
validation_index <- createDataPartition(datos2$cnt, p=0.80, list=FALSE)

# El 20% par validadaci?n

validacion <- datos2[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- datos2[validation_index,]




save_model<-function(modelo){
  name<-paste(path,"/",deparse(substitute(modelo)),".Rdata",sep="")
  print(name)
  saveRDS(modelo,file= name)
}


# EVALUACI?N DE MODELOS

# EJECUCI?N DEL ALGORITMO CON  10-fold cross validation

control <- trainControl(method="repeatedcv", number=10, repeats=3)

metric <- "RMSE"

fiveStats = function(...) c (twoClassSummary(...), defaultSummary(...))
control <- trainControl(method = "repeatedcv", 
                        number = 3,
                        repeats = 1, 
                        summaryFunction = defaultSummary,
                        returnResamp = "final",
                        allowParallel = TRUE)

metric <-"RMSE"

# lm (MODELO LINEAL GENERAL)

set.seed(7)

fit.lm <- train(cnt~., data=dataset, method="lm", metric=metric, trControl=control)
save_model(fit.lm)
predic_lm <- predict(fit.lm , newdata=validacion)
plot(fit.lm)
rmse <- RMSE(predic_lm, validacion$cnt)
r2 <- R2( predic_lm, validacion$cnt)
print(paste(" En la Regresi?n Lineal el valor del RMSE es:", round(rmse,2), "y el del R2:", round(r2,2)))

lmImp <- varImp(fit.lm )

plot(lmImp)


#LASSO

set.seed(7)


# Control de la T?cnica de Remuestreo: 100 muestras bootstrap

lasso.ctrl = trainControl ( method = "boot" , number = 100)

lassoGrid = expand.grid ( .alpha = 1 , .lambda = seq ( .0001 , .1, length=20 ))

fit.lasso <- train(cnt~., data=dataset, method="glmnet", tuneGrid=lassoGrid, metric=metric, preProc=c("center", "scale"), trControl=lasso.ctrl)
save_model(fit.lasso)
plot(fit.lasso, metric="RMSE")
plot(fit.lasso, metric="Rsquared")
plot(fit.lasso, metric="ROC")
# CART

set.seed(7)
grid_cart <- expand.grid(.cp=seq(0, 0.05,0.005))
fit.cart <- train(cnt~., data=dataset, method="rpart", metric=metric, tuneGrid=grid_cart, trControl=control)
save_model(fit.cart)
plot(fit.cart)
plot(varImp(fit.cart))

predic_cart <- predict(fit.cart , newdata=validacion)

rmse <- RMSE(predic_cart, validacion$cnt)
r2 <- R2( predic_cart, validacion$cnt)
print(paste(" En la Regresi?n Lineal el valor del RMSE es:", round(rmse,2), "y el del R2:", round(r2,2)))

grid_cart <- expand.grid(.cp=0)
fit.cart2 <- train(cnt~., data=dataset, method="rpart", metric=metric, tuneGrid=grid_cart, trControl=control)
save_model(fit.cart2)
# rpart.plot(fit.cart2$finalModel)

stopCluster(clusterCPU)
#randomforest 

##seguir con esto
rfGrid <-  expand.grid(mtry = seq(120,160,10))
control_rf <- trainControl("oob")
clusterCPU <- makePSOCKcluster(detectCores()-1)
registerDoParallel(clusterCPU)
rf2 <- train(cnt ~ ., data = datos2, 
             method = "rf", 
             metric = metric, 
             trControl = control,
             tuneGrid = rfGrid)
save_model(rf2)
# stopCluster(clusterCPU)
rf2
plot(rf2)


predic_rf <- predict(rf2 , newdata=validacion)

rmse <- RMSE(predic_rf, validacion$cnt)
r2 <- R2( predic_rf, validacion$cnt)

# KNN


set.seed(7)
grid_knn <- expand.grid(.k = c(50:180))

fit.knn <- train(cnt~., data=dataset, method="knn", metric=metric,tuneGrid=grid_knn, trControl=control)
save_model(fit.knn)
fit.knn
plot(fit.knn)


predic_knn <- predict(fit.knn , newdata=validacion)

rmse <- RMSE(predic_knn, validacion$cnt)
r2 <- R2( predic_knn, validacion$cnt)



#RED NEURONAL
set.seed(7)
#system.time()

grid_mlp <- expand.grid(.size=c(1:10,c(5,2)))

fit.mlp <- train(cnt~., data=dataset, method="mlp", tuneGrid=grid_mlp, trControl=control)
save_model(fit.mlp)
plot(fit.mlp)


grid_neural <- expand.grid(layer1=10,layer2=8,layer3=5)
# control<-trainControl ( method = "boot" , number = 100)


library(neuralnet)

nn <- neuralnet(cnt~.,data=dataset, hidden=5, linear.output=F, threshold=0.01)
save_model(nn)
plot(nn)
nn$result.matrix

nnetGrid <-  expand.grid(size = seq(from = 1, to = 10, by = 1),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))
fit.nnet <- train(cnt~., data=dataset, tuneGrid=nnetGrid,method="nnet",trControl=control)

save_model(fit.nnet)
# SVM

set.seed(7)
grid_svm <- expand.grid(.C = c(0.05, 1, 5,10, 50, 100), .sigma = c(0.001,0.005,0.01,0.05,0.1))
# grid_svm <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))

fit.svm <- train(cnt~., data=dataset, method="svmRadial",tuneGrid=grid_svm, preProc=c("center", "scale"), trControl=control)
save_model(fit.svm)
print(fit.svm)
plot(fit.svm)

# # RANDOM FOREST
# 
# grid_rf <- expand.grid(.mtry=c(5, 10, 15, 25, 50, 75, 100, 200))
# set.seed(7)
# system.time(fit.rf <- train(medv~., data=dataset, method="rf", metric=metric, tuneGrid=grid_rf, preProc=c("center", "scale"), trControl=control))
# 
# # M5
# set.seed(7)
# fit.M5 <- train(medv~., data=dataset, method="M5", metric=metric, preProc=c("center", "scale"), trControl=control)

# GBM Stochastic Gradient Boosting

set.seed(7)

fit.gbm <- train(cnt~., data=dataset, method="gbm",  trControl=control, verbose=FALSE)
save_model(fit.gbm)
# GBM eXtreme Gradient Boosting (xgbTree, xgbLinear)

set.seed(7)

dataset <- as.matrix(dataset)
stopCluster(clusterCPU)
registerDoParallel(cores=1)
fit.xgbTree <- train(cnt~., data=dataset, method="xgbTree", trControl=control, verbose=FALSE)
save_model(fit.xgbTree)
predic_xgbTree <- predict(fit.xgbTree , newdata=validacion)

rmse <- RMSE(predic_xgbTree, validacion$cnt)
r2 <- R2(predic_xgbTree, validacion$cnt)

print(paste(" En XGBTREE el valor del RMSE es:", round(rmse,2), "y el del R2:", round(r2,2)))

xgbTreeImp <- varImp(fit.xgbTree)

plot(xgbTreeImp)


#bagging
#Bagging

# Para reproducir siempre igual la parte aleatoria
set.seed(9)
control_oob <- trainControl(method = "oob")
registerDoParallel(clusterCPU)
tune_grid = expand.grid(mtry=seq(50,200,50))
modelo_bagging <- train( cnt~., datos2, method = "rf",metric=metric, trControl = control_oob, tuneGrid=tune_grid)
save_model(modelo_bagging)
var_imp_bagging <- varImp(modelo_bagging)

plot(var_imp_bagging)

#validacion

prediccion_prob_bagging <- predict(modelo_bagging,transfusiones_test,type = "prob")


prediccion_bagging <- predict(modelo_bagging,transfusiones_test)
prediccion_bagging <- factor(prediccion_bagging)
