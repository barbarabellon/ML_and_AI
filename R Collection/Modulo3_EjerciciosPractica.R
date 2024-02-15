#Repaso módulo 3

##Instale y active la librería Rpart, acceda a la base de datos car90 y muestre su estructura
install.packages("rpart")
library(EnvStats)
library(rpart)
data("car90")
datos=car90
str(car90)

## Estime un modelo lineal simple, entre el precio de los coches (Pricei) y los caballos de vapor (HPi):

mod1=lm(Price~HP,data=car90)
summary(mod1)

## Realice una predicción por intervalo para un coche de 100 caballos de vapor.

newdatos = data.frame(HP=100)
predict(mod1, newdatos, interval="prediction", type="response")


## 4.- Realice el contraste de hipotesis de significación individual para ??1, mostrando:

#El valor crítico para un ??=0.05 (utilice la función qt y consulte su ayuda para obtener el valor crítico)

#p-valor

#Intervalo de confianza

t_tablas=qt(0.975,mod1$df.residual)

pvalor=summary(mod1)$coefficients[2,4]

estimadorb1=summary(mod1)$coefficients[2,1]

estimador_sd_b1=summary(mod1)$coefficients[2,2]

IC=c(estimadorb1-t_tablas*estimador_sd_b1,estimadorb1+t_tablas*estimador_sd_b1)

##Estime el modelo:Pricei=??0+??1HPi+??2Mileagei+??3Tanki+??4Widthi+??5Reat.Hdi+ui siendo:

mod2=lm(Price~HP+Mileage+Tank+Width+Rear.Hd,data=car90)

summary(mod2)

##modelo forward

modf1=lm(Price~HP,car90)
summary(modf1)

modf2=lm(Price~HP+Mileage,car90)
summary(modf2)

modf3=lm(Price~HP+Mileage+Tank,car90)
summary(modf3)

modf4=lm(Price~HP+Mileage+Tank+Width,car90)
summary(modf4)

modf5=lm(Price~HP+Mileage+Tank+Width+Rear.Hd,car90)
summary(modf5)

library(leaps)

regfit_fw=regsubsets(Price~HP+Mileage+Tank+Width+Rear.Hd,data=car90,method="forward")
summary(regfit_fw)


##7 El data set CO2 contiene información sobre la absorción de CO2 de seis plantas de Quebec y seis plantas de Mississippi. 
#Se midió en varios niveles de concentración de CO2 ambiental. La mitad de las plantas de cada tipo se refrigeraron durante la 
#noche antes de llevar a cabo el experimento. Las variables registradas son las siguientes:

#Con estos datos, realice un modelo ANOVA con el objeto de predecir la tasa de absorción de CO2
# en función del tipo de planta y el tratamiento. 
#¿Es significativo el efecto interacción entre los dos niveles de los factores?

