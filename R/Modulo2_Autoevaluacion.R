#cargar paquetes

library(tidyverse)
library(ggplot2)
library(data.table)

#database que se va a usar
diamonds

str(diamonds)
DT<-data.table(diamonds)
DT[,order(price),by=carat]

#Un data.frame ordenado por max(price) en funci�n de carat.
DT%>% group_by(carat)  %>%  
  arrange(
    carat, 
    desc(price)
  )

DT[,max(price),by=carat]

#Un data.frame ordenado por max(price) en funci�n de color.
DT %>% group_by(color)  %>%  
  arrange(
    color, 
    desc(price)
  )
DT[,max(price),by=color]
#Un data.frame ordenado por max(price) en funci�n de clarity.
DT %>% group_by(clarity)  %>%  
  arrange(
    clarity, 
    desc(price)
  )
DT[,max(price),by=clarity]

#Un data.frame ordenado por max(price) en funci�n de carat y color.
DT %>% group_by(carat,color)  %>%  
  arrange(carat,
    color, 
    desc(price)
  )
DT[,max(price),by=c('carat','color')]

#Un data.frame ordenado por max(price) en funci�n de carat y cut.
DT %>% group_by(carat,cut)  %>%  
  arrange(carat,
          cut, 
          desc(price)
  )
DT[,max(price),by=c('carat','cut')]

#Un data.frame ordenado por max(price) en funci�n de carat y clarity.

DT %>% group_by(carat,clarity)  %>%  
  arrange(carat,
          clarity, 
          desc(price)
  )
DT[,max(price),by=c('carat','clarity')]


##4. Realizar un gr�fico ggplot, de puntos (geom_point()) que:
#permita ver la relaci�n entre carat y price en un solo gr�fico.
)
p<-ggplot(DT,aes(carat,price))+ 
  geom_point()+
  theme_bw()+
  labs(title="Carat vs. price", x="carat", y="price")
p
#modificar el gr�fico anterior asociando a cada punto el color en funci�n de cut.
p+geom_point(aes(color=cut))
  
#modificar el gr�fico anterior de forma condicionada facet_grid() con cut.
p+geom_point(aes(color=cut))+facet_grid(rows = vars(cut))
#modificar el gr�fico anterior de forma condicionada con dos niveles cut y color.
p+geom_point(aes(color=cut))+facet_grid(rows = vars(color), cols = vars(cut))


##5Realizar un gr�fico ggplot, de tipo caja (geom_boxplot()) que:
#muestre la variaci�n de price en funci�n de cut.
bp<-ggplot(DT)+ 
  geom_boxplot(aes(cut,price))+
  theme_bw()+
  labs(title="Cut vs. price", x="cut", y="price")
bp
#y otro que muestre la variaci�n de price en funci�n de color.
bp2<-ggplot(DT)+ 
  geom_boxplot(aes(color,price))+
  theme_bw()+
  labs(title="Color vs. price", x="color", y="price")
bp2


##6 Del an�lisis del gr�fico de puntos:
#�Es lineal la relaci�n entre price con respecto a cut y color.
p2<-ggplot(DT,aes(color,price))+ 
  geom_point()+
  theme_bw()+
  labs(title="Carat vs. price", x="carat", y="price")
p2
p3<-ggplot(DT,aes(cut,price))+ 
  geom_point()+
  theme_bw()+
  labs(title="Carat vs. price", x="carat", y="price")
p3
#�Dir�as que hay outliers en alguna de las combinaciones?.
#Para describir matem�ticamente la relaci�n entre price y las variables cut y color, �qu� tipo de modelo aplicar�as entonces?.