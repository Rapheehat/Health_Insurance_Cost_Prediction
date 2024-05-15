# Title: Health Insurance Cost Prediction Using Linear Regression
# Author: Rafiat Olaide Ajibode
# Date: 11th January 2023


# load the required packages
library(dplyr)
library(ggplot2)
library(caret)
library(corrplot)
library(GGally)
library(tidyverse)
library(MLmetrics)  
library(auditor)
library(ggfortify)

# Load the Health Insurance Data
health_data <- read.csv("Health_insurance.csv", stringsAsFactors = TRUE)

# ============================== PRE-PROCESSING ================================

# dimensions of health insurance dataset
dim(health_data)

# Have a peek at our dataset
head(health_data, n=10) #shows the first six lines of the Health Insurance Data

# Structure of the dataset
str(health_data)

# list types for each columns in the dataset
sapply(health_data, class)

# check for missing values
colSums(is.na(health_data))

#checking for duplicates
anyDuplicated(health_data)

#delete 582nd row
health_data <- health_data[-582,]
dim(health_data)

# statistical summary of the dataset
summary(health_data)

# ========================= EXPLORTORY DATA ANALYSIS ===========================

#plotting categorical attributes

#bar plot for sex distribution
ggplot(health_data, aes(sex, fill = sex)) + 
  geom_bar() + xlab("Sex") + 
  ylab("Count") + ggtitle("Sex: Male or Female") + 
  scale_fill_manual(values = c("#B7D5D4", "#373E40")) +
  theme_classic()+ 
  geom_text(stat='count',
            aes(label=paste0(round(after_stat(prop*100), digits=1), "%"),group=1),
            vjust=-0.4,
            size=4)

#bar plot for smokers
ggplot(health_data, aes(smoker, fill = smoker)) +
  geom_bar() + xlab("Smokers") + 
  ylab("Count") + ggtitle("Smoker Yes Or No") + 
  scale_fill_manual(values = c("#373E40", "#B7D5D4")) +
  theme_classic()+ 
  geom_text(stat='count',
            aes(label=paste0(round(after_stat(prop*100), digits=1), "%"),group=1),
            vjust=-0.4,
            size=4)

#bar plot for regions
ggplot(health_data, aes(region, fill = region)) +
  geom_bar() + xlab("Regions") + 
  ylab("Count") + ggtitle(" Regions") + 
  scale_fill_manual(values = c("#8EAF9D", "#EF7674","#373E40", "#B7D5D4")) +
  theme_classic()+ 
  geom_text(stat='count',
            aes(label=paste0(round(after_stat(prop*100), digits=1), "%"),group=1),
            vjust=-0.4,
            size=4)

#plotting numerical attributes

# Histogram for each numerical Attribute
health_data %>%
  gather(Attributes, value, c(1,3,4,7)) %>%
  ggplot(aes(x=value, fill=Attributes)) +
  geom_histogram(colour="black", show.legend=FALSE, bins = 15) +
  facet_wrap(~Attributes, scales="free") +
  labs(x="Values", y="Frequency",
       title="Health Insurance Distribution (Numerical Attributes)") +
  scale_fill_manual(values = c("#8EAF9D", "#EF7674","#373E40", "#B7D5D4")) +
  theme_bw()

#Correlation between numerical attributes
figsize <- options(repr.plot.width=12, repr.plot.height=8)

age <- health_data %>%
  ggplot(aes(x=age, y=charges)) +
  geom_point()+
  labs(
    x = "Ages",
    y = "Charges($)",
    title = "Charges and Age Correlation"
  )

bmi <- health_data %>%
  ggplot(aes(x=bmi, y=charges)) +
  geom_point()+
  labs(
    x = "BMI",
    y = "Charges($)",
    title = "Charges and BMI Correlation"
  )

children <- health_data %>% 
  ggplot(aes(x=children, y=charges))+
  geom_point()+
  labs(
    x = "Children",
    y = "Charges($)",
    title = "Charges and Number of Children Correlation")
 

cowplot::plot_grid(age, bmi, children,labels="AUTO", ncol = 2, nrow = 2)


#Correlation between categorical attributes
figsize <- options(repr.plot.width=12, repr.plot.height=8)

smoker <- health_data %>%
  ggplot(aes(x=forcats::fct_reorder(smoker, charges, .fun=median, .desc=TRUE),
      y=charges,
      fill=smoker)) +
  geom_boxplot(show.legend = TRUE) +
  labs(x = "", y = "Charges($)",
       title = "Distribution of charges by Smokers") + 
  scale_x_discrete(
    labels = c("no" = "Non-smoker", "yes" = "Smoker"))

sex <- health_data %>%
  ggplot(aes(x=forcats::fct_reorder(sex, charges, .fun=median, .desc=TRUE),
      y=charges,
      fill=sex)) +
  geom_boxplot(show.legend = TRUE) +
  labs(x = "", y = "Charges($)",
       title = "Distribution of charges by Sex") +
  scale_fill_manual(values = c("#B7D5D4", "#373E40")) 

region <- health_data %>%
  ggplot(aes(x=forcats::fct_reorder(region, charges, .fun=median, .desc=TRUE),
      y=charges,
      fill=region)) +
  geom_boxplot(show.legend = TRUE) +
  labs(x = "", y = "Charges($)",
       title = "Distribution of charges by Region") +
  scale_fill_manual(values = c("#8EAF9D", "#EF7674","#373E40", "#B7D5D4")) +
  theme_bw()

cowplot::plot_grid(smoker, sex, region,labels="AUTO", ncol = 2, nrow = 2)

#Pairplot for numerical attributes

pairs(health_data[c("age", "bmi", "children", "charges")])

# ==================== PREPARING DATA FOR MACHINE LEARNING =====================

#converting to numeric
health_data$sex <- as.numeric(health_data$sex )
health_data$smoker <- as.numeric(health_data$smoker)
health_data$region <- as.numeric(health_data$region)
health_data

# correlation plot
ggpairs(health_data[1:7])

# correlation coefficient of the attributes
cor(health_data[1:7])

# correlation coefficient of the attributes by Charges
cor(health_data[1:7])[,"charges"] 

# summarize attribute distributions
summary(health_data[1:7])

#Correlation Plot
ggcorr(health_data[1:7], name = "correlation", label = TRUE) +
  labs(title = "Health Insurance Dataset", fontface='bold')

#The histogram for the response variable Charges shows that it is skewed.
# Taking the log of the variable normalizes it.

#Normalize distribution
charges <- health_data %>%
  ggplot(
    aes(x=charges)) +
  geom_histogram(
    binwidth = 2000, bins = 30,
    show.legend = FALSE,
    fill = "#666A86")+
  labs(
    x = "Charges($)",
    y = "Count",
    title = "Distribution of Charges (Before Normalization)"
  )

charges_log <- health_data %>%
  ggplot(
    aes(x=log(charges))) +
  geom_histogram(bins = 30,
    show.legend = FALSE,
    fill = "#1C2541")+
  labs(
    x = "log(Charges)",
    y = "Count",
    title = "Distribution of charges (After Normalization)"
  )

cowplot::plot_grid(
  charges, charges_log, labels="AUTO", ncol = 2, nrow = 1)

# ============================ DATA TRANSFORMATION ============================= 

# log10 transform of response variable 
health_data$charges_log <- log10(health_data$charges)
health_data

# Data splitting - 80:20
set.seed(334)
validationIndex <- createDataPartition(health_data$charges_log, p=0.80, list=FALSE)

# 20% of the data for test set
test <- health_data[-validationIndex,]

# 80% of the data for training set
training <- health_data[validationIndex,]

# ================ MODEL BUILDING - MULTIPLE LINEAR REGRESSION ================

model <- lm(formula = charges_log ~., data = training)

#statistical summary of the model
summary(model)

#use model to predict probability of default
pred <- predict(model, test)

#Evaluation metrics
r2 <- R2_Score(pred, test$charges_log)
mae <- MAE(pred, test$charges_log)
rmse <- RMSE(pred, test$charges_log)
mse <- MSE(pred, test$charges_log)

#Attributes of the test data
attributes(model)

#Model Performance
#plotting the predicted values vs the actual values
test$pred <- predict(model, newdata = test)
ggplot(test, aes(x = pred, y = charges_log)) + 
  geom_point(color = "dark blue", alpha = 3.7) + 
  geom_abline(color = "red") +
  ggtitle("Predicted vs. Actual values")

#plotting the residual histogram
test$residuals <- test$charges_log - test$pred
ggplot(test, aes(x = residuals)) + 
  geom_histogram(bins = 15, fill = "#666A86") +
  ggtitle("Residual Histogram")

#plotting the predicted values vs the residual values
test$residuals <- test$charges_log - test$pred

ggplot(data = test, aes(x = pred, y = residuals)) +
  geom_pointrange(aes(ymin = 0, ymax = residuals), color = "darkblue", alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = 4, color = "#F4442E") +
  ggtitle("Residuals vs. Predicted")

#regression diagnostics 
autoplot(model)


#plotting the REC curve
lm_audit <- audit(model, data = training, y = training$charges_log)

# validating the model with auditor
mr_lm <- model_residual(lm_audit)
plot_rec(mr_lm)







