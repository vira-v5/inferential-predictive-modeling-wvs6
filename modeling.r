### Group project 13 
### Created:  01/03/2020
### Modified: 03/04/2020


### Preliminaries

# Clear the environment
rm(list = ls(all = TRUE))

# Setting the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Source codes
source("studentFunctions.R")
source("miPredictionRoutines.R")

# Set random number seeds
set.seed(173649)

# Libraries
library(mice)
library(mitools)
library(MLmetrics)
library(miceadds)
library(naniar)
library(stargazer)
library(tidyr)
library(purrr)
library(ggplot2)

#### Data Preparation

# Load the data
dataDir  <- "../data/"
fileName <- "wvs_data.rds"

df <- readRDS(paste0(dataDir, fileName))

# Selection of variables
myvars <- c("V5", "V10", "V11", "V23", "V55", "V56", "V59", "V62", 
"V67", "V71", "V96", "V98", "V101", "V131", "V145", "V147")

# New data frame with the selected variables
newdf <- df[ ,myvars]


### Data cleaning

# Reorder the scale of V5, 10, 11, and 71 for easier interpretation

old = c(1,2,3,4,-1,-2,-3,-4,-5)
new = c(4,3,2,1,-1,-2,-3,-4,-5)

newdf$V5 <- new[match(newdf$V5,old)]
newdf$V10 <- new[match(newdf$V10,old)]
newdf$V11 <- new[match(newdf$V11,old)]

old1 = c(1,2,3,4,5,6,-1,-2,-3,-4,-5)
new1 = c(6,5,4,3,2,1,-1,-2,-3,-4,-5)

newdf$V71 <- new1[match(newdf$V71,old1)]

# Converting missing values
newdf[ newdf <= 0 ] <- NA

# Some part of EDA

# Summarize the data
summary(newdf)
stargazer(newdf, type="text", title="Summary Statistics", median=TRUE)

# Histogram of the variables on the dataset
dfhist <- newdf %>%
            keep(is.numeric) %>% 
            gather() %>% 
            ggplot(aes(value)) +
            facet_wrap(~ key, scales = "free") +
            geom_histogram()
dfhist

# Checking the proportion of missing value for selected variables
pm <- colMeans(is.na(newdf))
pm
gg_miss_var(newdf, show_pct = TRUE)


### Univariate outlier analysis

# Find potential outliers in all numeric columns
bpOut <- lapply(newdf[ , ], bpOutliers)
bpOut

# Turn possible and probabale outliers into NA and treat it together with missing values
newdf$V5[bpOut$V5$possible] <- NA
newdf$V10[bpOut$V10$possible] <- NA

# Checking the proportion of missing value for selected variables
pm2 <- colMeans(is.na(newdf))
pm2
gg_miss_var(newdf, show_pct = TRUE)

# Convert categorical variable into factor
categorical <- c("V62", "V147")
newdf[,categorical] <- lapply(newdf[categorical], factor)

# Checking covariance coverage
cc <- md.pairs(newdf)$rr / nrow(newdf)
cc

# Checking whether there are covariance coverages that are less than 0.75
sum(cc[lower.tri(cc)] < 0.75)

# Compute the missing data patterns
pat <- md.pattern(newdf, plot = FALSE)
pat


### Multiple imputation

# Define method vector
meth        <- rep("pmm", ncol(newdf))
names(meth) <- colnames(newdf)
meth[categorical] <- "polr"

# Use mice::quickpred to generate a predictor matrix
predMat <- quickpred(newdf, mincor = 0.05)
predMat

# Impute missing values using the method vector 
miceOut <- mice(newdf, m = 20, maxit = 10, predictorMatrix = predMat, method = meth, seed = 173649)

# Create list of multiply imputed datasets
impList <- mice::complete(miceOut, "all")

# Checking whether all missing values imputed
pm3 <- lapply(impList, function(x) colMeans(is.na(x)))
pm3

## Convergence Checks

# Create traceplots of imputed variables' means and SDs
plot(miceOut)

# Sanity check the imputations by plotting observed vs. imputed densities
densityplot(miceOut)


### Multivariate outlier analysis

# Dropping categorical variable for multivariate analysis with Mahalanobis distance
noCat <- lapply(impList, function(x) x[!(names(x) %in% c("V62", "V147"))])

# Using Mahalanobis distance to detect multivariate outliers
mdOut <- lapply(noCat, mdOutliers, critProb = 0.99, statType = "mcd", ratio = 0.75, seed = 173649)
mdOut

# Count the number of times each observation is flagged as an outlier:
mdCounts <- table(unlist(mdOut))

# Based on the frequency, we decided to remove all the observation that voted by at least 
# 50% of the results of Multivariate outlier analysis

thresh <- ceiling(miceOut$m / 2)
outs <- as.numeric(names(mdCounts[mdCounts >= thresh]))

# Exclude outlying observations from mids object
miceOutFin <- subset_datlist(datlist = miceOut,
                             subset  = setdiff(1 : nrow(newdf), outs),
                             toclass = "mids")

# Create list of multiply imputed datasets with multivariate outliers removed
impListFin <- mice::complete(miceOutFin, "all")

#-----------------------------------------------------------------------------------------------------------#

#### Inferential Modelling


## Simple linear model

fit_s <- with(miceOutFin, lm(V23 ~ V131)) 
est_s <- pool(fit_s)
summary(est_s)
est_s

## Multiple regression model
fit_m <- with(miceOutFin, lm(V23 ~ V131 + V145))
est_m <- pool(fit_m)
summary(est_m)

## Compute the pooled R^2
pool.r.squared(fit_s)
pool.r.squared(fit_m)

## Compare models
summary(D1(fit_m, fit_s))

# Compute increase in R^2
pool.r.squared(fit_m)[1] - pool.r.squared(fit_s)[1] 

# Do an F-test for the increase in R^2
fTest <- pool.compare(fit_m, fit_s)

fTest$Dm     # Test statistic
fTest$pvalue # P-Value

#-----------------------------------------------------------------------------------------------------------#

#### Predictive modelling

### Prediction
  
## Split the multiply imputed datasets into training and testing sets:
n <- nrow(impListFin[[1]])

index <- sample(
  c(rep("train", 10000), rep("test", n - 10000))
)

impList2 <- splitImps(imps = impListFin, index = index)

## Train a model on each multiply imputed training set:
fits <- lapply(impList2$train, function(x) lm(V59 ~ V5 + V10 + V11 + V56 + V147, data = x))
# every dataset (20 of them) are divided into 2 parts - test and train. 

## Generate imputation-specific predictions:
preds0 <- predictMi(fits = fits, newData = impList2$test, pooled = FALSE)
preds0
# predicted values per observation in each set. 


###--------------------------------------------------------------------------###

### MI-Based Prediction Cross-Validation ###

### Split-Sample Cross-Validation:

## Split the multiply imputed data into training, validation, and testing sets:
n <- nrow(impListFin[[1]])
index <- sample(
  c(rep("train", 7000), rep("valid", 3000), rep("test", n - 10000))
)

impListPred <- splitImps(imps = impListFin, index = index)

## Define some models to compare:
mods <- c("V59 ~ V5 + V10 + V11 + V56 ",
          "V59 ~ V55 + V67 + V71 + V96 + V98 + V101",
          "V59 ~ V55  + V67 + V71 ",
          "V59 ~ V71 + V96 + V98 +V101",
          'V59 ~ V5 + V10 + V11 + V56 * V62')



## Merge the MI training a validations sets:
index3   <- gsub(pattern = "valid", replacement = "train", x = index)
impList4 <- splitImps(impListFin, index3)


### K-Fold Cross-Validation:

## Conduct 10-fold cross-validation in each multiply imputed dataset:
tmp <- sapply(impList4$train, cv.lm, K = 10, models = mods, seed = 173649)

## Aggregate the MI-based CVEs:
cve <- rowMeans(tmp)
cve

## Refit the winning model and compute test-set MSEs:
fits <- lapply(X   = impList4$train,
               FUN = function(x, mod) lm(mod, data = x),
               mod = mods[which.min(cve)])
mse <- mseMi(fits = fits, newData = impList4$test)

mse
