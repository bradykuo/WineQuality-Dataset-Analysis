# Load required libraries for wine quality classification

# tidyverse: Collection of R packages for data manipulation and visualization
# Includes ggplot2, dplyr, tidyr, readr, etc.
library(tidyverse)  

# caret: Framework for training and plotting machine learning models
# Used here for cross-validation (createFolds function)
library(caret)      

# MASS: Modern Applied Statistics library
# Provides LDA and QDA implementations
library(MASS)       

# randomForest: Implementation of random forest algorithm
# Used for ensemble learning classification
library(randomForest)

# class: Contains k-nearest neighbor (KNN) implementation
# Used for instance-based learning
library(class)      

# nnet: Neural network package
# Provides multinomial logistic regression for multi-class classification
library(nnet)

# Read and verify the red wine data
red_wine <- read.csv("winequality-red.csv", header = TRUE)

# Print structure and check for constant variables
str(red_wine)
summary(red_wine)

# Perform cross-validation with error handling
perform_cv <- function(data, target, model_type, k = 5) {
  set.seed(123)
  tryCatch({
    folds <- createFolds(data[[target]], k = k, list = TRUE)
    accuracies <- numeric(k)
    
    for(i in 1:k) {
      test_indices <- folds[[i]]
      train_data <- data[-test_indices, ]
      test_data <- data[test_indices, ]
      
      # Remove type column for quality prediction
      if(target == "quality_category") {
        train_data <- train_data[, !names(train_data) %in% c("type")]
        test_data <- test_data[, !names(test_data) %in% c("type")]
      }
      
      if(model_type == "logistic") {
        # For multiclass, use multinomial regression
        if(length(unique(data[[target]])) > 2) {
          model <- multinom(as.formula(paste(target, "~ . -quality_category -quality")), 
                            data = train_data)
        } else {
          model <- glm(as.formula(paste(target, "~ . -quality_category -quality")), 
                       data = train_data, family = "binomial")
        }
        pred <- predict(model, test_data)
        
      } else if(model_type == "lda") {
        # Check for constant variables
        var_sd <- apply(train_data[, !names(train_data) %in% c(target, "quality_category", "quality")], 2, sd)
        non_constant_vars <- names(var_sd)[var_sd > 0]
        
        formula <- as.formula(paste(target, "~", paste(non_constant_vars, collapse = " + ")))
        model <- lda(formula, data = train_data)
        pred <- predict(model, test_data)$class
        
      } else if(model_type == "qda") {
        # Similar check for QDA
        var_sd <- apply(train_data[, !names(train_data) %in% c(target, "quality_category", "quality")], 2, sd)
        non_constant_vars <- names(var_sd)[var_sd > 0]
        
        formula <- as.formula(paste(target, "~", paste(non_constant_vars, collapse = " + ")))
        model <- qda(formula, data = train_data)
        pred <- predict(model, test_data)$class
        
      } else if(model_type == "knn") {
        predictors <- names(train_data)[!names(train_data) %in% c(target, "quality_category", "quality")]
        train_scaled <- scale(train_data[, predictors])
        test_scaled <- scale(test_data[, predictors])
        pred <- knn(train_scaled, test_scaled, train_data[[target]], k = 5)
        
      } else if(model_type == "rf") {
        model <- randomForest(as.formula(paste(target, "~ . -quality_category -quality")), 
                              data = train_data)
        pred <- predict(model, test_data)
      }
      
      actual <- test_data[[target]]
      accuracies[i] <- mean(pred == actual)
    }
    
    return(mean(accuracies))
  }, error = function(e) {
    message("Error in model ", model_type, ": ", e$message)
    return(NA)
  })
}

# Add quality categories
red_wine$quality_category <- cut(red_wine$quality,
                                 breaks = c(0, 5, 6, 10),
                                 labels = c("Low", "Median", "High"))

# Ensure quality_category is a factor
red_wine$quality_category <- factor(red_wine$quality_category)

# Run models
models <- c("logistic", "lda", "qda", "knn", "rf")

# Multi-class Classification for red wine
message("Starting red wine classification...")

red_results <- sapply(models, function(m) {
  message("Running model: ", m)
  perform_cv(red_wine, "quality_category", m)
})

names(red_results) <- c("Logistic Regression", "LDA", "QDA", "KNN", "Random Forest")

# Print results
cat("\nRed Wine Quality Classification Results:\n")
print(red_results)

# Create visualization for available results
results_df <- data.frame(
  Model = names(red_results)[!is.na(red_results)],
  Accuracy = red_results[!is.na(red_results)]
)

# Plot results with accuracy values
ggplot(results_df, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "darkred") +
  geom_text(aes(label = sprintf("%.3f", Accuracy)), 
            vjust = -0.5,  # Position text above bars
            color = "black",
            size = 4) +  # Text size
  theme_minimal() +
  labs(title = "Red Wine Quality Classification Model Performance",
       x = "Model Type",
       y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(limits = c(0, max(results_df$Accuracy) * 1.1))  # Extend y-axis for labels
