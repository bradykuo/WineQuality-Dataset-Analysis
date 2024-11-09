# Load required libraries
library(tidyverse)    
library(caret)        
library(MASS)         
library(class)        
library(randomForest) 
library(gridExtra)    

# Read the wine data
wine_data <- read.csv("winequality.csv", stringsAsFactors = TRUE)  # Added stringsAsFactors

# Convert type to factor explicitly
wine_data$type <- factor(wine_data$type)

# Fix column names - remove any spaces and special characters
names(wine_data) <- make.names(names(wine_data))

# Data composition visualization
wine_distribution <- ggplot(wine_data, aes(x = type)) +
  geom_bar(aes(fill = type)) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +  # Fixed deprecated ..count..
  theme_minimal() +
  labs(title = "Distribution of Wine Types",
       x = "Wine Type",
       y = "Count") +
  scale_fill_manual(values = c("red" = "#8B0000", "white" = "#F0E68C"))

# Perform cross-validation with error handling
perform_cv <- function(data, model_type, k = 5) {
  set.seed(123)
  
  # Create folds
  folds <- createFolds(data$type, k = k, list = TRUE)
  accuracies <- numeric(k)
  
  for(i in 1:k) {
    # Split data
    test_indices <- folds[[i]]
    train_data <- data[-test_indices, ]
    test_data <- data[test_indices, ]
    
    # Select features (excluding type and quality)
    features <- names(data)[!names(data) %in% c("type", "quality")]
    
    tryCatch({
      if(model_type == "logistic") {
        model <- glm(type ~ ., data = train_data[, c("type", features)], 
                     family = "binomial")
        pred_prob <- predict(model, test_data, type = "response")
        pred <- factor(ifelse(pred_prob > 0.5, "white", "red"), levels = levels(data$type))
        
      } else if(model_type == "lda") {
        model <- lda(type ~ ., data = train_data[, c("type", features)])
        pred <- predict(model, test_data)$class
        
      } else if(model_type == "qda") {
        model <- qda(type ~ ., data = train_data[, c("type", features)])
        pred <- predict(model, test_data)$class
        
      } else if(model_type == "knn") {
        train_scaled <- scale(train_data[, features])
        test_scaled <- scale(test_data[, features])
        # Add center and scale attributes to test data
        attributes(test_scaled)$`scaled:center` <- attributes(train_scaled)$`scaled:center`
        attributes(test_scaled)$`scaled:scale` <- attributes(train_scaled)$`scaled:scale`
        pred <- knn(train_scaled, test_scaled, train_data$type, k = 5)
        
      } else if(model_type == "rf") {
        model <- randomForest(type ~ ., data = train_data[, c("type", features)],
                              ntree = 500)
        pred <- predict(model, test_data)
      }
      
      accuracies[i] <- mean(pred == test_data$type)
    }, error = function(e) {
      warning(paste("Error in fold", i, "for model", model_type, ":", e$message))
      accuracies[i] <- NA
    })
  }
  
  return(mean(accuracies, na.rm = TRUE))
}

# Run all models
models <- c("logistic", "lda", "qda", "knn", "rf")
results <- sapply(models, function(m) perform_cv(wine_data, m))

# Create results dataframe
results_df <- data.frame(
  Model = c("Logistic Regression", "LDA", "QDA", "KNN", "Random Forest"),
  Accuracy = results
)

# Plot model comparison with accuracy values
model_comparison <- ggplot(results_df, aes(x = reorder(Model, -Accuracy), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "#8B0000") +
  geom_text(aes(label = sprintf("%.3f", Accuracy)), 
            vjust = -0.5,  # Position above the bars
            color = "black",
            size = 4) +  # Text size
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Model Performance Comparison",
       x = "Model Type",
       y = "Accuracy") +
  scale_y_continuous(limits = c(0, 1.1))  # Extend y-axis to make room for labels

# Feature importance using Random Forest
set.seed(123)
rf_model <- randomForest(type ~ ., 
                         data = wine_data[, !names(wine_data) %in% c("quality")],
                         ntree = 500,
                         importance = TRUE)

# Create importance dataframe
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "MeanDecreaseGini"]
)

# Plot feature importance
feature_importance <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#8B0000") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance for Wine Type Classification",
       subtitle = "Based on Random Forest Mean Decrease in Gini",
       x = "Features",
       y = "Importance Score") +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  )

# Display all plots
grid.arrange(wine_distribution, model_comparison, feature_importance, ncol = 1)

# Print detailed results
cat("\nClassification Results:\n")
print(results_df[order(-results_df$Accuracy), ])

cat("\nFeature Importance Scores:\n")
print(importance_df[order(-importance_df$Importance), ])