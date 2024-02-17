# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(broom)
library(glmnet)
library(pROC)
library(caret)

# Set working directory
setwd("C:/Users/shrad/Documents")

# Read datasets kidney_stone_data.csv into data
data <- read_csv("kidney_stone_data.csv")
head(data)

# Calculate the number and frequency of success and failure of each treatment 
data %>% 
  group_by(treatment, success) %>%
  summarise(N = n()) %>% 
  mutate(Freq = round(N/sum(N), digits = 3))

# Check the structure of the dataset
str(data)

# Check for missing values
sum(is.na(data))

# Summary statistics
summary(data)

# Recreate the Treatment X Success summary table
table_summary <- data %>%
  group_by(treatment) %>%
  summarise(success_count = sum(success), total_cases = n(), success_rate = mean(success))

# Print the summary table
print(table_summary)

# Data Visualization - Bar plot for Treatment X Success
ggplot(data, aes(x = treatment, fill = factor(success))) +
  geom_bar(position = "fill") +
  labs(title = "Treatment X Success",
       x = "Treatment",
       y = "Proportion of Success",
       fill = "Success") +
  theme_minimal()

# Data Visualization - Stratified bar plot by stone size
ggplot(data, aes(x = treatment, fill = factor(success))) +
  geom_bar(position = "fill") +
  facet_grid(~stone_size) +
  labs(title = "Stratified Treatment X Success by Stone Size",
       x = "Treatment",
       y = "Proportion of Success",
       fill = "Success") +
  theme_minimal()

# Feature selection: Consider selecting relevant features
# Example: Use only stone_size and treatment for simplicity
selected_data <- data %>% select(success, stone_size, treatment)

# Multiple logistic regression with feature selection
m <- glm(success ~ stone_size + treatment, data = selected_data, family = "binomial")
tidy_m <- tidy(m)

coefficients <- tidy_m %>%
  select(term, estimate, std.error, p.value)

# Plot the coefficient estimates with 95% CI for each term in the model
tidy_m %>%
  ggplot(aes(x = term, y = estimate)) +
  geom_pointrange(aes(ymin = estimate - 1.96 * std.error, ymax = estimate + 1.96 * std.error)) +
  geom_hline(yintercept = 0)


# Logistic regression predictions and confusion matrix
predictions <- predict(m, newdata = selected_data, type = "response")
threshold <- 0.5
predicted_classes <- ifelse(predictions > threshold, 1, 0)
conf_matrix <- confusionMatrix(factor(predicted_classes), factor(selected_data$success))
print("Confusion Matrix:")
print(conf_matrix)


# Evaluate the model with ROC curve and AUC
roc_curve <- roc(response = selected_data$success, predictor = predictions)
plot(roc_curve, main = "Receiver Operating Characteristic (ROC) Curve", col = "blue")
text(0.8, 0.2, paste("AUC =", round(auc(roc_curve), 2)), col = "blue")




#############################################################
# Convert categorical variables to numerical using one-hot encoding
data_encoded <- model.matrix(~ treatment + stone_size + 0, data = data)
data_encoded <- as.data.frame(data_encoded)
head(data_encoded)

# Combine one-hot encoded features with the 'success' column in the original data
data_combined <- cbind(data_encoded, success = data$success)

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
train_index <- sample(seq_len(nrow(data_combined)), 0.8 * nrow(data_combined))
train_data <- data_combined[train_index, ]
test_data <- data_combined[-train_index, ]

# Build a logistic regression model
logistic_model <- glm(success ~ ., data = train_data, family = "binomial")

# Make predictions on the test set
predictions <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
threshold <- 0.5
predicted_classes <- ifelse(predictions > threshold, 1, 0)

# Evaluate the model with confusion matrix
conf_matrix <- confusionMatrix(factor(predicted_classes), factor(test_data$success))

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Check the dimensions of the confusion matrix
print("Dimensions of Confusion Matrix:")
print(dim(conf_matrix))

# Continue with evaluation metrics
accuracy <- conf_matrix$overall["Accuracy"]

# Print evaluation metrics
cat("Accuracy:", round(accuracy, 2), "\n")


# Build a Lasso regression model using glmnet
lasso_model <- cv.glmnet(as.matrix(train_data[, -ncol(train_data)]), 
                         train_data$success, 
                         alpha = 1)  # alpha = 1 for Lasso

# Make predictions on the test set
lasso_predictions <- predict(lasso_model, newx = as.matrix(test_data[, -ncol(test_data)]), s = "lambda.min", type = "response")

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
threshold <- 0.5
lasso_predicted_classes <- ifelse(lasso_predictions > threshold, 1, 0)

# Evaluate the Lasso model
lasso_conf_matrix <- confusionMatrix(factor(lasso_predicted_classes), factor(test_data$success))

# Print the confusion matrix
print("Lasso Confusion Matrix:")
print(lasso_conf_matrix)

# Continue with evaluation metrics
lasso_accuracy <- lasso_conf_matrix$overall["Accuracy"]

# Print evaluation metrics
cat("Lasso Accuracy:", round(lasso_accuracy, 2), "\n")

