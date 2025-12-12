# 03_lca_model_estimation.R
# Fit latent class analysis models (1-6 class solutions)
# Author: Erica L. Tartt, PhD
# Date: 2023

library(poLCA)
library(tidyverse)

# ==============================================================================
# LOAD PREPARED DATA
# ==============================================================================

load("data/yrbss_prepared.RData")

cat(sprintf("Loaded data: n = %d\n", nrow(yrbss_final)))

# ==============================================================================
# MODEL SPECIFICATION
# ==============================================================================

# Formula for LCA: specify indicators (left side) as a function of intercept only
lca_formula <- cbind(
  disc_school,
  disc_community,
  disc_police,
  disc_online,
  disc_public
) ~ 1

# ==============================================================================
# FIT MODELS WITH 1-6 CLASSES
# ==============================================================================

# Set seed for reproducibility
set.seed(42)

# Storage for models
lca_models <- list()
fit_stats <- tibble(
  n_classes = integer(),
  log_likelihood = numeric(),
  aic = numeric(),
  bic = numeric(),
  entropy = numeric(),
  n_params = integer(),
  smallest_class = numeric()
)

# Fit models iteratively
for (k in 1:6) {
  
  cat(sprintf("\n========================================\n"))
  cat(sprintf("Fitting %d-class model...\n", k))
  cat(sprintf("========================================\n"))
  
  # Fit model with multiple random starts to avoid local maxima
  lca_k <- poLCA(
    formula = lca_formula,
    data = yrbss_final,
    nclass = k,
    maxiter = 5000,
    nrep = 50,  # 50 random starts
    verbose = FALSE,
    calc.se = TRUE
  )
  
  # Store model
  lca_models[[k]] <- lca_k
  
  # Calculate entropy
  # Entropy = 1 - (sum of squared posterior probabilities / N)
  # Higher entropy (closer to 1) = better separation
  posterior_probs <- lca_k$posterior
  entropy <- 1 - sum(apply(posterior_probs, 1, function(x) sum(x^2))) / nrow(yrbss_final)
  
  # Find smallest class proportion
  class_sizes <- table(lca_k$predclass) / nrow(yrbss_final)
  smallest_class <- min(class_sizes) * 100
  
  # Store fit statistics
  fit_stats <- fit_stats %>%
    add_row(
      n_classes = k,
      log_likelihood = lca_k$llik,
      aic = lca_k$aic,
      bic = lca_k$bic,
      entropy = entropy,
      n_params = lca_k$npar,
      smallest_class = smallest_class
    )
  
  cat(sprintf("Log-likelihood: %.2f\n", lca_k$llik))
  cat(sprintf("BIC: %.2f\n", lca_k$bic))
  cat(sprintf("Entropy: %.3f\n", entropy))
  cat(sprintf("Smallest class: %.1f%%\n", smallest_class))
}

# ==============================================================================
# COMPARE MODEL FIT
# ==============================================================================

cat("\n========================================\n")
cat("MODEL COMPARISON\n")
cat("========================================\n\n")

print(fit_stats)

# Plot BIC across solutions
p_bic <- ggplot(fit_stats, aes(x = n_classes, y = bic)) +
  geom_line(size = 1.2, color = "#2c3e50") +
  geom_point(size = 3, color = "#e74c3c") +
  scale_x_continuous(breaks = 1:6) +
  labs(
    title = "Model Fit Comparison: Bayesian Information Criterion",
    subtitle = "Lower BIC indicates better fit",
    x = "Number of Classes",
    y = "BIC"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

ggsave("output/figures/bic_comparison.png", p_bic, width = 8, height = 5, dpi = 300)

# Plot entropy across solutions
p_entropy <- ggplot(fit_stats, aes(x = n_classes, y = entropy)) +
  geom_line(size = 1.2, color = "#2c3e50") +
  geom_point(size = 3, color = "#3498db") +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "#e74c3c", size = 1) +
  scale_x_continuous(breaks = 1:6) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Classification Quality: Entropy",
    subtitle = "Entropy > 0.80 indicates good class separation (red line)",
    x = "Number of Classes",
    y = "Entropy"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

ggsave("output/figures/entropy_comparison.png", p_entropy, width = 8, height = 5, dpi = 300)

# ==============================================================================
# SELECT OPTIMAL MODEL
# ==============================================================================

# Based on BIC, entropy, and interpretability, select 4-class solution
optimal_k <- 4
optimal_model <- lca_models[[optimal_k]]

cat(sprintf("\n✓ Optimal solution: %d classes\n", optimal_k))
cat(sprintf("  BIC: %.2f\n", optimal_model$bic))
cat(sprintf("  Entropy: %.3f\n", fit_stats$entropy[optimal_k]))

# ==============================================================================
# CLASS SIZES
# ==============================================================================

class_counts <- table(optimal_model$predclass)
class_proportions <- prop.table(class_counts) * 100

cat("\nClass Sizes:\n")
for (i in 1:optimal_k) {
  cat(sprintf("  Class %d: n = %d (%.1f%%)\n", 
              i, class_counts[i], class_proportions[i]))
}

# ==============================================================================
# AVERAGE POSTERIOR PROBABILITIES (Classification Quality)
# ==============================================================================

# Calculate average posterior probability of assigned class
avg_posterior <- sapply(1:optimal_k, function(k) {
  mean(optimal_model$posterior[optimal_model$predclass == k, k])
})

cat("\nAverage Posterior Probabilities (should be > 0.70):\n")
for (i in 1:optimal_k) {
  cat(sprintf("  Class %d: %.3f\n", i, avg_posterior[i]))
}

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

save(lca_models, fit_stats, optimal_model, 
     file = "output/lca_models.RData")

write_csv(fit_stats, "output/tables/model_fit_comparison.csv")

cat("\n✓ Models saved to: output/lca_models.RData\n")
cat("✓ Fit statistics saved to: output/tables/model_fit_comparison.csv\n")
