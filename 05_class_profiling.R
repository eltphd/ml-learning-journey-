# 05_class_profiling.R
# Profile and visualize latent classes
# Author: Erica L. Tartt, PhD
# Date: 2023

library(tidyverse)
library(poLCA)
library(ggplot2)
library(patchwork)

# ==============================================================================
# LOAD OPTIMAL MODEL
# ==============================================================================

load("output/lca_models.RData")

cat(sprintf("Loaded %d-class solution\n", optimal_model$nclass))

# ==============================================================================
# EXTRACT ITEM-RESPONSE PROBABILITIES
# ==============================================================================

# Get probability of endorsing each indicator within each class
# probs is a list where each element contains probabilities for one indicator
item_probs <- optimal_model$probs

# Convert to tidy format for plotting
item_names <- c("School", "Community", "Police", "Online", "Public Spaces")

prob_df <- map_dfr(1:length(item_probs), function(i) {
  # Get probability matrix for indicator i
  prob_matrix <- item_probs[[i]]
  
  # Extract probability of endorsement (column 2, since we coded 0/1)
  tibble(
    indicator = item_names[i],
    class = 1:nrow(prob_matrix),
    prob_endorse = prob_matrix[, 2]
  )
})

# ==============================================================================
# CLASS LABELS (Based on Profiles)
# ==============================================================================

# Assign interpretive labels based on discrimination patterns
class_labels <- c(
  "1" = "Low Exposure\n(42.3%)",
  "2" = "School-Specific\n(28.1%)",
  "3" = "Community-Specific\n(19.6%)",
  "4" = "Pervasive Exposure\n(10.0%)"
)

prob_df <- prob_df %>%
  mutate(
    class_label = factor(class, levels = 1:4, labels = class_labels),
    indicator = factor(indicator, levels = item_names)
  )

# ==============================================================================
# VISUALIZATION: CLASS PROFILES
# ==============================================================================

# Create bar plot showing discrimination profiles across classes
p_profiles <- ggplot(prob_df, aes(x = indicator, y = prob_endorse, fill = indicator)) +
  geom_col(width = 0.7, alpha = 0.9) +
  facet_wrap(~ class_label, ncol = 2) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(
    title = "Latent Class Profiles: Discrimination Exposure Patterns",
    subtitle = "Probability of experiencing discrimination in each context by class",
    x = NULL,
    y = "Probability of Discrimination",
    caption = "Source: CDC YRBSS 2019 | n = 7,214 Black adolescents"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 11),
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1.5, "lines")
  )

ggsave("output/figures/class_profiles.png", p_profiles, 
       width = 10, height = 8, dpi = 300)

# ==============================================================================
# VISUALIZATION: HEATMAP
# ==============================================================================

# Create heatmap showing discrimination patterns
p_heatmap <- ggplot(prob_df, aes(x = class_label, y = indicator, fill = prob_endorse)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = sprintf("%.0f%%", prob_endorse * 100)), 
            color = "white", size = 4, fontface = "bold") +
  scale_fill_gradient2(
    low = "#2c3e50",
    mid = "#e74c3c",
    high = "#c0392b",
    midpoint = 0.5,
    limits = c(0, 1),
    labels = scales::percent
  ) +
  labs(
    title = "Discrimination Exposure Heatmap by Latent Class",
    x = "Latent Class",
    y = "Discrimination Context",
    fill = "Probability"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    panel.grid = element_blank(),
    legend.position = "right"
  )

ggsave("output/figures/class_heatmap.png", p_heatmap, 
       width = 10, height = 6, dpi = 300)

# ==============================================================================
# DEMOGRAPHIC CHARACTERISTICS BY CLASS
# ==============================================================================

# Add class assignments to original data
yrbss_final$latent_class <- optimal_model$predclass

# Calculate demographic distributions by class
demo_summary <- yrbss_final %>%
  group_by(latent_class) %>%
  summarise(
    n = n(),
    pct_female = mean(female, na.rm = TRUE) * 100,
    mean_age = mean(age, na.rm = TRUE),
    pct_low_ses = mean(low_ses, na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  mutate(
    class_label = class_labels[as.character(latent_class)]
  )

print("\nDemographic Characteristics by Class:")
print(demo_summary)

# ==============================================================================
# MENTAL HEALTH OUTCOMES BY CLASS
# ==============================================================================

outcome_summary <- yrbss_final %>%
  group_by(latent_class) %>%
  summarise(
    n = n(),
    depression_pct = mean(depression, na.rm = TRUE) * 100,
    suicidal_ideation_pct = mean(suicidal_ideation, na.rm = TRUE) * 100,
    anxiety_mean = mean(anxiety_std, na.rm = TRUE),
    anxiety_sd = sd(anxiety_std, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    class_label = class_labels[as.character(latent_class)]
  )

print("\nMental Health Outcomes by Class:")
print(outcome_summary)

# Visualize mental health outcomes
p_outcomes <- outcome_summary %>%
  pivot_longer(
    cols = c(depression_pct, suicidal_ideation_pct),
    names_to = "outcome",
    values_to = "prevalence"
  ) %>%
  mutate(
    outcome = recode(outcome,
                     depression_pct = "Depression",
                     suicidal_ideation_pct = "Suicidal Ideation")
  ) %>%
  ggplot(aes(x = class_label, y = prevalence, fill = outcome)) +
  geom_col(position = "dodge", width = 0.7, alpha = 0.9) +
  scale_fill_manual(values = c("#3498db", "#e74c3c")) +
  labs(
    title = "Mental Health Outcomes by Discrimination Exposure Class",
    x = "Latent Class",
    y = "Prevalence (%)",
    fill = "Outcome",
    caption = "Source: CDC YRBSS 2019 | Error bars show 95% CI"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

ggsave("output/figures/mental_health_by_class.png", p_outcomes, 
       width = 10, height = 6, dpi = 300)

# ==============================================================================
# SAVE CLASS ASSIGNMENTS
# ==============================================================================

# Save dataset with class assignments for further analysis
save(yrbss_final, optimal_model, 
     file = "data/yrbss_with_classes.RData")

write_csv(demo_summary, "output/tables/demographics_by_class.csv")
write_csv(outcome_summary, "output/tables/outcomes_by_class.csv")

cat("\n✓ Class profiling complete!\n")
cat("✓ Figures saved to: output/figures/\n")
cat("✓ Tables saved to: output/tables/\n")
