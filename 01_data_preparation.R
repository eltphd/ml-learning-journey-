# 01_data_preparation.R
# Prepare CDC YRBSS data for latent class analysis
# Author: Erica L. Tartt, PhD
# Date: 2023

library(tidyverse)
library(haven)
library(psych)

# ==============================================================================
# LOAD RAW DATA
# ==============================================================================

# Load CDC YRBSS dataset (download from https://www.cdc.gov/healthyyouth/data/yrbs/)
yrbss_raw <- read_sas("data/raw/yrbss_2019.sas7bdat")

# ==============================================================================
# FILTER TO BLACK ADOLESCENTS
# ==============================================================================

# Filter to self-identified Black/African American students
yrbss_black <- yrbss_raw %>%
  filter(race == 3) %>%  # 3 = Black or African American
  filter(age >= 13 & age <= 18)

cat(sprintf("Sample size: n = %d\n", nrow(yrbss_black)))

# ==============================================================================
# CREATE DISCRIMINATION INDICATORS
# ==============================================================================

# Recode discrimination items (original scale: 1=Never, 2=Rarely, 3=Sometimes, 4=Often, 5=Always)
# Dichotomize: 0 = Never/Rarely, 1 = Sometimes/Often/Always

yrbss_prepared <- yrbss_black %>%
  mutate(
    # School-based discrimination
    disc_school = case_when(
      q_disc_school %in% c(1, 2) ~ 0,  # Never/Rarely
      q_disc_school %in% c(3, 4, 5) ~ 1,  # Sometimes or more
      TRUE ~ NA_real_
    ),
    
    # Community discrimination
    disc_community = case_when(
      q_disc_neighborhood %in% c(1, 2) ~ 0,
      q_disc_neighborhood %in% c(3, 4, 5) ~ 1,
      TRUE ~ NA_real_
    ),
    
    # Police encounters
    disc_police = case_when(
      q_disc_police %in% c(1, 2) ~ 0,
      q_disc_police %in% c(3, 4, 5) ~ 1,
      TRUE ~ NA_real_
    ),
    
    # Online/social media discrimination
    disc_online = case_when(
      q_disc_online %in% c(1, 2) ~ 0,
      q_disc_online %in% c(3, 4, 5) ~ 1,
      TRUE ~ NA_real_
    ),
    
    # Public spaces (stores, restaurants, etc.)
    disc_public = case_when(
      q_disc_public %in% c(1, 2) ~ 0,
      q_disc_public %in% c(3, 4, 5) ~ 1,
      TRUE ~ NA_real_
    )
  )

# ==============================================================================
# CREATE OUTCOME VARIABLES
# ==============================================================================

yrbss_prepared <- yrbss_prepared %>%
  mutate(
    # Depression (felt sad/hopeless almost every day for 2+ weeks)
    depression = case_when(
      q26 == 1 ~ 1,  # Yes
      q26 == 2 ~ 0,  # No
      TRUE ~ NA_real_
    ),
    
    # Suicidal ideation (seriously considered suicide)
    suicidal_ideation = case_when(
      q25 == 1 ~ 1,
      q25 == 2 ~ 0,
      TRUE ~ NA_real_
    ),
    
    # Anxiety (continuous scale, standardized)
    anxiety_std = scale(q_anxiety)[,1]
  )

# ==============================================================================
# CREATE DEMOGRAPHIC COVARIATES
# ==============================================================================

yrbss_prepared <- yrbss_prepared %>%
  mutate(
    # Gender
    female = if_else(q2 == 2, 1, 0),
    male = if_else(q2 == 1, 1, 0),
    
    # Age groups
    age_13_14 = if_else(age %in% c(13, 14), 1, 0),
    age_15_16 = if_else(age %in% c(15, 16), 1, 0),
    age_17_18 = if_else(age %in% c(17, 18), 1, 0),
    
    # Grade
    grade = recode(grade_num,
                   `9` = "9th",
                   `10` = "10th",
                   `11` = "11th",
                   `12` = "12th"),
    
    # Socioeconomic proxy (free/reduced lunch)
    low_ses = case_when(
      q_lunch == 1 ~ 1,  # Eligible for free/reduced lunch
      q_lunch == 2 ~ 0,  # Not eligible
      TRUE ~ NA_real_
    )
  )

# ==============================================================================
# MISSING DATA ANALYSIS
# ==============================================================================

# Calculate missing data percentages
missing_summary <- yrbss_prepared %>%
  select(starts_with("disc_"), depression, suicidal_ideation, anxiety_std) %>%
  summarise(across(everything(), ~mean(is.na(.)) * 100)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing")

print(missing_summary)

# Pattern of missingness
md_pattern <- mice::md.pattern(
  yrbss_prepared %>% select(starts_with("disc_"))
)

# ==============================================================================
# FINAL SAMPLE SELECTION
# ==============================================================================

# Require at least 4 out of 5 discrimination indicators
yrbss_final <- yrbss_prepared %>%
  mutate(
    n_disc_items = rowSums(!is.na(select(., starts_with("disc_"))))
  ) %>%
  filter(n_disc_items >= 4)

cat(sprintf("\nFinal analytic sample: n = %d\n", nrow(yrbss_final)))
cat(sprintf("Excluded due to missing data: n = %d (%.1f%%)\n", 
            nrow(yrbss_prepared) - nrow(yrbss_final),
            (nrow(yrbss_prepared) - nrow(yrbss_final)) / nrow(yrbss_prepared) * 100))

# ==============================================================================
# DESCRIPTIVE STATISTICS
# ==============================================================================

# Discrimination prevalence
disc_prevalence <- yrbss_final %>%
  summarise(across(starts_with("disc_"), ~mean(., na.rm = TRUE) * 100)) %>%
  pivot_longer(everything(), names_to = "indicator", values_to = "prevalence_pct")

print("\nDiscrimination prevalence:")
print(disc_prevalence)

# Mental health outcomes
outcome_prevalence <- yrbss_final %>%
  summarise(
    depression_pct = mean(depression, na.rm = TRUE) * 100,
    suicidal_ideation_pct = mean(suicidal_ideation, na.rm = TRUE) * 100,
    anxiety_mean = mean(anxiety_std, na.rm = TRUE),
    anxiety_sd = sd(anxiety_std, na.rm = TRUE)
  )

print("\nMental health outcomes:")
print(outcome_prevalence)

# ==============================================================================
# SAVE PREPARED DATA
# ==============================================================================

save(yrbss_final, file = "data/yrbss_prepared.RData")
write_csv(yrbss_final, "data/yrbss_prepared.csv")

cat("\n✓ Data preparation complete!\n")
cat("✓ Saved to: data/yrbss_prepared.RData\n")
