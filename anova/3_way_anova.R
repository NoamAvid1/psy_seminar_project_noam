if (!requireNamespace("tidyverse", quietly = TRUE)) {
  install.packages("tidyverse")
  install.packages("ggpubr")
  install.packages("rstatix")
}
library(tidyverse)
library(ggpubr)
library(rstatix)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
if (!requireNamespace("ez", quietly = TRUE)) {
  install.packages("ez")
}
library(ez)

df <- read.csv("C:/Users/t-noamavidor/Documents/personal/University/Face_recognition_seminar/psy_seminar_project_noam/seminar_dists_results/experiment_1_2/batches_aucs/all.csv")


# summary statistics:
stat_df <- df %>%
  group_by(dcnn_domain, img_domain, orientation) %>%
  get_summary_stats(auc, type = "mean_sd")

write.csv(stat_df, "C:/Users/t-noamavidor/Documents/personal/University/Face_recognition_seminar/psy_seminar_project_noam/seminar_dists_results/experiment_1_2/batches_aucs/stats.csv")


anova_result <- ezANOVA(data = df, 
                        dv=auc,
                        wid = id, 
                        within = .(img_domain, orientation), 
                        between = dcnn_domain, 
                        type = 3, 
                        detailed = TRUE)

print(anova_result)

print("second way to test anova:-------------")

res.aov <- anova_test(
  data = df, dv = auc, wid = id,
  between = dcnn_domain, within = c(img_domain, orientation)
)
anova_table <- get_anova_table(res.aov)
write.csv(anova_table, "C:/Users/t-noamavidor/Documents/personal/University/Face_recognition_seminar/psy_seminar_project_noam/seminar_dists_results/experiment_1_2/batches_aucs/anova_table.csv")



# PostHoc
print("Post-hoc:-------------")
# Two-way ANOVA at each orientation group level
two.way <- df %>%
  group_by(img_domain) %>%
  anova_test(dv = auc, wid = id, within = c(orientation))
print(two.way)  
write.csv(two.way, "C:/Users/t-noamavidor/Documents/personal/University/Face_recognition_seminar/psy_seminar_project_noam/seminar_dists_results/experiment_1_2/batches_aucs/post_hoc_inversion.csv")