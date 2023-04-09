library('backports')
library('checkmate')
library('insight')
library('marginaleffects') # parametric g-formula
library('zoo') 
library('lmtest') 
library('sandwich') # cluster-robust SEs
library('lme4')
library('arm') # bayes GLM
library('tipr')
library('EValue') # E-value for unmeasured confouding

fit <- function(data, trt, out, family, covs, lag, behavior=TRUE) {
    f <- paste(out, "~", trt, "+", paste(covs, collapse="+"))
    if (behavior) {
        f <- paste(
            f, "+", 
            paste0(lag, "*log_time_prev_1")
        )
    }
    model <- bayesglm(as.formula(f), data=data, family=family)
    return (model)
}

pseudoR2 <- function(model, data) {
    llnull <- logLik(update(model, ~ 1, data=model.frame(model)))
    pR2 <- 1 - exp((llnull - logLik(model)) * (2/nrow(data)))
    return (pR2)
}

sensitivity <- function(orig_lo, orig_hi, adj_model, trt, transform) {
    adj <- coeftest(adj_model, vcov. = vcovCL, cluster=~user_id)[trt,]
    adj_est <- exp(adj[["Estimate"]])
    adj_lo <- exp(adj[["Estimate"]] - 1.96*adj[["Std. Error"]])
    adj_hi <- exp(adj[["Estimate"]] + 1.96*adj[["Std. Error"]])
    obs_e <- observed_covariate_e_value(
        orig_lo, orig_hi, adj_lo, adj_hi, transform=transform 
    )
    return(list("evalue"=obs_e))
}

DATA_DIR <- "data"
COS_SIM_CUTOFF <- 0.85

for (cmpt_group in c("extracted_results", "google_services")) {

    if (cmpt_group == "extracted_results") {
        treatments <- c(
            "direct_answer",
            "featured_snippet",
            "knowledge_panel_rhs",
            "top_stories",
            "top_image_carousel"
        )
        outcomes <- c("click", "time_elapsed")
    } else {
        treatments <- c(
            "videos",
            "images",
            "ad",
            "shopping_ads",
            "local_results",
            "scholarly_articles",
            "map_results"
        )
        outcomes <- c("organic_click_3p", "organic_click_1p")
    }
    pR2 <- c()

    for (trt in treatments) {

        for (out in outcomes) {

            print(paste('computing ATT of', trt, 'on', out, "with cos_sim", COS_SIM_CUTOFF))
            trt_dir <- ifelse(trt == "direct_answer", "direct_answer_combined", trt)
            fp_in <- file.path(DATA_DIR, trt_dir, paste0("matches_", COS_SIM_CUTOFF, ".csv"))
            data <- read.table(fp_in, sep=",", comment.char="", header=TRUE)

            # 1. prep covariates
            covs <- colnames(data)[6:(ncol(data)-6)]
            comps_mask <- sapply(covs, function(c) startsWith(c, "n_"))
            other_comps <- covs[comps_mask]
            topics <- covs[comps_mask == FALSE]
            if (out %in% c("click", "organic_click_3p", "organic_click_1p")) {
                lag <- paste0(out, "_prev_1")
                family <- binomial(link="logit")
            } else {
                lag <- "log_time_elapsed_prev_1"
                family <- Gamma(link="log")
            }

            vars <- c(covs, lag, "log_time_prev_1")
            data[vars] <- sapply(data[vars], function(cov) rescale(cov))

            # 2. outcome models: logistic (click) / log-linked gamma (time_elapsed)
            model <- fit(data, trt, out, family, c(other_comps, topics), lag) 
            fp_out <- file.path(DATA_DIR, trt_dir, paste0("model_", out, "_", COS_SIM_CUTOFF, ".csv"))
            write.csv(round(summary(model)$coef, 2), file=fp_out)
            
            # 2b. diagnostics: binned residuals (for logistic regression) and Cox-Snell pseudo R^2
            fp_out <- file.path(DATA_DIR, trt_dir, paste0("residuals_", out, "_", COS_SIM_CUTOFF, ".png"))
            png(fp_out)
            if (out %in% c("click", "organic_click_3p", "organic_click_1p")) {
                binnedplot(fitted(model), resid(model, type="response"))
            }
            else {
                plot(fitted(model), resid(model))
            }
            dev.off()
            pR2 <- rbind(pR2, list(
                "treatment"=trt, 
                "outcome"=out, 
                "cos_sim_cutoff"=COS_SIM_CUTOFF,
                "pR2"=pseudoR2(model, data)
            ))

            # 3. parametric g-formula w/ cluster-robust SEs
            cmp <- comparisons(model, newdata=data, variables=c(trt), vcov=~user_id)
            fp_out <- file.path(DATA_DIR, trt_dir, paste0("att_", out, "_", COS_SIM_CUTOFF, ".csv"))
            write.csv(summary(cmp), file=fp_out)

            # 4. EValue sensitivity analysis
            print('conducting EValue sensitivity analysis')
            orig <- coeftest(model, vcov. = vcovCL, cluster=~user_id)[trt,]
            orig_est <- exp(orig[["Estimate"]])
            orig_lo <- exp(orig[["Estimate"]] - 1.96*orig[["Std. Error"]])
            orig_hi <- exp(orig[["Estimate"]] + 1.96*orig[["Std. Error"]])

            if (out %in% c("click", "organic_click_3p", "organic_click_1p")) {
                rare <- mean(data[[out]]) < 0.15
                if (rare) {transform <- NULL} else {transform <- "OR"}
                eval <- evalues.OR(orig_est, orig_lo, orig_hi, rare=rare)
            } else {
                transform <- NULL
                eval <- evalues.RR(orig_est, orig_lo, orig_hi) 
            }
            tip_point <- list("evalue"=eval["E-values", "point"])
            eval_lb <- ifelse(orig_est < 1, "upper", "lower")
            tip_ci <- list("evalue"=eval["E-values", eval_lb])

            # 4b. observed covariate EValue
            fp_out <- file.path(DATA_DIR, trt_dir, paste0("evalue_", out, "_", COS_SIM_CUTOFF, ".csv"))
            if (eval["E-values", eval_lb] == 1) {
                write.csv(rbind(tip_point, tip_ci), fp_out)
            } else {
                print('benchmarking EValue against measured confounders')
                no_comps <- fit(data, trt, out, family, c(topics), lag)
                no_topics <- fit(data, trt, out, family, c(other_comps), lag)
                no_behavior <- fit(data, trt, out, family, c(other_comps, topics), lag, behavior=FALSE)

                no_comps <- sensitivity(orig_lo, orig_hi, no_comps, trt, transform)
                no_topics <- sensitivity(orig_lo, orig_hi, no_topics, trt, transform)
                no_behavior <- sensitivity(orig_lo, orig_hi, no_behavior, trt, transform)

                write.csv(
                    rbind(tip_point, tip_ci, no_comps, no_topics, no_behavior),
                    fp_out
                )
            }
        }
    }

    fp_out <- file.path(DATA_DIR, paste0("pseudoR2", "_", cmpt_group, ".csv"))
    write.csv(pR2, file=fp_out)
}