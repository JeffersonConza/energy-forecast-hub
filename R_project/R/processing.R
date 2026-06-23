library(tidyverse)
library(lubridate)
library(slider)

# Keep lookup in a global variable to avoid reloading it on every API request
.rolling_mean_lookup <- NULL

get_rolling_mean_lookup <- function() {
  if (!is.null(.rolling_mean_lookup)) {
    return(.rolling_mean_lookup)
  }
  
  path_train <- "data/df_train.csv"
  path_test <- "data/df_test.csv"
  if (!file.exists(path_train)) {
    path_train <- "../data/df_train.csv"
    path_test <- "../data/df_test.csv"
  }
  
  if (file.exists(path_train) && file.exists(path_test)) {
    df_train <- read_csv(path_train, show_col_types = FALSE)
    df_test <- read_csv(path_test, show_col_types = FALSE)
    df_combined <- bind_rows(df_train, df_test) %>%
      mutate(
        parsed_date = as.Date(parse_date_time(date, orders = c("ymd", "mdy", "dmy")))
      ) %>%
      arrange(parsed_date)
    
    df_combined <- df_combined %>%
      mutate(
        power_rolling_mean_7d = slide_dbl(power_consumption, mean, .before = 7, .after = -1, .complete = TRUE)
      )
    
    df_combined <- df_combined %>%
      mutate(
        date_str = format(parsed_date, "%Y-%m-%d")
      ) %>%
      filter(!is.na(date_str))
    
    lookup <- setNames(df_combined$power_rolling_mean_7d, df_combined$date_str)
    assign(".rolling_mean_lookup", lookup, envir = .GlobalEnv)
  } else {
    assign(".rolling_mean_lookup", numeric(), envir = .GlobalEnv)
  }
  
  return(get(".rolling_mean_lookup", envir = .GlobalEnv))
}

create_features <- function(df) {
  days_english <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
  
  df <- df %>%
    mutate(
      date = as.Date(parse_date_time(date, orders = c("ymd", "mdy", "dmy"))),
      
      year = as.integer(year(date)),
      month = as.integer(month(date)),
      semester = as.integer(ifelse(month <= 6, 1, 2)),
      quarter = as.integer(quarter(date)),
      day_in_week = as.character(days_english[wday(date, week_start = 1)]),
      week_in_year = as.integer(isoweek(date)),
      day_in_year = as.integer(yday(date))
    ) %>%
    filter(!is.na(date))
  
  if (!"power_rolling_mean_7d" %in% colnames(df)) {
    lookup <- get_rolling_mean_lookup()
    date_strs <- format(df$date, "%Y-%m-%d")
    
    rolling_means <- lookup[date_strs]
    rolling_means[is.na(rolling_means)] <- 1592.96
    
    df$power_rolling_mean_7d <- as.numeric(rolling_means)
  }
  
  return(df)
}