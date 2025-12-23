library(tidyverse)
library(lubridate)

create_features <- function(df) {
  df %>%
    mutate(
      # FIX: Use parse_date_time to handle multiple potential formats
      # It tries YYYY-MM-DD, MM/DD/YYYY, and DD/MM/YYYY
      date = as.Date(parse_date_time(date, orders = c("ymd", "mdy", "dmy"))),
      
      year = year(date),
      month = month(date),
      semester = semester(date),
      quarter = quarter(date),
      day_in_week = wday(date, label = TRUE, abbr = FALSE, week_start = 1),
      week_in_year = isoweek(date),
      day_in_year = yday(date)
    ) %>%
    # Safety Check: Remove rows where date failed to parse (preventing the "0 cases" error)
    filter(!is.na(date))
}