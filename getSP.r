# Install once:
# install.packages(c("alphavantager","quantmod","dplyr","readr","lubridate","purrr"))

library(alphavantager)
library(quantmod)   # xts tools, write.zoo, charting
library(dplyr)
library(readr)
library(lubridate)
library(purrr)

# 1) Put your Alpha Vantage key here (free to get on their site)
av_api_key("YOUR_ALPHA_VANTAGE_KEY")

# 2) Helper to download all 24 monthly "extended" slices (last 2 years)
av_fetch_intraday_extended <- function(symbol, interval = "5min",
                                       adjusted = TRUE,
                                       sleep_secs = 12) {
  slices <- c(paste0("year1month", 1:12), paste0("year2month", 1:12))
  adj_flag <- if (adjusted) "true" else "false"

  pieces <- map(slices, function(sl) {
    # Respect the AV free tier rate limit (~5 req/min)
    Sys.sleep(sleep_secs)
    tryCatch(
      av_get(symbol,
             av_fun   = "TIME_SERIES_INTRADAY_EXTENDED",
             interval = interval,
             slice    = sl,
             adjusted = adj_flag),
      error = function(e) NULL
    )
  })

  pieces <- compact(pieces)
  if (length(pieces) == 0) return(NULL)
  bind_rows(pieces)
}

# 3) Main: get SPY 5-min for 2024-01-01..2024-02-01, convert to xts, save CSV
get_spy_5m_jan2024 <- function(start_date = "2024-01-01",
                               end_date   = "2024-02-01",
                               tz_out     = "America/Los_Angeles",
                               outfile    = "SPY_5m_2024-01.csv",
                               filter_rth = TRUE) {
  raw <- av_fetch_intraday_extended("SPY", interval = "5min", adjusted = TRUE)

  if (is.null(raw) || nrow(raw) == 0) {
    stop("Alpha Vantage returned no data. Check your API key and plan limits.")
  }

  df <- raw %>%
    mutate(
      DatetimeUTC   = ymd_hms(time, tz = "UTC"),
      DatetimeLocal = with_tz(DatetimeUTC, tz_out)
    ) %>%
    filter(DatetimeUTC >= ymd_hms(paste0(start_date, " 00:00:00"), tz = "UTC"),
           DatetimeUTC <  ymd_hms(paste0(end_date,   " 00:00:00"), tz = "UTC")) %>%
    arrange(DatetimeUTC) %>%
    rename(Open = open, High = high, Low = low, Close = close,
           AdjClose = adjusted_close, Volume = volume)

  if (!nrow(df)) stop("No rows in the requested window after filtering.")

  # Optional: keep Regular Trading Hours (NYSE) 09:30–16:00 America/New_York
  if (filter_rth) {
    ny   <- with_tz(df$DatetimeUTC, "America/New_York")
    mins <- hour(ny) * 60 + minute(ny)
    rth  <- mins >= (9*60 + 30) & mins <= (16*60)
    df   <- df[rth, , drop = FALSE]
  }

  df_out <- df %>% select(DatetimeUTC, DatetimeLocal, Open, High, Low, Close, AdjClose, Volume)
  write_csv(df_out, outfile)

  x <- xts::xts(as.matrix(df_out[, c("Open","High","Low","Close","AdjClose","Volume")]),
                order.by = df_out$DatetimeUTC, tzone = "UTC")

  message("Saved ", nrow(df_out), " rows to: ", outfile, " (symbol: SPY)")
  invisible(list(data = df_out, xts = x))
}

# 4) Run it
get_spy_5m_jan2024(
  start_date = "2024-01-01",
  end_date   = "2024-02-01",
  tz_out     = "America/Los_Angeles",
  outfile    = "SPY_5m_2024-01.csv",
  filter_rth = TRUE
)

