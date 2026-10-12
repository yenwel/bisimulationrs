fmt_date <- function(x) {
  if (inherits(x, c("Date", "POSIXt"))) format(x, "%d/%m/%Y") else as.character(x)
}

pepa_timeline <- function(start, unit, level) {
  units <- c(s = "second", sec = "second", second = "second", seconds = "second",
             min = "minute", minute = "minute", minutes = "minute",
             h = "hour", hour = "hour", hours = "hour",
             d = "day", day = "day", days = "day", w = "week", week = "week", weeks = "week")
  if (!unit %in% names(units)) stop("unknown time unit '", unit, "'; use second, minute, hour, day or week")
  list(start = fmt_date(start), unit = unname(units[unit]), level = level)
}

observation_list <- function(observations) {
  if (is.character(observations) && length(observations) == 1) {
    observations <- pepa_read_observations(observations)
  }
  observations <- as.data.frame(observations, stringsAsFactors = FALSE)
  missing <- setdiff(c("date", "state", "count"), names(observations))
  if (length(missing)) stop("observations need columns date, state and count; missing ", paste(missing, collapse = ", "))
  lapply(seq_len(nrow(observations)), function(i) {
    o <- list(date = fmt_date(observations$date[i]), state = as.character(observations$state[i]),
              count = as.integer(observations$count[i]))
    if (!is.null(observations$component) && !is.na(observations$component[i])) {
      o$component <- as.character(observations$component[i])
    }
    o
  })
}

report_from <- function(r) {
  rows <- as.data.frame(r$rows, stringsAsFactors = FALSE)
  structure(
    list(on_track = r$on_track, explanation = r$explanation,
         table = rows[, c("date", "elapsed", "component", "state", "observed", "expected", "low", "high", "p_value", "status")],
         start = r$start, unit = r$unit, level = r$level, threshold = r$threshold, methods = r$methods),
    class = "pepa_observation_report"
  )
}

#' Read observations from a CSV file
#'
#' @param path CSV with columns `date` (dd/mm/yyyy), `state`, `count` and
#'   optionally `component`.
#' @return A data frame.
#' @export
pepa_read_observations <- function(path) {
  utils::read.csv(path, colClasses = c(date = "character", state = "character"), stringsAsFactors = FALSE)
}

#' Compare dated counts with the plan
#'
#' For each count, the model (the plan) gives the exact distribution of that
#' count on that date: an expected value, a range that holds with probability
#' `level`, and a p-value. The overall verdict adjusts for the number of
#' counts checked, which stays valid when the same plants are counted
#' repeatedly. A stage modelled as `Stage_1`, `Stage_2` is observed as `Stage`.
#'
#' @param model A `pepa_model` (the plan).
#' @param observations Data frame (or CSV path) with `date` (dd/mm/yyyy or
#'   `Date`), `state`, `count` and optionally `component`.
#' @param start Date of the model's initial state (dd/mm/yyyy or `Date`).
#' @param unit Time unit of the model's rates: `"second"`, `"minute"`,
#'   `"hour"`, `"day"` or `"week"`.
#' @param level Probability covered by the ranges.
#' @return An object of class `pepa_observation_report` with `on_track`,
#'   `explanation` and `table`.
#' @examples
#' plan <- pepa_model("
#'   Plant = (grow, 1/28).Flowering; Flowering = 0;
#'   Plant[20]
#' ")
#' obs <- data.frame(date = "12/06/2026", state = "Flowering", count = 9)
#' pepa_check_observations(plan, obs, start = "15/05/2026")
#' @export
pepa_check_observations <- function(model, observations, start, unit = "day", level = 0.95) {
  stopifnot(inherits(model, "pepa_model"))
  r <- pepa_call(list(op = "observe", source = model$source, timeline = pepa_timeline(start, unit, level),
                      observations = observation_list(observations), max_states = model$max_states))
  if (!is.null(r$error)) pepa_abort(diagnostics_frame(r$error), "observation check")
  report_from(r$ok)
}

#' @export
print.pepa_observation_report <- function(x, ...) {
  cat(if (x$on_track) "On track" else "Off plan", "\n", x$explanation, "\n\n", sep = "")
  print(x$table, row.names = FALSE, digits = 3)
  invisible(x)
}

#' Expected count and range over time
#'
#' @inheritParams pepa_check_observations
#' @param state State (or stage) to forecast.
#' @param times Times since `start`, in `unit`.
#' @param component Component, when the model has several.
#' @return A data frame with `elapsed`, `date`, `expected`, `low` and `high`.
#' @export
pepa_forecast <- function(model, state, times, start, unit = "day", component = NULL, level = 0.95) {
  stopifnot(inherits(model, "pepa_model"))
  req <- list(op = "forecast", source = model$source, timeline = pepa_timeline(start, unit, level),
              state = state, times = I(as.numeric(times)), max_states = model$max_states)
  if (!is.null(component)) req$component <- component
  r <- pepa_call(req)
  if (!is.null(r$error)) pepa_abort(diagnostics_frame(r$error), "forecast")
  data.frame(elapsed = r$ok$elapsed, date = r$ok$dates, expected = r$ok$expected,
             low = r$ok$low, high = r$ok$high, stringsAsFactors = FALSE)
}

#' Fit rate parameters to observed counts
#'
#' Finds the values of `params` that make the counts most likely. Fit one
#' parameter at a time to see which stage differs from the plan, then
#' compare the plan with `$model` using [pepa_compare()].
#'
#' @inheritParams pepa_check_observations
#' @param params Names of rate parameters to fit.
#' @return A list with `params` (data frame: name, planned, fitted, ratio),
#'   `model` (the fitted `pepa_model`), `check` (observation report under
#'   the fitted model) and `explanation`.
#' @export
pepa_fit <- function(model, observations, params, start, unit = "day", level = 0.95) {
  stopifnot(inherits(model, "pepa_model"))
  r <- pepa_call(list(op = "fit", source = model$source, timeline = pepa_timeline(start, unit, level),
                      observations = observation_list(observations), params = I(as.character(params)),
                      max_states = model$max_states))
  if (!is.null(r$error)) pepa_abort(diagnostics_frame(r$error), "fit")
  structure(
    list(params = as.data.frame(r$ok$params, stringsAsFactors = FALSE),
         model = pepa_model(r$ok$fitted_source, max_states = model$max_states),
         check = report_from(r$ok$fitted_check),
         explanation = r$ok$explanation),
    class = "pepa_fit"
  )
}

#' @export
print.pepa_fit <- function(x, ...) {
  cat(x$explanation, "\n\n")
  print(x$params, row.names = FALSE, digits = 4)
  invisible(x)
}

#' Plot the plan's ranges with the observed counts
#'
#' One panel per observed state: the expected count (line), its range
#' (band) and the counts (points; red when outside the range).
#'
#' @inheritParams pepa_check_observations
#' @param horizon Last time shown, in `unit` (default: a bit past the last count).
#' @return The observation report, invisibly.
#' @export
pepa_plot_observations <- function(model, observations, start, unit = "day", level = 0.95, horizon = NULL) {
  report <- pepa_check_observations(model, observations, start, unit, level)
  tab <- report$table
  if (is.null(horizon)) horizon <- max(1, max(tab$elapsed) * 1.25)
  times <- seq(0, horizon, length.out = 101)
  states <- unique(tab$state)
  old <- graphics::par(mfrow = c(length(states), 1), mar = c(3, 4.5, 1.5, 1), oma = c(2, 0, 2, 0))
  on.exit(graphics::par(old))
  for (st in states) {
    rows <- tab[tab$state == st, ]
    band <- pepa_forecast(model, st, times, start, unit, component = rows$component[1], level = level)
    ylim <- range(0, band$high, rows$observed)
    plot(band$elapsed, band$expected, type = "n", ylim = ylim, xlab = "", ylab = st, las = 1)
    graphics::polygon(c(band$elapsed, rev(band$elapsed)), c(band$low, rev(band$high)),
                      col = grDevices::adjustcolor("#2a78d6", 0.18), border = NA)
    graphics::lines(band$elapsed, band$expected, col = "#2a78d6", lwd = 2)
    graphics::points(rows$elapsed, rows$observed, pch = 19,
                     col = ifelse(rows$status == "on_track", "#0a7d32", "#c0352f"))
  }
  graphics::mtext(sprintf("%ss since %s", report$unit, report$start), side = 1, outer = TRUE, line = 0.5)
  graphics::mtext(if (report$on_track) "On track" else "Off plan", side = 3, outer = TRUE, line = 0.5, font = 2)
  invisible(report)
}
