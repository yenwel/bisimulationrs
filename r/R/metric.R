model_source <- function(m) if (inherits(m, "pepa_model")) m$source else m

drop_null <- function(x) x[!vapply(x, is.null, logical(1))]

or_na <- function(x, na = NA_real_) if (is.null(x)) na else x

pepa_atom <- function(state, component = NULL) {
  if (is.list(state)) return(drop_null(list(component = state$component, state = state$state)))
  drop_null(list(component = component, state = as.character(state)))
}

pepa_ok <- function(r, label) {
  if (!is.null(r$error)) pepa_abort(diagnostics_frame(r$error), label)
  r$ok
}

#' How far apart two models are
#'
#' How differently models `a` and `b` behave, seen through the count of
#' `state`: both run side by side in the way that makes the count look most
#' alike, and differences `half_life` time units ahead count half (a
#' bisimulation distance, Calo et al., NeurIPS 2024).
#'
#' @param a,b `pepa_model` objects or PEPA source text.
#' @param state The local state to count (a stage `Stage_1`, `Stage_2` is
#'   counted as `Stage`).
#' @param half_life Time units after which a difference counts half.
#' @param component Component, when the model has several.
#' @param max_states State-space limit for each model.
#' @return An object of class `pepa_distance`: `distance` in \[0, 1\] (0: the
#'   count cannot tell the models apart; 1: it always differs by its whole
#'   range, `scale`), `error_bound`, `half_life`, `scale`, `pairs` (data
#'   frame of the state pairs adding most: `a`, `b`, `gap`, `occupancy`,
#'   `contribution`) and `explanation`.
#' @examples
#' herd <- function(infect) sprintf("infect = %s;
#'   Healthy = (infect, infect).Sick; Sick = (recover, 0.5).Healthy;
#'   Healthy[4]", infect)
#' pepa_distance(herd(0.1), herd(0.3), "Sick", half_life = 5)
#' @export
pepa_distance <- function(a, b, state, half_life, component = NULL, max_states = 200000) {
  r <- pepa_ok(pepa_call(list(op = "distance", a = model_source(a), b = model_source(b),
                              observe = pepa_atom(state, component), half_life = as.numeric(half_life),
                              max_states = max_states)), "distance")
  pairs <- if (length(r$pairs)) as.data.frame(r$pairs, stringsAsFactors = FALSE)[, c("a", "b", "gap", "occupancy", "contribution")] else
    data.frame(a = character(), b = character(), gap = numeric(), occupancy = numeric(), contribution = numeric())
  structure(list(distance = r$distance, error_bound = r$error_bound, half_life = r$half_life,
                 scale = r$scale, pairs = pairs, explanation = r$explanation),
            class = "pepa_distance")
}

#' @export
print.pepa_distance <- function(x, ...) {
  cat(sprintf("Distance %.4f (± %.1e)\n%s\n", x$distance, x$error_bound, x$explanation))
  if (nrow(x$pairs)) {
    cat("\n")
    print(utils::head(x$pairs, 5), row.names = FALSE, digits = 3)
  }
  invisible(x)
}

#' Are two models within a tolerance?
#'
#' (epsilon, delta)-bisimulation (Spork et al., CAV 2025): related states
#' have equal counts of `labels`, jump probabilities within `epsilon` and
#' exit rates within a factor `exp(delta)`. With `horizon`, also bounds how
#' much the probability of reaching any labelled situation by then can differ.
#'
#' @inheritParams pepa_distance
#' @param labels States whose counts label each global state: a character
#'   vector, or a list of `list(component =, state =)`.
#' @param delta Tolerance on rates (log factor), at least 0.
#' @param epsilon Tolerance on jump probabilities in \[0, 1\]; `NULL` reports
#'   only the smallest one.
#' @param horizon Time for the reachability bound, or `NULL`.
#' @return A list of class `pepa_epsdelta` with `delta`, `min_epsilon` (`NA`
#'   when no epsilon is enough), `within` (`NA` without `epsilon`), `rate`,
#'   `reach_bound`, `rate_only_bound` and `explanation`.
#' @export
pepa_epsdelta <- function(a, b, labels, delta, epsilon = NULL, horizon = NULL, max_states = 200000) {
  atoms <- lapply(if (is.character(labels)) as.list(labels) else labels, pepa_atom)
  r <- pepa_ok(pepa_call(drop_null(list(op = "epsdelta", a = model_source(a), b = model_source(b),
                                        labels = atoms, delta = as.numeric(delta),
                                        epsilon = if (!is.null(epsilon)) as.numeric(epsilon),
                                        horizon = if (!is.null(horizon)) as.numeric(horizon),
                                        max_states = max_states))), "tolerance check")
  structure(list(delta = r$delta, min_epsilon = or_na(r$min_epsilon), within = or_na(r$within, NA),
                 rate = r$rate, reach_bound = or_na(r$reach_bound), rate_only_bound = or_na(r$rate_only_bound),
                 explanation = r$explanation),
            class = "pepa_epsdelta")
}

#' @export
print.pepa_epsdelta <- function(x, ...) {
  cat(x$explanation, "\n")
  invisible(x)
}

stream_report <- function(r) {
  m <- if (is.matrix(r$matching)) r$matching else matrix(unlist(r$matching), ncol = 2, byrow = TRUE)
  structure(list(estimate = r$estimate, steps = r$steps, matching = stats::setNames(m[, 2], m[, 1]),
                 explanation = or_na(r$explanation, NA_character_)),
            class = "pepa_stream_distance")
}

#' Distance between two logs of states
#'
#' Estimates the bisimulation distance from two sequences of observed states,
#' without a model (Calo et al., NeurIPS 2025). Consecutive entries are
#' transitions. The estimate settles slowly: compare logs of equal length.
#'
#' @param a,b Character vectors: the states observed, in order.
#' @param values Named numeric vector: each state's observation, e.g.
#'   `c(ok = 0, sick = 1)`.
#' @param gamma Discount in (0, 1) for later steps.
#' @return A list of class `pepa_stream_distance` with `estimate`, `steps`,
#'   `matching` (named character vector: the state of `b` each state of `a`
#'   is paired with) and `explanation`.
#' @export
pepa_stream_distance <- function(a, b, values, gamma) {
  r <- pepa_ok(pepa_call(list(op = "stream_distance", a = I(as.character(a)), b = I(as.character(b)),
                              values = as.list(values), gamma = as.numeric(gamma))), "stream distance")
  stream_report(r)
}

#' @export
print.pepa_stream_distance <- function(x, ...) {
  cat(sprintf("Estimated distance %.4f after %d steps\n", x$estimate, x$steps))
  if (!is.na(x$explanation)) cat(x$explanation, "\n")
  invisible(x)
}

#' Compare how two sensor series evolve
#'
#' Dynamical similarity analysis (Ostrow et al., NeurIPS 2023): linear
#' dynamics are fitted to each series and compared up to a rotation of
#' coordinates. Levels do not matter, only how the signals evolve. This is
#' not a bisimulation.
#'
#' @param x,y Matrices or data frames with one row per time step and one
#'   column per channel; a numeric vector is one channel.
#' @param delays Number of delays in the embedding, or `NULL` for the default.
#' @param rank Rank of the fitted dynamics, or `NULL` for the default.
#' @return A list with `distance`, `angle` (radians) and `explanation`.
#' @export
pepa_dsa <- function(x, y, delays = NULL, rank = NULL) {
  series <- function(s) {
    s <- as.matrix(s)
    storage.mode(s) <- "double"
    unname(s)
  }
  r <- pepa_ok(pepa_call(drop_null(list(op = "dsa", x = series(x), y = series(y),
                                        delays = if (!is.null(delays)) as.integer(delays),
                                        rank = if (!is.null(rank)) as.integer(rank)))), "series comparison")
  list(distance = r$distance, angle = r$angle, explanation = r$explanation)
}

#' Sessions: comparisons updated as observations arrive
#'
#' Open a session, push events with `pepa_session_push()` as they come, and
#' get the updated answer each time without recomputing what came before.
#' Close it with `pepa_session_close()` when done (it is also closed when
#' garbage collected).
#'
#' - `pepa_session_prefix()`: two systems observed transition by transition,
#'   from initial states `left` and `right`. Events are
#'   `list(left = c(s, a, t))`, `list(right = c(s, a, t))`,
#'   `list(complete_left = s)` and `list(complete_right = s)` (no more
#'   transitions from `s`); a push returns `changed` and `verdict`.
#' - `pepa_session_observe()`: dated counts against a plan, as in
#'   [pepa_check_observations()]; a push takes a data frame of counts and
#'   returns a `pepa_observation_report`.
#' - `pepa_session_stream()`: the sample distance of
#'   [pepa_stream_distance()], one step at a time; events are
#'   `list(a = c(from, to), b = c(from, to))`.
#'
#' @param left,right Initial states of the two systems.
#' @param max_pairs Largest game the session may build.
#' @return A `pepa_session`.
#' @examples
#' s <- pepa_session_prefix("p0", "q0")
#' pepa_session_push(s, list(list(left = c("p0", "a", "p1")), list(complete_left = "p0")))
#' pepa_session_close(s)
#' @export
pepa_session_prefix <- function(left, right, max_pairs = NULL) {
  session_open(drop_null(list(kind = "prefix", left = left, right = right,
                              max_pairs = if (!is.null(max_pairs)) as.integer(max_pairs))))
}

#' @rdname pepa_session_prefix
#' @inheritParams pepa_check_observations
#' @export
pepa_session_observe <- function(model, start, unit = "day", level = 0.95) {
  stopifnot(inherits(model, "pepa_model"))
  session_open(list(kind = "observe", source = model$source, timeline = pepa_timeline(start, unit, level),
                    max_states = model$max_states))
}

#' @rdname pepa_session_prefix
#' @param states_a,states_b The states each log can be in.
#' @inheritParams pepa_stream_distance
#' @export
pepa_session_stream <- function(states_a, states_b, values, gamma) {
  session_open(list(kind = "stream_distance", states_a = I(as.character(states_a)),
                    states_b = I(as.character(states_b)), values = as.list(values), gamma = as.numeric(gamma)))
}

session_open <- function(spec) {
  r <- pepa_ok(pepa_call(list(op = "session_open", session = spec)), "session")
  s <- new.env(parent = emptyenv())
  s$id <- r$id
  s$kind <- spec$kind
  s$closed <- FALSE
  reg.finalizer(s, function(e) if (!e$closed) try(pepa_call(list(op = "session_close", id = e$id)), silent = TRUE),
                onexit = TRUE)
  structure(list(env = s), class = "pepa_session")
}

#' @rdname pepa_session_prefix
#' @param session A `pepa_session`.
#' @param events The events to add (see above).
#' @export
pepa_session_push <- function(session, events) {
  stopifnot(inherits(session, "pepa_session"))
  s <- session$env
  events <- if (s$kind == "observe") observation_list(events) else unname(events)
  r <- pepa_ok(pepa_call(list(op = "session_push", id = s$id, events = events)), "session")
  switch(s$kind, observe = report_from(r), stream_distance = stream_report(r), r)
}

#' @rdname pepa_session_prefix
#' @export
pepa_session_close <- function(session) {
  stopifnot(inherits(session, "pepa_session"))
  s <- session$env
  if (!s$closed) {
    s$closed <- TRUE
    pepa_call(list(op = "session_close", id = s$id))
  }
  invisible(session)
}

#' @export
print.pepa_session <- function(x, ...) {
  cat(sprintf("<PEPA session %d (%s%s)>\n", x$env$id, x$env$kind, if (x$env$closed) ", closed" else ""))
  invisible(x)
}
