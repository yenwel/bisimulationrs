#' @keywords internal
"_PACKAGE"

pepa_call <- function(request) {
  response <- handle_json(jsonlite::toJSON(request, auto_unbox = TRUE, digits = NA))
  jsonlite::fromJSON(response, simplifyVector = TRUE, simplifyDataFrame = TRUE)
}

diagnostics_frame <- function(items) {
  if (length(items) == 0) {
    return(data.frame(severity = character(), line = integer(), column = integer(),
                      message = character(), stringsAsFactors = FALSE))
  }
  as.data.frame(items, stringsAsFactors = FALSE)[, c("severity", "line", "column", "message")]
}

pepa_abort <- function(diags, label = "model") {
  where <- ifelse(diags$line > 0, sprintf("line %d, column %d: ", diags$line, diags$column), "")
  msg <- paste0("The ", label, " has problems:\n", paste0("  ", diags$severity, ": ", where, diags$message, collapse = "\n"))
  stop(structure(class = c("pepa_error", "error", "condition"),
                 list(message = msg, call = NULL, diagnostics = diags)))
}

#' Check PEPA source text
#'
#' Checks a model without analysing it.
#'
#' @param source PEPA source text.
#' @return A data frame of problems (`severity`, `line`, `column`, `message`);
#'   no rows means the model is valid. Line and column are 1-based, 0 when the
#'   problem is not tied to one place.
#' @examples
#' pepa_check("P = (a, r).P;\nP")
#' @export
pepa_check <- function(source) {
  diagnostics_frame(pepa_call(list(op = "check", source = source))$ok)
}

#' Create a PEPA model
#'
#' PEPA describes a system as components that perform actions at exponential
#' rates (events per second). The model is checked immediately; problems stop
#' with an error of class `pepa_error` whose `diagnostics` element lists them.
#'
#' @param source PEPA source text.
#' @param max_states Stop when the state space grows beyond this many states.
#' @return An object of class `pepa_model`.
#' @examples
#' herd <- pepa_model("
#'   r_disease = 0.02;
#'   Healthy = (disease, r_disease).Sick;
#'   Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
#'   Dead = 0;
#'   Healthy[3]
#' ")
#' herd
#' @export
pepa_model <- function(source, max_states = 200000) {
  stopifnot(is.character(source), length(source) == 1)
  diags <- pepa_check(source)
  errors <- diags[diags$severity == "error", , drop = FALSE]
  if (nrow(errors) > 0) pepa_abort(errors)
  cache <- new.env(parent = emptyenv())
  structure(list(source = source, max_states = max_states, cache = cache), class = "pepa_model")
}

#' @rdname pepa_model
#' @param path Path to a `.pepa` file.
#' @param ... Passed on to `pepa_model()`.
#' @export
pepa_read <- function(path, ...) {
  pepa_model(paste(readLines(path, encoding = "UTF-8", warn = FALSE), collapse = "\n"), ...)
}

pepa_analysis <- function(model) {
  stopifnot(inherits(model, "pepa_model"))
  if (is.null(model$cache$analysis)) {
    r <- pepa_call(list(op = "analyse", source = model$source, max_states = model$max_states))
    if (!is.null(r$error)) pepa_abort(diagnostics_frame(r$error))
    model$cache$analysis <- r$ok
  }
  model$cache$analysis
}

#' @export
summary.pepa_model <- function(object, ...) {
  a <- pepa_analysis(object)
  listed <- is.null(a$note)
  list(
    states = if (listed) nrow(a$full$states) else NA_integer_,
    transitions = if (listed) NROW(a$full$transitions) else NA_integer_,
    lumped_states = if (listed) nrow(a$lumped$states) else NA_integer_,
    components = a$leaves,
    note = a$note
  )
}

#' @export
print.pepa_model <- function(x, ...) {
  s <- summary(x)
  size <- if (is.na(s$states)) "state space too large to list" else
    sprintf("%d states, %d after lumping", s$states, s$lumped_states)
  cat(sprintf("<PEPA model: %s; components %s>\n", size, paste(s$components, collapse = ", ")))
  invisible(x)
}

#' States of a model
#'
#' One row per state with its long-run (steady-state) probability. With
#' `lumped = TRUE`, states that behave identically (Markovian bisimilar) are
#' merged, and `merged_states` says how many each row stands for.
#'
#' @param model A `pepa_model`.
#' @param lumped Merge equivalent states?
#' @return A data frame.
#' @export
pepa_states <- function(model, lumped = FALSE) {
  a <- pepa_analysis(model)
  full <- a$full$states
  full$steady_state <- a$steady_state
  if (!lumped) {
    return(data.frame(state = full$id, name = full$name, initial = full$initial,
                      lumped_state = full$block, steady_state = full$steady_state,
                      stringsAsFactors = FALSE))
  }
  blocks <- a$lumped$states
  steady <- tapply(full$steady_state, full$block, sum)
  data.frame(state = blocks$id, name = blocks$name, initial = blocks$initial,
             merged_states = blocks$size, steady_state = as.numeric(steady[as.character(blocks$id)]),
             stringsAsFactors = FALSE)
}

#' Transitions of a model
#'
#' @inheritParams pepa_states
#' @return A data frame with `source`, `target`, `action` and total `rate`
#'   (events per second).
#' @export
pepa_transitions <- function(model, lumped = FALSE) {
  t <- pepa_analysis(model)[[if (lumped) "lumped" else "full"]]$transitions
  if (length(t) == 0) {
    return(data.frame(source = integer(), target = integer(), action = character(), rate = numeric()))
  }
  as.data.frame(t, stringsAsFactors = FALSE)[, c("source", "target", "action", "rate")]
}

#' Local behaviour of each component
#'
#' @inheritParams pepa_states
#' @return A named list with one data frame per sequential component:
#'   `from`, `to`, `action`, `rate` and `passive` (rate set by a partner).
#' @export
pepa_components <- function(model) {
  comps <- pepa_analysis(model)$components
  out <- lapply(seq_len(nrow(comps)), function(i) {
    names <- comps$states[[i]]
    t <- comps$transitions[[i]]
    data.frame(from = names[t$source + 1], to = names[t$target + 1], action = t$action,
               rate = t$rate, passive = t$passive, stringsAsFactors = FALSE)
  })
  stats::setNames(out, comps$component)
}

#' Behaviour over time
#'
#' How the system evolves from its initial state: for each time (seconds) and
#' each component, the expected number of copies in each local state (for a
#' single component: the probability of being in it).
#'
#' @inheritParams pepa_states
#' @param times Times in seconds.
#' @param wide Return one column per component and state instead of a long table?
#' @return A data frame (`time`, `component`, `state`, `expected`), or a wide one.
#' @export
pepa_over_time <- function(model, times, wide = FALSE) {
  stopifnot(inherits(model, "pepa_model"), is.numeric(times))
  r <- pepa_call(list(op = "transient", source = model$source, times = I(as.numeric(times)),
                      max_states = model$max_states))
  if (!is.null(r$error)) pepa_abort(diagnostics_frame(r$error))
  occ <- r$ok$occupancy
  long <- do.call(rbind, lapply(seq_len(nrow(occ)), function(i) {
    data.frame(time = as.numeric(times), component = occ$component[i], state = occ$derivative[i],
               expected = as.numeric(occ$values[[i]]), stringsAsFactors = FALSE)
  }))
  long <- long[order(long$time, long$component, long$state), , drop = FALSE]
  rownames(long) <- NULL
  if (!wide) return(long)
  long$column <- paste0(long$component, ": ", long$state)
  wide_df <- stats::reshape(long[, c("time", "column", "expected")], idvar = "time",
                            timevar = "column", direction = "wide")
  names(wide_df) <- sub("^expected\\.", "", names(wide_df))
  rownames(wide_df) <- NULL
  wide_df
}

#' Plot behaviour over time
#'
#' A line chart of [pepa_over_time()] for one component, with base graphics.
#'
#' @inheritParams pepa_states
#' @param horizon Last time point in seconds.
#' @param points Number of time points.
#' @param component Component to plot (default: the first).
#' @return The plotted data, invisibly.
#' @export
pepa_plot_over_time <- function(model, horizon = 100, points = 101, component = NULL) {
  d <- pepa_over_time(model, seq(0, horizon, length.out = points))
  if (is.null(component)) component <- d$component[1]
  d <- d[d$component == component, , drop = FALSE]
  states <- unique(d$state)
  y <- sapply(states, function(s) d$expected[d$state == s])
  t <- unique(d$time)
  cols <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
  graphics::matplot(t, y, type = "l", lty = 1, lwd = 2, col = cols[seq_along(states)],
                    xlab = "time (s)", ylab = if (grepl("\\]$", component)) "expected copies" else "probability",
                    main = component, las = 1)
  graphics::legend("topright", legend = states, col = cols[seq_along(states)], lwd = 2, bty = "n")
  invisible(d)
}

#' Compare two models
#'
#' Four checks, from strongest to weakest notion of "the same":
#' `same_behaviour` (same actions and rates: Markovian bisimulation),
#' `same_branching` (rates ignored: bisimulation), `mutually_similar` (each
#' can mimic the other) and `b_mimics_a` (B can do everything A can).
#' When a check fails, `formula` describes something one model can do and
#' the other cannot: read `<a>phi` (printed as ⟨a⟩φ) as "can do a, then phi",
#' and `⟨a⟩=r φ` as "does a at total rate r per second into states where phi holds".
#'
#' @param a,b `pepa_model` objects or PEPA source text.
#' @param max_states State-space limit for each model.
#' @return A data frame of class `pepa_comparison` with columns `check`,
#'   `holds`, `formula`, `formula_holds_in` and `explanation`.
#' @export
pepa_compare <- function(a, b, max_states = 200000) {
  src <- function(m) if (inherits(m, "pepa_model")) m$source else m
  r <- pepa_call(list(op = "compare", a = src(a), b = src(b), max_states = max_states))
  if (!is.null(r$error)) {
    pepa_abort(rbind(diagnostics_frame(r$error$a), diagnostics_frame(r$error$b)), "comparison")
  }
  keys <- c(same_behaviour = "markovian_bisimilar", same_branching = "bisimilar",
            mutually_similar = "mutually_similar", b_mimics_a = "b_simulates_a")
  rows <- lapply(names(keys), function(name) {
    v <- r$ok[[keys[[name]]]]
    data.frame(check = name, holds = v$holds,
               formula = if (is.null(v$formula)) NA_character_ else v$formula,
               formula_holds_in = if (is.null(v$formula_holds_in)) NA_character_ else v$formula_holds_in,
               explanation = v$explanation, stringsAsFactors = FALSE)
  })
  structure(do.call(rbind, rows), class = c("pepa_comparison", "data.frame"), note = r$ok$note)
}

#' @export
print.pepa_comparison <- function(x, ...) {
  for (i in seq_len(nrow(x))) {
    mark <- if (x$holds[i]) "✓" else "✗"
    extra <- if (is.na(x$formula[i])) "" else sprintf(": %s holds in %s", x$formula[i], x$formula_holds_in[i])
    cat(sprintf("%s %s%s\n", mark, x$check[i], extra))
  }
  if (!is.null(attr(x, "note"))) cat("Note:", attr(x, "note"), "\n")
  invisible(x)
}

#' Compare two models up to causality
#'
#' Do the models have not only the same steps but the same dependencies
#' between them? Two workers watering and weeding independently allow the
#' same sequences of actions as one farmer doing both, even at the same
#' rates, but not the same dependencies. The check compares the models'
#' causal nets (history-preserving bisimulation); rates are ignored. Models
#' with functional rates have no causal semantics and are refused with an error.
#'
#' When the dependencies differ, `causal_witness` is a formula true of A and
#' false of B: read `⟨a ← 1,2⟩φ` as "can do an a that depends on the 1st and
#' 2nd step so far, then φ" and `⟨a ← ∅⟩φ` as "can do an a that depends on
#' none of the earlier steps, then φ".
#'
#' @inheritParams pepa_compare
#' @return A list of class `pepa_causal_comparison` with `interleaving_bisimilar`
#'   (same sequences of actions), `interleaving_witness`, `causal` (`"holds"`,
#'   `"violated"` or `"unknown"` when the search budget ran out),
#'   `causal_witness`, `nets` (a data frame with the `places`, `transitions` and
#'   `components` of each model's causal net) and `explanation`. Missing
#'   witnesses are `NA`.
#' @examples
#' workers <- "Waterer = (water, 1).Waterer; Weeder = (weed, 1).Weeder; Waterer || Weeder"
#' farmer <- "Farmer = (water, 1).Farmer + (weed, 1).Farmer; Farmer"
#' pepa_causal_compare(workers, farmer)
#' @export
pepa_causal_compare <- function(a, b, max_states = 200000) {
  src <- function(m) if (inherits(m, "pepa_model")) m$source else m
  r <- pepa_call(list(op = "causal", a = src(a), b = src(b), max_states = max_states))
  if (!is.null(r$error)) {
    pepa_abort(rbind(diagnostics_frame(r$error$a), diagnostics_frame(r$error$b)), "comparison")
  }
  na <- function(x) if (is.null(x)) NA_character_ else x
  nets <- as.data.frame(r$ok$nets)
  nets <- cbind(model = c("A", "B"), nets[, c("places", "transitions", "components")])
  structure(list(interleaving_bisimilar = r$ok$interleaving_bisimilar,
                 interleaving_witness = na(r$ok$interleaving_witness),
                 causal = r$ok$causal, causal_witness = na(r$ok$causal_witness),
                 nets = nets, explanation = r$ok$explanation),
            class = "pepa_causal_comparison")
}

#' @export
print.pepa_causal_comparison <- function(x, ...) {
  mark <- c(holds = "✓", violated = "✗", unknown = "?")[[x$causal]]
  extra <- if (is.na(x$causal_witness)) "" else sprintf(": %s holds in A", x$causal_witness)
  cat(sprintf("%s same dependencies (%s)%s\n", mark, x$causal, extra))
  cat(x$explanation, "\n")
  invisible(x)
}
