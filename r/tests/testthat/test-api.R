cycle <- "r = 1;
P1 = (start, r).P2;
P2 = (run, r).P3;
P3 = (stop, r).P1;
"
herd <- "r_disease = 0.02;
Healthy = (disease, r_disease).Sick;
Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
Dead = 0;
Healthy[3]"
p1_exact <- function(t) (1 + 2 * exp(-1.5 * t) * cos(sqrt(3) / 2 * t)) / 3

test_that("state space, lumping and steady state", {
  m <- pepa_model(paste0(cycle, "P1 || P1"))
  s <- summary(m)
  expect_equal(s$states, 9)
  expect_equal(s$lumped_states, 6)
  st <- pepa_states(m)
  expect_named(st, c("state", "name", "initial", "lumped_state", "steady_state"))
  expect_equal(sum(st$steady_state), 1, tolerance = 1e-9)
  lumped <- pepa_states(m, lumped = TRUE)
  expect_equal(nrow(lumped), 6)
  expect_equal(sum(lumped$merged_states), 9)
  expect_equal(sum(lumped$steady_state), 1, tolerance = 1e-9)
  expect_named(pepa_transitions(m), c("source", "target", "action", "rate"))
})

test_that("behaviour over time matches the closed form", {
  m <- pepa_model(paste0(cycle, "P1[2]"))
  d <- pepa_over_time(m, c(0, 1))
  p1 <- d$expected[d$state == "P1" & d$time == 1]
  expect_equal(p1, 2 * p1_exact(1), tolerance = 1e-8)
  w <- pepa_over_time(m, c(0, 1), wide = TRUE)
  expect_named(w, c("time", "P1[2]: P1", "P1[2]: P2", "P1[2]: P3"))
  expect_equal(w[["P1[2]: P1"]][1], 2)
})

test_that("components are listed per sequential component", {
  comps <- pepa_components(pepa_model(herd))
  expect_named(comps, "Healthy")
  expect_setequal(comps$Healthy$action, c("disease", "healing", "death"))
})

test_that("errors carry positions", {
  err <- tryCatch(pepa_model("P = (a, r).P;\nP"), pepa_error = identity)
  expect_s3_class(err, "pepa_error")
  expect_equal(err$diagnostics$line, 1)
  expect_equal(err$diagnostics$column, 9)
  expect_match(conditionMessage(err), "unknown rate parameter")
  expect_equal(nrow(pepa_check(paste0(cycle, "P1"))), 0)
  expect_equal(pepa_check("P = (a, 1).P; Q = (b, 1).Q; P <b> Q")$severity, "warning")
})

test_that("comparison explains the difference", {
  no_recovery <- "Healthy = (disease, 0.02).Sick; Sick = (death, 0.01).Dead; Dead = 0; Healthy[3]"
  cmp <- pepa_compare(pepa_model(herd), no_recovery)
  expect_s3_class(cmp, "pepa_comparison")
  row <- cmp[cmp$check == "b_mimics_a", ]
  expect_false(row$holds)
  expect_equal(row$formula, "⟨disease⟩⟨healing⟩tt")
  expect_equal(row$formula_holds_in, "A")
  same <- pepa_compare(paste0(cycle, "P1 || P1"), paste0(cycle, "P1[2]"))
  expect_true(all(same$holds))
})

test_that("large populations: listed only when small, over time always for independent copies", {
  big <- pepa_model(paste0(cycle, "P1[200]"), max_states = 50)
  expect_true(is.na(summary(big)$states))
  expect_equal(sum(pepa_over_time(big, 1)$expected), 200, tolerance = 1e-9)
  coupled <- pepa_model(paste0(cycle, "S = (start, 5).S; P1[200] <start> S"), max_states = 50)
  expect_error(pepa_over_time(coupled, 1), "exceeds the limit")
})

garden <- paste(readLines(test_path("../../../examples/garden/tomatoes.pepa")), collapse = "\n")
season <- pepa_read_observations(test_path("../../../examples/garden/observations.csv"))

test_that("garden: borderline in June, off plan by July", {
  plan <- pepa_model(garden)
  june <- pepa_check_observations(plan, season[1:4, ], start = "15/05/2026")
  expect_true(june$on_track)
  expect_equal(june$table$status, c("on_track", "on_track", "above_plan", "on_track"))
  july <- pepa_check_observations(plan, season, start = as.Date("2026-05-15"), unit = "days")
  expect_false(july$on_track)
  expect_match(july$explanation, "Ripe on 17/07/2026")
})

test_that("fitting finds the slow stage and the fitted model differs from the plan", {
  plan <- pepa_model(garden)
  fit <- pepa_fit(plan, season, "veg_days", start = "15/05/2026")
  expect_gt(fit$params$fitted, 35)
  expect_lt(fit$params$fitted, 55)
  expect_true(fit$check$on_track)
  cmp <- pepa_compare(plan, fit$model)
  expect_false(cmp$holds[cmp$check == "same_behaviour"])
  expect_match(attr(cmp, "note"), "one copy")
})

test_that("forecast and plot", {
  plan <- pepa_model(garden)
  f <- pepa_forecast(plan, "Flowering", c(0, 21), start = "15/05/2026")
  expect_equal(f$date, c("15/05/2026", "05/06/2026"))
  path <- tempfile(fileext = ".png")
  grDevices::png(path)
  report <- pepa_plot_observations(plan, season, start = "15/05/2026")
  grDevices::dev.off()
  expect_true(file.exists(path))
  expect_false(report$on_track)
  expect_error(pepa_check_observations(plan, data.frame(date = "05/06/2026", state = "Blooming", count = 1),
                                       start = "15/05/2026"), "no state `Blooming`")
})
