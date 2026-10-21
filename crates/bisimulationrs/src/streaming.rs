//! Verdict streams: feed observations in, get a verdict out whenever it
//! changes.
//!
//! The adapters drive a [`PrefixChecker`], which updates its three games in
//! place per event (see `bisim_refine::PrefixChecker`). Two shapes:
//!
//! - [`verdict_iter`]: any iterator of [`Event`]s, including an
//!   `std::sync::mpsc::Receiver` (`rx.into_iter()`), blocking as the source
//!   blocks;
//! - [`VerdictStream`]: any `futures_core::Stream` of events, as a stream of
//!   changed verdicts. It polls the source with the caller's context, so the
//!   source's waker reaches the caller's executor; no executor is bundled.
//!
//! Each item is a verdict that differs in kind from the previous one (the
//! first verdict is always reported), or the error for a transition from a
//! state already declared complete; the checker then stays as it was.
//!
//! ```
//! use bisimulationrs::streaming::{Event, verdict_iter};
//! use bisimulationrs::refine::PrefixChecker;
//!
//! let events = [
//!     Event::Left("p0", 'a', "p1"),
//!     Event::CompleteLeft("p0"),
//!     Event::CompleteLeft("p1"),
//!     Event::CompleteRight("q0"), // the right side never does `a`
//! ];
//! let changes: Vec<_> = verdict_iter(PrefixChecker::new("p0", "q0"), events).collect();
//! let last = changes.last().unwrap().as_ref().unwrap();
//! assert!(last.bisimilar.is_violated());
//! ```

use std::hash::Hash;
use std::pin::Pin;
use std::task::{Context, Poll};

use bisim_core::{ComparisonVerdict, Formula};
use bisim_refine::{PrefixChecker, PrefixError};

/// One observation about the two systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event<S, L> {
    /// A transition `(source, label, target)` of the left system.
    Left(S, L, S),
    /// A transition of the right system.
    Right(S, L, S),
    /// Every transition out of this left state has been observed.
    CompleteLeft(S),
    /// Every transition out of this right state has been observed.
    CompleteRight(S),
}

/// A changed verdict, or a rejected observation.
pub type VerdictChange<S, L> = Result<ComparisonVerdict<Formula<L, bool>>, PrefixError<S>>;

/// Applies one event; reports a changed verdict or an error.
fn apply<S: Eq + Hash + Clone, L: Eq + Hash + Clone>(
    checker: &mut PrefixChecker<S, L>,
    event: Event<S, L>,
) -> Option<VerdictChange<S, L>> {
    let applied = match event {
        Event::Left(s, l, t) => checker.observe_left([(s, l, t)]),
        Event::Right(s, l, t) => checker.observe_right([(s, l, t)]),
        Event::CompleteLeft(s) => {
            checker.complete_left(s);
            Ok(())
        }
        Event::CompleteRight(s) => {
            checker.complete_right(s);
            Ok(())
        }
    };
    match applied {
        Err(e) => Some(Err(e)),
        Ok(()) => checker.poll_change().map(Ok),
    }
}

/// The verdict changes caused by `events`, in order.
pub fn verdict_iter<S, L, I>(
    checker: PrefixChecker<S, L>,
    events: I,
) -> impl Iterator<Item = VerdictChange<S, L>>
where
    S: Eq + Hash + Clone,
    L: Eq + Hash + Clone,
    I: IntoIterator<Item = Event<S, L>>,
{
    let mut checker = checker;
    events
        .into_iter()
        .filter_map(move |event| apply(&mut checker, event))
}

/// A stream of verdict changes over a stream of [`Event`]s.
pub struct VerdictStream<S, L, St> {
    checker: PrefixChecker<S, L>,
    events: St,
}

impl<S, L, St> std::fmt::Debug for VerdictStream<S, L, St>
where
    S: Eq + Hash + Clone,
    L: Eq + Hash + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerdictStream")
            .field("checker", &self.checker)
            .finish_non_exhaustive()
    }
}

impl<S, L, St> VerdictStream<S, L, St> {
    /// Drives `checker` with `events`.
    pub fn new(checker: PrefixChecker<S, L>, events: St) -> Self {
        Self { checker, events }
    }

    /// The checker, for a verdict at any time.
    pub fn checker(&self) -> &PrefixChecker<S, L> {
        &self.checker
    }
}

impl<S, L, St> futures_core::Stream for VerdictStream<S, L, St>
where
    S: Eq + Hash + Clone + Unpin,
    L: Eq + Hash + Clone + Unpin,
    St: futures_core::Stream<Item = Event<S, L>> + Unpin,
{
    type Item = VerdictChange<S, L>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            // The caller's context goes to the source: its waker is the
            // caller's.
            match Pin::new(&mut this.events).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(event)) => {
                    if let Some(change) = apply(&mut this.checker, event) {
                        return Poll::Ready(Some(change));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_core::Stream;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Wake, Waker};

    /// Yields `Pending` once (keeping the waker) before each item.
    struct Gated {
        items: Vec<Event<&'static str, char>>,
        ready: bool,
        waker: Arc<Mutex<Option<Waker>>>,
    }

    impl Stream for Gated {
        type Item = Event<&'static str, char>;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            if !self.ready {
                self.ready = true;
                *self.waker.lock().unwrap() = Some(cx.waker().clone());
                return Poll::Pending;
            }
            self.ready = false;
            Poll::Ready(if self.items.is_empty() {
                None
            } else {
                Some(self.items.remove(0))
            })
        }
    }

    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn stream_passes_the_callers_waker_and_reports_changes() {
        let slot = Arc::new(Mutex::new(None));
        let events = Gated {
            items: vec![
                Event::Left("p0", 'a', "p1"),
                Event::CompleteLeft("p0"),
                Event::CompleteLeft("p1"),
                Event::CompleteRight("q0"),
            ],
            ready: false,
            waker: slot.clone(),
        };
        let mut s = VerdictStream::new(PrefixChecker::new("p0", "q0"), events);
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(count.clone());
        let mut cx = Context::from_waker(&waker);
        let mut changes = Vec::new();
        for _ in 0..100 {
            match Pin::new(&mut s).poll_next(&mut cx) {
                Poll::Pending => {
                    // The source kept our waker; waking it reaches us.
                    slot.lock().unwrap().take().expect("waker kept").wake();
                }
                Poll::Ready(Some(c)) => changes.push(c.unwrap()),
                Poll::Ready(None) => break,
            }
        }
        assert!(
            count.0.load(Ordering::SeqCst) > 0,
            "the caller's waker was used"
        );
        assert_eq!(changes.len(), 2, "holds-so-far, then violated");
        assert!(changes[1].bisimilar.is_violated());
    }

    #[test]
    fn iterator_reports_rejected_events_and_continues() {
        let (tx, rx) = std::sync::mpsc::channel();
        for e in [
            Event::CompleteLeft("p0"),
            Event::Left("p0", 'a', "p1"), // p0 is complete: rejected
            Event::CompleteRight("q0"),
        ] {
            tx.send(e).unwrap();
        }
        drop(tx);
        let out: Vec<_> = verdict_iter(PrefixChecker::new("p0", "q0"), rx).collect();
        assert!(out.iter().any(|c| c.is_err()));
        let last = out.last().unwrap().as_ref().unwrap();
        assert!(matches!(last.bisimilar, bisim_core::Verdict::Holds));
    }
}
