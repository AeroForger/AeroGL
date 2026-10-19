//! Small helpers for splitting pixel work across threads.
//! Made by AeroForger

use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Upper bound on worker threads. Beyond a point the per-band work is tiny and
/// the repeated vertex setup starts to cost more than the parallel gain.
const MAX_THREADS: usize = 16;

/// Test-only override, so a test can compare one-thread and many-thread output
/// in the same process without touching the environment.
#[cfg(test)]
static TEST_LIMIT: AtomicUsize = AtomicUsize::new(0);

/// The machine's worker count, honoring the `AEROGL_THREADS` override.
///
/// The value is read once per process, so changing the variable needs a
/// restart. That is fine for benchmarking and keeps the hot path free of
/// environment lookups.
pub(crate) fn thread_limit() -> usize {
    #[cfg(test)]
    {
        let t = TEST_LIMIT.load(Ordering::Relaxed);
        if t > 0 {
            return t;
        }
    }
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("AEROGL_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
            .min(MAX_THREADS)
    })
}

/// Forces a worker count for tests. `0` restores the automatic value.
#[cfg(test)]
pub(crate) fn set_thread_limit_for_tests(n: usize) {
    TEST_LIMIT.store(n, Ordering::Relaxed);
}

/// Worker count for `pixels` pixels, or 1 when the job is too small to be
/// worth the thread startup.
pub(crate) fn threads_for(pixels: usize, min_pixels: usize) -> usize {
    if pixels < min_pixels {
        1
    } else {
        thread_limit()
    }
}
