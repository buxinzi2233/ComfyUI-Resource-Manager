use pyo3::prelude::*;

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq)]
struct IdleState {
    idle_since_ms: Option<u64>,
    gpu_dispatched: bool,
    full_dispatched: bool,
}

#[pymethods]
impl IdleState {
    #[new]
    fn new(idle_since_ms: Option<u64>, gpu_dispatched: bool, full_dispatched: bool) -> Self {
        Self {
            idle_since_ms,
            gpu_dispatched,
            full_dispatched,
        }
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
struct Decision {
    state: IdleState,
    action: Option<String>,
}

fn decide(
    state: &IdleState,
    now_ms: u64,
    busy: bool,
    gpu_timeout_ms: Option<u64>,
    full_timeout_ms: Option<u64>,
) -> Decision {
    if busy {
        return Decision {
            state: IdleState::new(None, false, false),
            action: None,
        };
    }
    let since = state.idle_since_ms.unwrap_or(now_ms);
    let elapsed = now_ms.saturating_sub(since);
    let mut next = state.clone();
    next.idle_since_ms = Some(since);
    let action = if !next.full_dispatched && full_timeout_ms.is_some_and(|limit| elapsed >= limit) {
        next.full_dispatched = true;
        next.gpu_dispatched = true;
        Some("release_models".to_owned())
    } else if !next.gpu_dispatched && gpu_timeout_ms.is_some_and(|limit| elapsed >= limit) {
        next.gpu_dispatched = true;
        Some("unload_gpu".to_owned())
    } else {
        None
    };
    Decision {
        state: next,
        action,
    }
}

#[pyfunction]
fn idle_decision(
    state: &IdleState,
    now_ms: u64,
    busy: bool,
    gpu_timeout_ms: Option<u64>,
    full_timeout_ms: Option<u64>,
) -> Decision {
    decide(state, now_ms, busy, gpu_timeout_ms, full_timeout_ms)
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq)]
struct ModelIdentity {
    path: String,
    size: u64,
    mtime_ns: u64,
    role: String,
    loading_config: String,
}

#[pymethods]
impl ModelIdentity {
    #[new]
    fn new(path: String, size: u64, mtime_ns: u64, role: String, loading_config: String) -> Self {
        Self {
            path,
            size,
            mtime_ns,
            role,
            loading_config,
        }
    }

    fn matches(&self, other: &ModelIdentity) -> bool {
        self == other
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<IdleState>()?;
    module.add_class::<Decision>()?;
    module.add_class::<ModelIdentity>()?;
    module.add_function(wrap_pyfunction!(idle_decision, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_lifecycle() {
        let initial = IdleState::new(None, false, false);
        let started = decide(&initial, 100, false, Some(10), Some(20));
        assert_eq!(initial.idle_since_ms, None);
        assert_eq!(started.state.idle_since_ms, Some(100));
        let gpu = decide(&started.state, 110, false, Some(10), Some(20));
        assert_eq!(gpu.action.as_deref(), Some("unload_gpu"));
        assert_eq!(
            decide(&gpu.state, 111, false, Some(10), Some(20)).action,
            None
        );
        let full = decide(&gpu.state, 120, false, Some(10), Some(20));
        assert_eq!(full.action.as_deref(), Some("release_models"));
        assert_eq!(
            decide(&full.state, 200, false, Some(10), Some(20)).action,
            None
        );
        assert_eq!(
            decide(&full.state, 201, true, Some(10), Some(20)).state,
            initial
        );
    }

    #[test]
    fn full_release_supersedes_gpu_and_disabled_stays_disabled() {
        let state = IdleState::new(Some(0), false, false);
        assert_eq!(
            decide(&state, 100, false, Some(10), Some(10))
                .action
                .as_deref(),
            Some("release_models")
        );
        assert_eq!(decide(&state, 100, false, None, None).action, None);
        assert_eq!(
            decide(&state, 100, false, None, Some(10)).action.as_deref(),
            Some("release_models")
        );
    }

    #[test]
    fn identities_include_file_revision_and_loading_config() {
        let first = ModelIdentity::new("/a".into(), 1, 2, "diffusion_models".into(), "fp32".into());
        let replaced =
            ModelIdentity::new("/a".into(), 1, 3, "diffusion_models".into(), "fp32".into());
        let different_precision =
            ModelIdentity::new("/a".into(), 1, 2, "diffusion_models".into(), "fp16".into());
        assert!(first.matches(&first.clone()));
        assert!(!first.matches(&replaced));
        assert!(!first.matches(&different_precision));
    }
}
