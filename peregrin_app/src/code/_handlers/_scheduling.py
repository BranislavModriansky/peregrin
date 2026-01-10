# debounce_throttle.py
import functools
import time
from typing import Callable, Optional, Any

from shiny import reactive


def DebounceCalc(delay_secs: int | float) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """
    Decorator: delay propagating invalidations until `delay_secs` have passed
    *since the last* upstream invalidation.
    """
    def wrapper(f: Callable[[], object]) -> Callable[[], object]:
        when = reactive.Value[Optional[float]](None)
        trigger = reactive.Value(0)

        @reactive.calc
        def cached():
            # Wrap f in a calc so Shiny handles dependency tracking/caching.
            return f()

        @reactive.effect(priority=102)
        def primer():
            """
            Each time `cached()` invalidates, push the deadline out by `delay_secs`.
            """
            try:
                # Touch cached() to register dependency, but ignore its value/errors.
                cached()
            except Exception:
                ...
            finally:
                when.set(time.time() + delay_secs)

        @reactive.effect(priority=101)
        def timer():
            """
            If the deadline is in the future, wait; if it has passed, fire `trigger`.
            """
            deadline = when()
            if deadline is None:
                return

            remaining = deadline - time.time()
            if remaining <= 0:
                with reactive.isolate():
                    when.set(None)
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(remaining)

        @reactive.calc
        @reactive.event(trigger, ignore_none=False)
        @functools.wraps(f)
        def debounced():
            return cached()

        return debounced

    return wrapper


def ThrottleCalc(delay_secs: int | float) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """
    Decorator: propagate at most once every `delay_secs` even if upstream
    invalidates more frequently (first event gets through ASAP; subsequent ones
    are held until the window elapses).
    """
    def wrapper(f: Callable[[], object]) -> Callable[[], object]:
        last_signaled = reactive.Value[Optional[float]](None)  # when upstream last invalidated
        last_triggered = reactive.Value[Optional[float]](None) # when we last let a value through
        trigger = reactive.Value(0)

        @reactive.calc
        def cached():
            return f()

        @reactive.effect(priority=102)
        def primer():
            """
            Record that upstream invalidated (even if `cached()` errors).
            """
            try:
                cached()
            except Exception:
                ...
            finally:
                last_signaled.set(time.time())

        @reactive.effect(priority=101)
        def timer():
            """
            If we've never triggered, or the window has elapsed, trigger now.
            Otherwise, schedule a wake-up for the remaining window.
            """
            # Read to form dependencies
            signaled = last_signaled()
            triggered = last_triggered()

            # Nothing to do until we've seen at least one upstream signal.
            if signaled is None:
                return

            now = time.time()
            if triggered is None:
                # First-ever trigger goes through immediately.
                last_triggered.set(now)
                with reactive.isolate():
                    trigger.set(trigger() + 1)
                return

            elapsed = now - triggered
            if elapsed >= delay_secs:
                last_triggered.set(now)
                with reactive.isolate():
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(delay_secs - elapsed)

        @reactive.calc
        @reactive.event(trigger, ignore_none=False)
        @functools.wraps(f)
        def throttled():
            return cached()

        return throttled

    return wrapper


def _unwrap_effect_callable(obj: Any) -> Callable[[], object]:
    """
    DebounceEffect must receive a plain callable. If a Shiny Effect_ object is
    mistakenly passed (e.g., by stacking @reactive.effect), try to unwrap to the
    underlying function; otherwise raise a helpful error.
    """
    if callable(obj):
        return obj

    # Best-effort unwrapping for Shiny Effect_ objects (internal API; may change).
    inner = getattr(obj, "_fn", None)
    if callable(inner):
        return inner

    raise TypeError(
        "DebounceEffect must decorate a plain function (do not stack with @reactive.effect). "
        f"Got non-callable object of type: {type(obj)!r}"
    )


def DebounceEffect(delay_secs: int | float) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """
    Decorator: debounce an effect-like function.

    IMPORTANT: Use on an undecorated function. Do NOT also apply @reactive.effect.

    Example:
        @DebounceEffect(0.25)
        def do_stuff():
            ...
    """
    def wrapper(f: Callable[[], object]) -> Callable[[], object]:
        f_callable = _unwrap_effect_callable(f)

        # Per-effect state (do NOT share across all wrapped effects).
        when = reactive.Value[Optional[float]](None)
        trigger = reactive.Value(0)

        @reactive.effect(priority=102)
        def primer():
            """
            Each time upstream invalidates, push the deadline out by `delay_secs`.
            NOTE: This calls the function to establish reactive dependencies.
            """
            try:
                f_callable()
            except Exception:
                ...
            finally:
                when.set(time.time() + delay_secs)

        @reactive.effect(priority=101)
        def timer():
            deadline = when()
            if deadline is None:
                return

            remaining = deadline - time.time()
            if remaining <= 0:
                with reactive.isolate():
                    when.set(None)
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(remaining)

        @reactive.effect
        @reactive.event(trigger, ignore_none=False)
        @functools.wraps(f_callable)
        def debounced_effect():
            f_callable()

        return debounced_effect

    return wrapper