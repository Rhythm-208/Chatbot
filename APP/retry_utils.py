import time
import random

def call_with_retry(func, *args, max_retries=3, base_delay=5, max_delay=60, **kwargs):
    """
        Calls func(*args, **kwargs), retrying on failure with exponential
        backoff + jitter. Raises the last exception if all attempts fail.
     """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            last_exception = e
            error_text = str(e).lower()
            is_rate_limit = "429" in str(e) or "quota" in error_text or "rate limit" in error_text

            delay = min(base_delay * (2 ** attempt), max_delay)
            if is_rate_limit:
                delay = max(delay, 20)

                delay += random.uniform(0, 2)

                if attempt < max_retries - 1:
                    print(f"[retry] Attempt {attempt + 1}/{max_retries} failed "
                          f"({'rate limit' if is_rate_limit else 'error'}): {e}. "
                          f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

    raise last_exception





