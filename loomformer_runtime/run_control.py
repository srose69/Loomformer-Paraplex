from __future__ import annotations

import signal
import sys
import threading
from typing import Optional

from .distributed import ddp_is_main

class _GracefulInterrupt:
    def __init__(self) -> None:
        self.requested = False
        self._count = 0
        self._original = None

    def _handler(self, signum, frame) -> None:
        self._count += 1
        if self._count >= 2 and self._original is not None:
            signal.signal(signal.SIGINT, self._original)
            self._original(signum, frame)
            return
        self.requested = True

    def __enter__(self) -> "_GracefulInterrupt":
        self._original = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *exc) -> None:
        if self._original is not None:
            signal.signal(signal.SIGINT, self._original)


class _RunpointWatcher:

    def __init__(self) -> None:
        self.requested = threading.Event()
        self._old_termios = None
        self._stop = threading.Event()
        self._active = False
        self._thread: Optional[threading.Thread] = None

    def _reader_loop(self) -> None:
        import select
        while not self._stop.is_set():
            try:
                # Poll with a short timeout instead of a blocking read(1) --
                # a blocking read can only notice _stop on its NEXT keystroke,
                # which is exactly the race that let this thread steal the
                # first character of a y/N answer meant for _handle_interrupt's
                # input(). Polling means pause() actually stops this promptly
                # (~50ms), not "whenever the user happens to type again".
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
            except Exception:
                return
            if not ch:
                return
            if ch.lower() == "s":
                self.requested.set()

    def __enter__(self) -> "_RunpointWatcher":
        # Every local torchrun worker inherits the same controlling terminal.
        # Letting multiple ranks save/modify termios races on restore: a later
        # rank can save rank 0's cbreak/no-echo state as its "original" state
        # and leave the shell broken after an exception (notably CUDA OOM).
        if not ddp_is_main():
            return self
        try:
            import termios
            import tty
        except ImportError:
            return self  # e.g. Windows -- no-op, training is unaffected
        if not sys.stdin.isatty():
            return self  # quiet mode / piped / non-interactive -- no-op by design
        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            no_echo = termios.tcgetattr(sys.stdin)
            no_echo[3] &= ~termios.ECHO   # cbreak alone still echoes keystrokes;
                                          # turn that off so pressing 's' doesn't
                                          # leave a stray "s" in the training log
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, no_echo)
        except Exception:
            self._old_termios = None
            return self  # any terminal weirdness at all -- no-op, never crash training
        self._active = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def pause(self) -> None:
        """Stop the key reader and restore the terminal's original settings."""
        if not self._active:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def resume(self) -> None:
        """Restart the key reader in non-echoing cbreak mode after ``pause``."""
        if not self._active:
            return
        try:
            import termios
            import tty
            tty.setcbreak(sys.stdin.fileno())
            no_echo = termios.tcgetattr(sys.stdin)
            no_echo[3] &= ~termios.ECHO
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, no_echo)
        except Exception:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
        self._active = False

    def consume(self) -> bool:
        if self.requested.is_set():
            self.requested.clear()
            return True
        return False

__all__ = ("_GracefulInterrupt", "_RunpointWatcher")
