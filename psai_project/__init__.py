"""Sequential local project analysis; model stages run in separate processes."""


def run_project(*args, **kwargs):
    from .runtime import run_project as run
    return run(*args, **kwargs)


__all__ = ['run_project']
