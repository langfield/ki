"""A function decorator for use with functions returning ``result.Result``."""
import random
import functools
import itertools
from result import Ok, Err
from beartype import beartype
from beartype.typing import Callable, TypeVar, NoReturn

from beartype.roar._roarexc import (
    BeartypeCallHintReturnViolation,
    _BeartypeCallHintPepRaiseException,
)
from beartype._decor._error._errorsleuth import CauseSleuth
from beartype._util.text.utiltextlabel import (
    prefix_callable_decorated_return_value,
)
from beartype._util.hint.utilhinttest import die_unless_hint
from beartype._util.text.utiltextmunge import suffix_unless_suffixed

# pylint: disable=invalid-name

T = TypeVar("T")

PITH_NAME = "return"


@beartype
def raise_if_return_type_exception(
    func: Callable[[...], T],
    exception_prefix: str,
    pith_value: object,
    hint: object,
    helper: str,
) -> None:
    """
    Typecheck the return value of a function decorated with ``@safe``.

    Raise an error if ``pith_value`` doesn't match the type specified by
    ``hint``. This is a snippet copied from the internal implementation of
    beartype, and so it should be pinned to a version to avoid breakage when
    the private API inevitably changes.
    """

    # If this is *NOT* the PEP 484-compliant "typing.NoReturn" type hint
    # permitted *ONLY* as a return annotation, this is a standard type hint
    # generally supported by both parameters and return values. In this case...
    if hint is not NoReturn:
        # If type hint is *NOT* a supported type hint, raise an exception.
        die_unless_hint(hint=hint, exception_prefix=exception_prefix)
        # Else, this type hint is supported.

    # Human-readable string describing the failure of this pith to satisfy this
    # hint if this pith fails to satisfy this hint *OR* "None" otherwise (i.e.,
    # if this pith satisfies this hint).
    exception_cause = CauseSleuth(
        func=func,
        pith=pith_value,
        hint=hint,
        cause_indent="",
        exception_prefix=exception_prefix,
        random_int=random.getrandbits(32),
    ).get_cause_or_none()

    # If this pith does *NOT* satisfy this hint...
    if exception_cause:
        # This failure suffixed by a period if *NOT* yet suffixed by a period.
        exception_cause_suffixed = suffix_unless_suffixed(
            text=exception_cause, suffix="."
        )

        # Raise an exception of the desired class embedding this cause.
        raise BeartypeCallHintReturnViolation(  # type: ignore[misc]
            f"{exception_prefix}violates {helper} type hint {repr(hint)}, as "
            f"{exception_cause_suffixed}"
        )


@beartype
def safe(func: Callable[[...], T]) -> Callable[[...], T]:
    """A function decorator to chain functions that return a ``result.Result``."""

    @functools.wraps(func)
    def decorated(*args, **kwargs) -> T:
        """The decorated version of ``func``."""
        # If any of the arguments are Err, return early.
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, Err):
                return arg

        # Unpack arguments of type ``Ok`` passed to ``func()``. We pass the
        # modified versions of ``args``, ``kwargs`` to ``func()`` because if we
        # do not do this unpacking step, functions that are @beartype decorated
        # underneath the @safe decorator will ALWAYS fail to typecheck, since
        # they are expected whatever type hint the user provided, but they are
        # (possibly) receiving ``OkErr`` types.
        unpack = lambda arg: arg.value if isinstance(arg, Ok) else arg
        args = tuple(map(unpack, args))
        kwargs = {key: unpack(arg) for key, arg in kwargs.items()}

        # Call function with the OkErr-unpacked ``args`` and ``kwargs``.
        result = func(*args, **kwargs)

        # Human-readable label describing the unpacked return value.
        exception_prefix: str = prefix_callable_decorated_return_value(
            func=func, return_value=result.value
        )

        # If the return value is unannotated, raise an exception.
        if PITH_NAME not in func.__annotations__:
            raise _BeartypeCallHintPepRaiseException(f"{exception_prefix}unannotated.")

        # Unfold the return type annotation of ``func``, extracting the ``Ok``
        # return type hint and the ``Err`` return type hint.
        ok_hint, err_hint = func.__annotations__[PITH_NAME].__args__
        ok_ret_hint = ok_hint.__args__[0]
        err_ret_hint = err_hint.__args__[0]

        # Check return type, raising an error if the check fails. These calls
        # are no-ops otherwise.
        if isinstance(result, Ok):
            raise_if_return_type_exception(
                func, exception_prefix, result.value, ok_ret_hint, "Ok"
            )
        elif isinstance(result, Err):
            raise_if_return_type_exception(
                func, exception_prefix, result.value, err_ret_hint, "Err"
            )

        # Return the un-unpacked result, since this is what the user expects
        # (``func()`` is decorated with @safe, and so it is intended to return
        # a value of type ``result.Result``).
        return result

    return decorated
