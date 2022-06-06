from result import Result, Ok, Err
from beartype import beartype
from ki.monadic import monadic

@monadic
@beartype
def function() -> Result[int, Exception]:
    Err(ValueError("Bad")).unwrap()
    return 0

a = function()
print(a)
