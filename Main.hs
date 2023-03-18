module Main (main) where

import Ki (clone, getArgs)

main :: IO ()
main = do
  args <- getArgs
  case args of
    [col, tgt] -> clone col tgt
    _ -> putStrLn "Usage: ki <collection.anki2> <target>"
