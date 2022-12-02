{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import Path (Path, Abs, Rel, File, Dir, toFilePath, (</>))
import Path.IO (resolveFile', resolveDir', ensureDir, listDir, doesFileExist)
import Data.Text (Text)
import Data.Typeable (Typeable)
import Data.Digest.Pure.MD5 (md5)
import Text.Printf (printf)
import Control.Monad.IO.Class (MonadIO)

import qualified Path.Internal
import qualified Data.Text as T
import qualified Data.ByteString.Lazy as LB


class AbsIO a

-- Extra base types for `Path`.
data Empty deriving (Typeable)
data Extant deriving (Typeable)
data Missing deriving (Typeable)

instance AbsIO Empty
instance AbsIO Extant
instance AbsIO Missing

-- Extra type types for `Path`.
data Leaf deriving (Typeable)
data Link deriving (Typeable)
data Pseudo deriving (Typeable)
data WindowsLink deriving (Typeable)


-- Ensure a directory is empty, creating it if necessary.
ensureEmpty :: Path Abs Dir -> IO (Maybe (Path Empty Dir))
ensureEmpty dir = do
  ensureDir dir
  contents <- listDir dir
  return $
    case contents of
      ([], []) -> Just $ Path.Internal.Path (toFilePath dir)
      _        -> Nothing


-- Ensure a file exists, creating it if necessary.
ensureFile :: Path Abs File -> IO (Path Extant File)
ensureFile file = do
  exists <- doesFileExist file
  case exists of
    True -> return $ Path.Internal.Path (toFilePath file)
    False -> do
      appendFile (toFilePath file) ""
      return (Path.Internal.Path (toFilePath file))


getExtantFile :: Path Abs File -> IO (Maybe (Path Extant File))
getExtantFile file = do
  exists <- doesFileExist file
  case exists of
    True -> return $ Just $ Path.Internal.Path (toFilePath file)
    False -> return Nothing


absify :: AbsIO a => Path a b -> Path Abs b
absify (Path.Internal.Path s) = Path.Internal.Path s


-- Parse the collection and target directory, then call `continueClone`.
clone :: String -> String -> IO ()
clone colPath targetPath = do
  colFile <- resolveFile' colPath
  targetDir <- resolveDir' targetPath
  maybeColFile <- getExtantFile colFile
  maybeTargetDir <- ensureEmpty targetDir
  case (maybeColFile, maybeTargetDir) of
    (Nothing, _) -> printf "fatal: collection file '%s' does not exist" (show colFile)
    (_, Nothing) -> printf "fatal: targetdir '%s' not empty" (show targetDir)
    (Just colFile, Just targetDir) -> continueClone colFile targetDir


continueClone :: Path Extant File -> Path Empty Dir -> IO ()
continueClone colFile targetDir = do
  colFileContents <- LB.readFile (toFilePath colFile)
  let colFileMD5 = md5 colFileContents
  return ()
  where
    kiDir = ensureEmpty (absify targetDir </> Path.Internal.Path ".ki")
    mediaDir = ensureEmpty (absify targetDir </> Path.Internal.Path ".media")


main :: IO ()
main = print "Hello"
